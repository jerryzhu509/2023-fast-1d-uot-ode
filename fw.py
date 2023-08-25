import numpy as np
import numba
import time

@numba.jit(nopython=True)
def solve_ot(a, b, x, y, p):
    """Computes the 1D Optimal Transport between two histograms.

    _Important: one should have np.sum(a)=np.sum(b)._

    _Important:_ x and y needs to be sorted.

    Parameters
    ----------
    a: vector of length n with positive entries

    b: vector of length m with positive entries

    x: vector of real of length n

    y: vector of real of length m

    p: real, should >= 1


    Returns
    ----------
    I: vector of length q=n+m-1 of increasing integer in {0,...,n-1}

    J: vector of length q of increasing integer in {0,...,m-1}

    P: vector of length q of positive values of length q

    f: dual vector of length n

    g: dual vector of length m

    cost: (dual) OT cost
        sum a_i f_i + sum_j b_j f_j
        It should be equal to the primal cost
        = sum_k |x(i)-y(j)|^p where i=I(k), j=J(k)
    """
    n = len(a)
    m = len(b)
    q = m + n - 1
    a1 = a.copy()
    b1 = b.copy()
    I = np.zeros(q).astype(numba.int64)
    J = np.zeros(q).astype(numba.int64)
    P = np.zeros(q)
    f = np.zeros(n)
    g = np.zeros(m)
    g[0] = np.abs(x[0] - y[0]) ** p
    for k in range(q - 1):
        i = I[k]
        j = J[k]
        if (a1[i] < b1[j]) and (i < n - 1):
            I[k + 1] = i + 1
            J[k + 1] = j
            f[i + 1] = np.abs(x[i + 1] - y[j]) ** p - g[j]
        elif (a1[i] > b1[j]) and (j < m - 1):
            I[k + 1] = i
            J[k + 1] = j + 1
            g[j + 1] = np.abs(x[i] - y[j + 1]) ** p - f[i]
        elif i == n - 1:
            I[k + 1] = i
            J[k + 1] = j + 1
            g[j + 1] = np.abs(x[i] - y[j + 1]) ** p - f[i]
        elif j == m - 1:
            I[k + 1] = i + 1
            J[k + 1] = j
            f[i + 1] = np.abs(x[i + 1] - y[j]) ** p - g[j]
        t = min(a1[i], b1[j])
        P[k] = t
        a1[i] = a1[i] - t
        b1[j] = b1[j] - t
    P[k + 1] = max(a1[-1], b1[-1])  # remaining mass
    cost = np.sum(f * a) + np.sum(g * b)
    return I, J, P, f, g, cost


def logsumexp(f, a, stable_lse=True):
    """
    Computes the logsumexp operation, in stable form or not

    Parameters
    ----------
    f: numpy array of size n
    dual potential

    a: numpy array of size n with positive entries
    weights of measure

    stable_lse: bool
    If true, computes the logsumexp in stable form

    Return
    ------
    Float
    """
    if not stable_lse:
        return np.log(np.sum(a * np.exp(f)))
    else:
        xm = np.amax(f + np.log(a))
        return xm + np.log(np.sum(np.exp(f + np.log(a) - xm)))


def rescale_potentials(f, g, a, b, rho1, rho2=None, stable_lse=True):
    if rho2 is None:
        rho2 = rho1
    tau = (rho1 * rho2) / (rho1 + rho2)
    transl = tau * (logsumexp(-f / rho1, a, stable_lse=stable_lse) -
                    logsumexp(-g / rho2, b, stable_lse=stable_lse))
    return transl


def homogeneous_line_search(fin, gin, d_f, d_g, a, b, rho1, rho2, nits,
                            tmax=1.):
    """
    Convex interpolation ft = (1 - t) * fin + t * fout.

    Parameters
    ----------
    fin
    gin
    d_f
    d_g
    rho1
    rho2
    nits
    tmax

    Returns
    -------
    """
    t = 0.5
    tau1, tau2 = 1. / (1. + (rho2 / rho1)), 1. / (1. + (rho1 / rho2))
    for k in range(nits):
        ft = fin + t * d_f
        # print("fin", fin, "d_f", d_f)
        # print("gin", gin, "d_g", d_g)
        gt = gin + t * d_g

        # Compute derivatives for F
        a_z = np.sum(a * np.exp(-ft / rho1))
        a_p = - np.sum(a * np.exp(-ft / rho1) * (-d_f)) / rho1
        a_s = np.sum(a * np.exp(-ft / rho1) * (-d_f) ** 2) / rho1 ** 2
        a_s = tau1 * a_s * a_z ** (tau1 - 1) \
              + tau1 * (tau1 - 1) * a_p ** 2 * a_z ** (tau1 - 2)
        a_p = tau1 * a_p * a_z ** (tau1 - 1)
        a_z = a_z ** tau1

        # Compute derivatives for G
        b_z = np.sum(b * np.exp(-gt / rho2))
        b_p = - np.sum(b * np.exp(-gt / rho2) * (-d_g)) / rho2
        b_s = np.sum(b * np.exp(-gt / rho2) * (-d_g) ** 2) / rho2 ** 2
        # print("bs", b_s, "bp", b_p, "bz", b_z, "gt", gt, "tau1", tau1, "tau2", tau2)
        b_s = tau2 * b_s * b_z ** (tau2 - 1) \
              + tau2 * (tau2 - 1) * b_p ** 2 * b_z ** (tau2 - 2)
        b_p = tau2 * b_p * b_z ** (tau2 - 1)
        b_z = b_z ** tau2

        # Compute damped Newton step
        loss_p = a_p * b_z + a_z * b_p
        loss_s = a_s * b_z + 2 * a_p * b_p + a_z * b_s
        t = t + (loss_p / loss_s) / (1 + np.sqrt(loss_p ** 2 / loss_s))

        # Clamp to keep a convex combination
        t = np.maximum(np.minimum(t, tmax), 0.)

    
    return t


def sej(a, b, n, f, g, lamb, niter = 1000, line_search = False, fr = None):
    """
    Solve the optimal transport problem between densities f and g using FW.

    Parameters
    ----------
    a: float
        Lower bound of the support of the densities
    b: float
        Upper bound of the support of the densities
    n: int
        Number of points to discretize the densities
    f: function
        First density
    g: function
        Second density
    lamb: float
        Regularization parameter
    niter: int
        Number of iterations
    line_search: bool
        Whether to use line search or not
    fr: function
        Reference density

    Returns
    -------
    u: numpy array
        First dual variable
    v: numpy array
        Second dual variable
    A: numpy array
        First marginal
    B: numpy array
        Second marginal
    x_test: numpy array
        Discretization of the support of the first density
    y_test: numpy array
        Discretization of the support of the second density
    a_test: numpy array
        Discretization of the first density
    b_test: numpy array
        Discretization of the second density
    time_fw: numpy array
        Time at each iteration
    fr: function
        Reference density
    norm_fw: numpy array
        Error at each iteration
    """

    # Get measure for alg on the same density
    x_test = np.arange(a, b, (b - a)/n)
    a_test = np.array(f(x_test))
    y_test = np.arange(a, b, (b-a)/n)
    b_test = np.array(g(y_test))

    rho = float(lamb)
    p = 2.

    print('Computation of error for Vanilla FW')

    u, v = np.zeros_like(a_test), np.zeros_like(b_test)

    norm_fw, time_fw = [], []
    A, B = u, v
    transl = rescale_potentials(u, v, a_test, b_test, rho, rho)
    _, _, _, fs, gs, _ = solve_ot(A, B, x_test, y_test, p)
    u + transl, v - transl, np.exp(-u / rho) * a_test, np.exp(-v / rho) * b_test
    np.amax(np.abs(u - fr)) / np.amax(np.abs(fr))
    for j in range(niter):

        t0 = time.time()
        transl = rescale_potentials(u, v, a_test, b_test, rho, rho)
        u, v = u + transl, v - transl

        # Use percentage error instead
        err = np.amax(np.abs(u - fr)) / np.amax(np.abs(fr))
        norm_fw.append(err)

        if j % 1000 == 0:
            print(j, err)

        A = np.exp(-u / rho) * a_test
        B = np.exp(-v / rho) * b_test

        # update
        I, J, P, fs, gs, _ = solve_ot(A, B, x_test, y_test, p)

        if line_search:
            gamma = homogeneous_line_search(u, v, fs - u, gs - v, a_test, b_test, rho,
                                                rho,
                                                nits=5)
        else: gamma = 2. / (2. + j)  # fixed decaying weights
        u = u + gamma * (fs - u)
        v = v + gamma * (gs - v)
        
        if np.isnan(u).any():
            break

        time_fw.append(time.time() - t0)

    transl = rescale_potentials(u, v, a_test, b_test, rho, rho)
    u, v = u + transl, v - transl
    A = np.exp(-u / rho) * a_test
    B = np.exp(-v / rho) * b_test
    return u, v, A, B, x_test, y_test, a_test, b_test, time_fw, fr, norm_fw