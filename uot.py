import jax
import jax.numpy as jnp
import time
import fd
from functools import partial
jax.config.update("jax_enable_x64", True)

def uot(a, b, n, f, g, l, iters=30, init=True, fr = None, J=None, beta=0.1, its=10, second_order=True):
    """
    Computes the 1D unbalanced optimal transport between two densities f and g.

    Parameters
    ----------
    a : float
        Lower bound of the interval.
    b : float
        Upper bound of the interval.
    n : int
        Number of points to discretize the densities.
    f : function
        First density.
    g : function
        Second density.
    l : float
        Regularization parameter.
    iters : int
        Number of iterations.
    init : bool
        Whether to use the small lambda initial guess or not. Default true and occurs when l / (b - a) < 1e-3.
    fr : numpy array
        Reference density.
    J : function
        Jacobian of the cost function.
    beta : float
        Step size reduction factor for the line search.
    its : int
        Number of iterations for the line search.
    second_order : bool
        Whether to use the second order method. If false, then the fourth order method is used. Default true.

    Returns
    -------
    res : float
        The final cost.
    (x, u, u_prime, y, v, f_tilde, g_tilde) : tuple
        The discretized densities and auxiliary functions.
    ts : numpy array
        Time at each iteration.
    err : numpy array
        Error at each iteration.
    """
    h = (b - a) / n
    x = jnp.linspace(a + h, b - h, n - 1)
    
    # Initial guess
    if init and l / (b - a) < 1e-3:
        u_fd = jnp.ones(n - 1) * jnp.log(f(x) / g(x)) * l / 2
    else:
        u_fd = jnp.ones(n - 1) * jnp.log(jnp.mean(f(x)) / jnp.mean(g(x))) * l / 2


    # Initialize Jacobian if not provided
    if J == None:
        J = fd.get_jacobian(n, second_order=second_order)


    ts, err = [], []
    d = fd.sup_norm(fr)

    for i in range(iters + 1):
        t_ = time.time()

        if second_order:
            H = fd.cost2(a, b, x, l, f, g, u_fd)[0]
            G = J(a, b, x, l, f, g, u_fd)[0].data
            a1, b1, c1 = fd.get_d1(G), fd.get_d2(G), fd.get_d3(G)
            delta = -fd.TDMA(a1, b1, c1, H)
        else:
            H = fd.cost(a, b, x, l, f, g, u_fd)[0]
            G = J(a, b, x, l, f, g, u_fd)[0].data

            d1_last = G[-3]
            d1 = fd.cat_scalar_back(d1_last, fd.s1(G))

            d2_last = G[-2]
            d2 = fd.cat_scalar_back(d2_last, fd.s2(G))

            d3_first = G[0]
            d3_last = G[-1]
            d3 = fd.cat_scalar_back(d3_last, fd.cat_scalar_front(d3_first, fd.s3(G)))

            d4_first = G[1]
            d4 = fd.cat_scalar_front(d4_first, fd.s4(G))

            d5_first = G[2]
            d5 = fd.cat_scalar_front(d5_first, fd.s5(G))

            a1, b1, c1, rhs = fd.reduce_to_tridiagonal_jax(d1, d2, d3, d4, d5, H)
            delta = -fd.TDMA(a1, b1, c1, rhs)
        
        u_new = u_fd + delta
        alpha = 1.0  # Initial step size
        c2 = fd.l2_norm(H)

        # Line search
        for _ in range(its):
            u_new = u_fd + alpha * delta
            if second_order:
                c1 = fd.l2_norm(fd.cost2(a, b, x, l, f, g, u_new)[0])
            else:
                c1 = fd.l2_norm(fd.cost(a, b, x, l, f, g, u_new)[0])
            if c1 < c2:
                break
            else:
                alpha *= beta

        # Apply the update. Disregard the first iteration involving compilation
        if i > 0:
            u_fd = u_new
            e = fd.sup_norm(u_fd - fd.slice1(fr)) / d
            ts.append(time.time() - t_)
            err.append(e)
        # print(i, e, fd.sup_norm(H), time.time() - t_)
    
    # Compute the results
    if second_order:
        res = fd.cost2(a, b, x, l, f, g, u_fd)
    else:
        res = fd.cost(a, b, x, l, f, g, u_fd)

    u, u_prime = res[1]

    # Compute the boundary point
    if second_order:
        u_first = 4/3 * u[0] - 1/3 * u[1]
    else:
        u_first = 18 / 11 * u[0] - 9 / 11 * u[1] + 2 / 11 * u[2]

    u = fd.cat_scalar_front(u_first, u)
    u_prime = fd.cat_scalar_front(0., u_prime)
    x = fd.cat_scalar_front(a, x)

    # Compute the auxiliary functions
    y = x - 0.5 * u_prime
    v = 0.25 * u_prime * u_prime - u
    f_tilde = f(x) * jnp.exp(-u / l)
    g_tilde = g(y) * jnp.exp(-v / l)

    return res[0], (x, u, u_prime, y, v, f_tilde, g_tilde), ts, err


@partial(jax.jit, static_argnums=(2, 3))
def uot_values(u, x, f, g, l, f_tilde, g_tilde):
    """
    Return the optimal value of UOT and its Frechet derivative w.r.t. f.
    Integral approximated by trapezoidal rule.
    """
    int_f = jnp.trapz(f(x), x)
    int_g = jnp.trapz(g(x), x)
    int_f_tilde = jnp.trapz(f_tilde, x)
    int_g_tilde = jnp.trapz(g_tilde, x)
    uot_cost = int_f + int_g - int_f_tilde - int_g_tilde
    d_uot_cost = -l * (jnp.exp(-u / l) - 1)

    return uot_cost, d_uot_cost