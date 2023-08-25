import jax
import jax.numpy as jnp
from jax import jit
import sparsejac # https://github.com/mfschubert/sparsejac/tree/main
from functools import partial
jax.config.update("jax_enable_x64", True)


@partial(jit, static_argnums=(0, 1, 4, 5))
def cost(a, b, x, l, f, g, u):
    """
    Calculates the cost function for the fourth-order finite difference scheme.

    Parameters
    ----------
    a : float
        Lower bound of the interval.
    b : float
        Upper bound of the interval.
    x : jax.numpy.ndarray
        Discretization of the interval.
    l : float
        Regularization parameter.
    f : function
        First density.
    g : function
        Second density.
    u : jax.numpy.ndarray
        Dual variable.

    Returns
    -------
    F : jax.numpy.ndarray
        Cost function.
    u : jax.numpy.ndarray
        Dual variable.
    u_prime : jax.numpy.ndarray
        Derivative of the dual variable.
    """

    # Stepsize
    h = (b - a) / (len(x) + 1)
    u_first = 18 / 11 * u[0] - 9 / 11 * u[1] + 2 / 11 * u[2]
    u_last = 18 / 11 * u[-1] - 9 / 11 * u[-2] + 2 / 11 * u[-3]
    
    # Calculate u_prime
    u_prime_1 = (-2 * u_first - 3 * u[0] + 6 * u[1] - u[2]) / (6 * h)
    u_prime_2 = (u_first + -8*u[0] + 8*u[2] - u[3]) / (12 * h)
    u_prime_middle = (u[:-4] - 8*u[1:-3] + 8*u[3:-1] - u[4:]) / (12 * h)
    u_prime_n2 = (u[-4] - 8*u[-3] + 8*u[-1] - u_last) / (12 * h)
    u_prime_n1 = (u[-3] - 6 * u[-2] + 3 * u[-1] + 2 * u_last) / (6 * h)
    u_prime = cat_scalar_front_and_back(u_prime_2, u_prime_middle, u_prime_n2)
    u_prime = cat_scalar_front_and_back(u_prime_1, u_prime, u_prime_n1)

    # Apply bound
    u_prime = jnp.clip(u_prime, -2 * (b - a), 2 * (b - a))

    # Calculate u_double_prime
    u_double_prime_1 = (u[1] - 2 * u[0] + u_first) / h**2
    u_double_prime_2 = (-u_first + 16*u[0] - 30*u[1] + 16*u[2] - u[3]) / (12 * h**2)
    u_double_prime_middle = (-u[:-4] + 16*u[1:-3] - 30*u[2:-2] + 16*u[3:-1] - u[4:]) / (12 * h**2)
    u_double_prime_n2 = (-u[-4] + 16*u[-3] - 30*u[-2] + 16*u[-1] - u_last) / (12 * h**2)
    u_double_prime_n1 = (u_last - 2 * u[-1] + u[-2]) / h**2
    u_double_prime = cat_scalar_front_and_back(u_double_prime_2, u_double_prime_middle, u_double_prime_n2)
    u_double_prime = cat_scalar_front_and_back(u_double_prime_1, u_double_prime, u_double_prime_n1)

    # Cost
    reg = jnp.exp((-2 * u + 0.25 * u_prime**2) / l)
    F = u_double_prime - 2 * (1 - f(x) / g(jnp.clip(x - 0.5 * u_prime, a, b)) * reg)

    return F, (
        u, u_prime
    )


@partial(jit, static_argnums=(0, 1, 4, 5))
def cost2(a, b, x, l, f, g, u):
    """
    Calculates the cost function for the second-order finite difference scheme.

    Parameters
    ----------
    a : float
        Lower bound of the interval.
    b : float
        Upper bound of the interval.
    x : jax.numpy.ndarray
        Discretization of the interval.
    l : float
        Regularization parameter.
    f : function
        First density.
    g : function
        Second density.
    u : jax.numpy.ndarray
        Dual variable.

    Returns
    -------
    F : jax.numpy.ndarray
        Cost function.
    u : jax.numpy.ndarray
        Dual variable.
    u_prime : jax.numpy.ndarray
        Derivative of the dual variable.
    """
    # Stepsize
    h = (b - a) / (len(x) + 1)

    u_first = 4/3 * u[0] - 1/3 * u[1]
    u_last = 4/3 * u[-1] - 1/3 * u[-2]

    # Calculate u_prime
    u_prime_first = (u[1] - u_first) / (2 * h)
    u_prime_middle = (u[2:] - u[:-2]) / (2 * h)
    u_prime_last = (u_last - u[-2]) / (2 * h)

    u_prime = cat_scalar_front_and_back(u_prime_first, u_prime_middle, u_prime_last)

    # Apply bound
    u_prime = jnp.clip(u_prime, -2 * (b - a), 2 * (b - a))

    # Calculate u_double_prime
    u_double_prime_first = (u_first - 2 * u[0] + u[1]) / h**2
    u_double_prime_middle = (u[2:] - 2 * u[1:-1] + u[:-2]) / h**2
    u_double_prime_last = (u_last - 2 * u[-1] + u[-2]) / h**2
    u_double_prime = cat_scalar_front_and_back(u_double_prime_first, u_double_prime_middle, u_double_prime_last)

    # Cost
    reg = jnp.exp((-2 * u + 0.25 * u_prime**2) / l)
    F = u_double_prime - 2 * (1 - f(x) / g(jnp.clip(x - 0.5 * u_prime, a, b)) * reg)

    return F, (
        u, u_prime
    )

@jax.jit
def TDMA(a, b, c, d):
    """
    TDMA solver, a b c d can be NumPy array type or Python list type.
    Refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

    Parameters
    ----------
    a : array_like
        subdiagonal elements of the matrix
    b : array_like
        diagonal elements of the matrix
    c : array_like
        superdiagonal elements of the matrix
    d : array_like
        right hand side
    """
    # Write all in jax with jax.lax.scan
    n = len(d)

    def forward_step(carry, _):
        i, w, g = carry
        w_new = c[i] / (b[i] - a[i - 1] * w)
        g_new = (d[i] - a[i - 1] * g) / (b[i] - a[i - 1] * w)
        return (i + 1, w_new, g_new), (w_new, g_new)
    
    def backward_step(carry, _):
        i, p = carry
        p_new = g[i] - w[i] * p
        return (i - 1, p_new), p_new

    w_first, g_first = c[0] / b[0], d[0] / b[0]
    
    _, (w, g) = jax.lax.scan(forward_step, (1, w_first, g_first), jnp.arange(1, n - 1))

    w, g = cat_scalar_front(w_first, w), cat_scalar_front(g_first, g)

    last_g = (d[-1] - a[-1] * g[-1]) / (b[-1] - a[-1] * w[-1])
    _, p = jax.lax.scan(backward_step, (n - 2, last_g), jnp.arange(n - 2, -1, -1), reverse=True)
    return cat_scalar_back(last_g, p)


def get_jacobian(n, second_order=True, fwd=True):
    """
    Get the Jacobian of the cost function.

    Parameters
    ----------
    n : int
        Number of discretization points.
    second_order : bool
        Whether to use the second-order finite difference scheme or the fourth-order one.
    fwd : bool
        Whether to use forward or backward automatic differentiation.
    """
    if second_order:
        rows = jnp.concatenate([jnp.array([0, 0]), jnp.repeat(jnp.arange(1, n - 2), 3), jnp.array([n - 2, n - 2])])
        cols = jnp.concatenate([jnp.array([0, 1]), jnp.tile(jnp.array([0, 1, 2]), n - 3) + jnp.repeat(jnp.arange(n - 3), 3), jnp.array([n - 3, n - 2])])
        non_zero_indices = jnp.stack((rows, cols), axis=1)
        data = jnp.ones(len(non_zero_indices))
        sp = jax.experimental.sparse.BCOO((data, non_zero_indices), shape=(n-1, n-1))
        if fwd:
            return jax.jit(sparsejac.jacfwd(cost2, sp, argnums=6, has_aux=True), static_argnums=(0, 1, 4, 5,))
        return jax.jit(sparsejac.jacrev(cost2, sp, argnums=6, has_aux=True), static_argnums=(0, 1, 4, 5,))

    rows = jnp.concatenate([
        jnp.array([0, 0, 0]),
        jnp.array([1, 1, 1, 1]),
        jnp.repeat(jnp.arange(2, n - 3), 5),
        jnp.array([n - 3, n - 3, n - 3, n - 3]),
        jnp.array([n - 2, n - 2, n - 2])
    ])
    cols = jnp.concatenate([
        jnp.array([0, 1, 2]),
        jnp.array([0, 1, 2, 3]),
        jnp.tile(jnp.array([0, 1, 2, 3, 4]), n - 5) + jnp.repeat(jnp.arange(n - 5), 5),
        jnp.array([n - 5, n - 4, n - 3, n - 2]),
        jnp.array([n - 4, n - 3, n - 2])
    ])
    non_zero_indices = jnp.stack((rows, cols), axis=1)
    data = jnp.ones(len(non_zero_indices))
    sp = jax.experimental.sparse.BCOO((data, non_zero_indices), shape=(n-1, n-1))
    if fwd:
        return jax.jit(sparsejac.jacfwd(cost, sp, argnums=6, has_aux=True), static_argnums=(0, 1, 4, 5,))
    return jax.jit(sparsejac.jacrev(cost, sp, argnums=6, has_aux=True), static_argnums=(0, 1, 4, 5,))


@jax.jit
def reduce_to_tridiagonal_jax(d1, d2, d3, d4, d5, rhs):
    """
    Reduce a pentadiagonal system to a tridiagonal system.

    Parameters
    ----------
    d1, d2, d3, d4, d5 : jax.numpy.ndarray
        The five diagonals of the pentadiagonal system, starting with d1 as the subsubdiagonal.
    rhs : jax.numpy.ndarray
        The right-hand side of the system.

    Returns
    -------
    d2, d3, d4 : jax.numpy.ndarray
        The resulting diagonals of the tridiagonal system, starting with d2 as the subdiagonal.
    rhs : jax.numpy.ndarray
        The resulting right-hand side of the system.
    """
    n = len(rhs)

    def forward_body(carry, _):
        i, d2_now, d3_now, d4_now, rhs_now = carry
        m = d1[i] / d2_now
        d2_next = d2[i + 1] - m * d3_now
        d3_next = d3[i + 2] - m * d4_now
        d4_next = d4[i + 2] - m * d5[i + 1]
        rhs_next = rhs[i + 2] - m * rhs_now
        return (i + 1, d2_next, d3_next, d4_next, rhs_next), (d2_next, d3_next, d4_next, rhs_next)

    _, (td2, td3, td4, trhs) = jax.lax.scan(forward_body, (0, d2[0], d3[1], d4[1], rhs[1]), None, length=n-3)

    m = d1[n-3] / td2[-1]
    d2_last = d2[n-2] - m * td3[-1]
    d3_last = d3[n-1] - m * td4[-1]
    rhs_last = rhs[n-1] - m * trhs[-1]

    d2 = cat_scalar_back(d2_last, cat_scalar_front(d2[0], td2))
    d3 = cat_scalar_back(d3_last, cat_scalar_front(d3[0], cat_scalar_front(d3[1], td3)))
    d4 = cat_scalar_front(d4[0], cat_scalar_front(d4[1], td4))
    rhs = cat_scalar_back(rhs_last, cat_scalar_front(rhs[0], cat_scalar_front(rhs[1], trhs)))


    def backward_body(carry, _):
        i, d3_now, d4_now, rhs_now = carry
        m = d5[i] / d4_now
        d3_next = d3[i] - m * d2[i]
        d4_next = d4[i] - m * d3_now
        rhs_next = rhs[i] - m * rhs_now
        return (i - 1, d3_next, d4_next, rhs_next), (d3_next, d4_next, rhs_next)

    _, (td3, td4, trhs) = jax.lax.scan(backward_body, (n - 3, d3[n-2], d4[n-2], rhs[n-2]), None, length=n-2, reverse=True)

    d3 = cat_scalar_back(d3[n-1], cat_scalar_back(d3[n-2], td3))
    d4 = cat_scalar_back(d4[n-1], td4)
    rhs = cat_scalar_back(rhs[n-1], cat_scalar_back(rhs[n-2], trhs))

    return d2, d3, d4, rhs

@jax.jit
def get_d1(arr):
    return arr[2::3]

@jax.jit
def get_d2(arr):
    return arr[::3]

@jax.jit
def get_d3(arr):
    return arr[1::3]

@jax.jit
def cat_scalar_front_and_back(c1, x, c2):
    return jnp.concatenate([jnp.array([c1]), x, jnp.array([c2])])

@jax.jit
def s1(arr):
    return arr[7:-3:5]

@jax.jit
def s2(arr):
    return arr[3:-2:5]

@jax.jit
def s3(arr):
    return arr[4::5]

@jax.jit
def s4(arr):
    return arr[5::5]

@jax.jit
def s5(arr):
    return arr[6:-6:5]

@jax.jit
def cat_scalar_front(s, x):
    return jnp.concatenate([jnp.array([s]), x])

@jax.jit
def cat_scalar_back(s, x):
    return jnp.concatenate([x, jnp.array([s])])

@jax.jit
def l2_norm(x):
    return jnp.sqrt(jnp.sum(x**2))

@jax.jit
def sup_norm(x):
    return jnp.max(jnp.abs(x))

@jax.jit
def slice1(x):
    return x[1:]