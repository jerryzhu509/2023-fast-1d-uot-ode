import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def generate_random_gaussian_mixture(a, b, num_clusters):
    """
    Generate random means, sigmas, and weights for the specified number of clusters

    Parameters
    ----------
    a: float
        Lower bound of the interval
    b: float
        Upper bound of the interval
    num_clusters: int
        Number of clusters
    """
    means = np.random.uniform(a, b, size=num_clusters)
    sigmas = np.random.uniform((b - a) / 8, (b - a) / 4, size=num_clusters)
    weights = np.random.random(size=num_clusters)
    weights /= np.sum(weights)  # Normalize weights to sum up to 1
    
    return means, sigmas, weights


def gaussian_mixture(x, means, sigmas, weights, bal, a=0, b = 20, l = 1, ad = False, clip = False):
    """
    Calculates the probability density function of a Gaussian mixture model at point x.
    The density is truncated outside the interval [a, b].

    Parameters
    ----------
    x: float
        Point at which to evaluate the density
    means: numpy array
        Means of the Gaussian components
    sigmas: numpy array
        Standard deviations of the Gaussian components
    weights: numpy array
        Weights of the Gaussian components
    bal: float
        Scaling factor for the density
    a: float
        Lower bound of the interval
    b: float
        Upper bound of the interval
    l: float
        Constant added to the density
    ad: bool
        Whether to use automatic differentiation or not
    clip: bool
        Whether to truncate the density outside the interval [a, b]
    """
    if not ad:
        if x < a or x > b:
            return 0
        
        gaussian_components = [
            w * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
            for w, mu, sigma in zip(weights, means, sigmas)
        ]
        
        return bal * np.sum(gaussian_components) + l
    
    x = jnp.atleast_1d(x)

    # Define a single Gaussian component
    def single_component(x, w, mu, sigma):
        return w * (1 / (sigma * jnp.sqrt(2 * jnp.pi))) * jnp.exp(-(x - mu)**2 / (2 * sigma**2))

    # Vectorize the computation across different Gaussian components
    single_component_vmap = jax.vmap(single_component, in_axes=(0, None, None, None))

    gaussian_components = [single_component_vmap(x, w, mu, sigma) for w, mu, sigma in zip(weights, means, sigmas)]

    # Truncate the density outside the interval [a, b]
    if clip:
        return jnp.where((x < a) | (x > b), 0, bal * jnp.sum(jnp.array(gaussian_components), axis=0) + l)
    return jnp.squeeze(bal * jnp.sum(jnp.array(gaussian_components), axis=0) + l)


def plot_results(f0, g0, a, b, n, x, y, u, v, f_tilde, g_tilde, u_prime):
    """
    Plot the results of the optimal transport problem

    Parameters
    ----------
    f0: function
        First density
    g0: function
        Second density
    a: float
        Lower bound of the support of the densities
    b: float
        Upper bound of the support of the densities
    n: int
        Number of points to discretize the densities
    x: numpy array
        Support of the first density
    y: numpy array
        Transport map
    u: numpy array
        First dual variable
    v: numpy array
        Second dual variable
    f_tilde: numpy array
        First marginal
    g_tilde: numpy array
        Second marginal
    u_prime: numpy array
        First dual variable derivative
    """
    p, q = np.linspace(a, b, n), np.linspace(a, b, len(f_tilde))

    # Actual densities
    plt.plot(x, f0(x), alpha = 0.5)
    plt.plot(y, g0(y), alpha = 0.5)

    # Marginals
    plt.plot(q, f_tilde)
    plt.plot(y, g_tilde)
    plt.plot(y, v)

    # Estimated densities
    plt.plot(p, u)
    plt.plot(p, u_prime)

    # Add legend
    plt.legend(['f0', 'g0', 'f_tilde', 'g_tilde', 'v', 'u', 'u_prime', 'y'])
    plt.show()


def plot_comparisons(p, time_fw, norm_fw, ts):
    """
    Plot the comparison between the algorithms.

    Parameters
    ----------
    p: numpy array
        Error of the FD algorithm
    time_fw: numpy array
        Time at each iteration of the FW algorithm
    norm_fw: numpy array
        Error at each iteration of the FW algorithm
    ts: numpy array
        Time at each iteration of the FD algorithm
    """
    plt.scatter(np.cumsum(ts), 100 * np.array(p), lw=2)
    plt.scatter(np.cumsum(time_fw),  100 * np.array(norm_fw), linewidth=1)
    plt.xlabel('Time', fontsize=15)
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Relative Max Error (%)', fontsize=15)
    plt.legend(['MS', 'FW'])
    plt.tight_layout()
    plt.show()