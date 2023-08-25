from uot import uot, uot_values
from fd import get_jacobian
from fw import sej
from utils import gaussian_mixture
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)


def test_lower_bound(a, b, n, f, g, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, beta = 0.25, ls_its=0, init = False):
    u_s0, v_s0, _, _, _, _, _, _, time_fw0, _, norm_fw0 = sej(a, b, n, f, g, lamb, sej_its, False, fr)
    second_order = True
    F, res, ts_fd_2nd, e_fd_2nd = uot(a, b, n, f, g, lamb, fd2_its, init, fr, J_2nd, beta, ls_its, second_order=second_order)
    u_fd_2nd = res[1]
    u_prime_fd_2nd = res[2]
    second_order = False
    F, res, ts_fd, e_fd = uot(a, b, n, f, g, lamb, fd4_its, init, fr, J, beta, ls_its, second_order=second_order)
    u_fd = res[1]
    u_prime_fd = res[2]
    dx = (b - a) / n
    x = jnp.linspace(a, b - dx, n)
    plt.plot(x, u_s0, label = 'SEJ')
    plt.plot(x, u_fd_2nd, label = '2nd FD')
    plt.plot(x, u_fd, label = '4th FD')
    plt.plot(x, fr, label = 'ref')
    plt.legend()
    plt.show()

    plt.plot(x, u_prime_fd_2nd, label = '2nd FD')
    plt.plot(x, u_prime_fd, label = '4th FD')
    plt.show()

    plt.plot(np.cumsum(time_fw0), norm_fw0, label='SEJ')
    plt.plot(np.cumsum(ts_fd_2nd), e_fd_2nd, label='2nd FD')
    plt.plot(np.cumsum(ts_fd), e_fd, label='4th FD')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

    return time_fw0, norm_fw0, ts_fd_2nd, e_fd_2nd, ts_fd, e_fd


if __name__ == '__main__':
    fr_2 = jnp.load('ref/uref-1_2.npy')
    fr_1 = jnp.load('ref/uref-1_1.npy')
    fr_075 = jnp.load('ref/uref-1_.75.npy')
    fr_05 = jnp.load('ref/ref-1.npy')
    fr_04 = jnp.load('ref/uref-1_.4.npy')
    fr_025 = jnp.load('ref/uref-1_.25.npy')
    fr_01 = jnp.load('ref/uref-1_.1.npy')
    fr_0025 = jnp.load('ref/uref-1_.025.npy')
    fr_0 = jnp.load('ref/uref-1_.0.npy')


    # Similar density
    a, b = 0, 1
    l_f, l_g = 1., 1.
    bal_f, bal_g = 3., 1.

    # For the f distribution with 10 components
    meansf = jnp.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    sigmasf = np.array([0.03, 0.02, 0.018, 0.02, 0.03, 0.02, 0.018, 0.02, 0.015, 0.03])
    # sigmasf = jnp.array([0.03, 0.025, 0.02, 0.025, 0.035, 0.025, 0.02, 0.025, 0.02, 0.03])
    weightsf = jnp.array([0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.05])

    # For the g distribution with 12 components
    meansg = jnp.array([0.03, 0.10, 0.2, 0.26, 0.35, 0.45, 0.52, 0.6, 0.68, 0.76, 0.84, 0.92])
    sigmasg = jnp.array([0.02, 0.02, 0.012, 0.02, 0.015, 0.02, 0.012, 0.02, 0.012, 0.02, 0.012, 0.06])
    # sigmasg = jnp.array([0.02, 0.025, 0.015, 0.025, 0.02, 0.025, 0.015, 0.025, 0.015, 0.025, 0.015, 0.06])
    weightsg = jnp.array([0.04, 0.08, 0.12, 0.04, 0.08, 0.12, 0.04, 0.08, 0.12, 0.04, 0.08, 0.08])

    f = jax.jit(lambda x: gaussian_mixture(x, meansf, sigmasf, weightsf, bal_f, a, b, l_f, True, False))
    g = jax.jit(lambda x: gaussian_mixture(x, meansg, sigmasg, weightsg, bal_g, a, b, l_g, True, False))
    n = 5000
    lamb = 1e-1
    dx = (b-a)/n
    x = jnp.linspace(a, b-dx, n-1)


    l_f, l_g = 2., 2.
    f = jax.jit(lambda x: gaussian_mixture(x, meansf, sigmasf, weightsf, bal_f, a, b, l_f, True, False))
    g = jax.jit(lambda x: gaussian_mixture(x, meansg, sigmasg, weightsg, bal_g, a, b, l_g, True, False))
    fr = fr_2[::500000//n]
    f(x), g(x)
    sej_its = 250
    fd2_its = 15
    fd4_its = 15
    beta = 0.25
    ls_its = 0
    J_2nd = get_jacobian(n, second_order=True)
    J = get_jacobian(n, second_order=False)
    time_fw0_2, norm_fw0_2, ts_fd_2nd_2, e_fd_2nd_2, ts_fd_2, e_fd_2 = test_lower_bound(a, b, n, f, g, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, beta, ls_its)

    l_f, l_g = 1., 1.
    f = jax.jit(lambda x: gaussian_mixture(x, meansf, sigmasf, weightsf, bal_f, a, b, l_f, True, False))
    g = jax.jit(lambda x: gaussian_mixture(x, meansg, sigmasg, weightsg, bal_g, a, b, l_g, True, False))
    fr = fr_1[::500000//n]
    f(x), g(x)
    sej_its = 250
    fd2_its = 15
    fd4_its = 15
    beta = 0.25
    ls_its = 0
    J_2nd = get_jacobian(n, second_order=True)
    J = get_jacobian(n, second_order=False)
    time_fw0_1, norm_fw0_1, ts_fd_2nd_1, e_fd_2nd_1, ts_fd_1, e_fd_1 = test_lower_bound(a, b, n, f, g, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, beta, ls_its)

    l_f, l_g = .75, .75
    f = jax.jit(lambda x: gaussian_mixture(x, meansf, sigmasf, weightsf, bal_f, a, b, l_f, True, False))
    g = jax.jit(lambda x: gaussian_mixture(x, meansg, sigmasg, weightsg, bal_g, a, b, l_g, True, False))
    fr = fr_075[::500000//n]
    f(x), g(x)
    sej_its = 250
    fd2_its = 15
    fd4_its = 15
    beta = 0.25
    ls_its = 0
    J_2nd = get_jacobian(n, second_order=True)
    J = get_jacobian(n, second_order=False)
    time_fw0_075, norm_fw0_075, ts_fd_2nd_075, e_fd_2nd_075, ts_fd_075, e_fd_075 = test_lower_bound(a, b, n, f, g, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, beta, ls_its)

    l_f, l_g = .5, .5
    f = jax.jit(lambda x: gaussian_mixture(x, meansf, sigmasf, weightsf, bal_f, a, b, l_f, True, False))
    g = jax.jit(lambda x: gaussian_mixture(x, meansg, sigmasg, weightsg, bal_g, a, b, l_g, True, False))
    fr = fr_05[::500000//n]
    f(x), g(x)
    sej_its = 250
    fd2_its = 15
    fd4_its = 15
    beta = 0.25
    ls_its = 0
    J_2nd = get_jacobian(n, second_order=True)
    J = get_jacobian(n, second_order=False)
    time_fw0_05, norm_fw0_05, ts_fd_2nd_05, e_fd_2nd_05, ts_fd_05, e_fd_05 = test_lower_bound(a, b, n, f, g, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, beta, ls_its)

    l_f, l_g = .4, .4
    f = jax.jit(lambda x: gaussian_mixture(x, meansf, sigmasf, weightsf, bal_f, a, b, l_f, True, False))
    g = jax.jit(lambda x: gaussian_mixture(x, meansg, sigmasg, weightsg, bal_g, a, b, l_g, True, False))
    fr = fr_04[::500000//n]
    f(x), g(x)
    sej_its = 250
    fd2_its = 30
    fd4_its = 30
    beta = 0.25
    ls_its = 0
    J_2nd = get_jacobian(n, second_order=True)
    J = get_jacobian(n, second_order=False)
    time_fw0_04, norm_fw0_04, ts_fd_2nd_04, e_fd_2nd_04, ts_fd_04, e_fd_04 = test_lower_bound(a, b, n, f, g, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, beta, ls_its)

    l_f, l_g = .25, .25
    f = jax.jit(lambda x: gaussian_mixture(x, meansf, sigmasf, weightsf, bal_f, a, b, l_f, True, False))
    g = jax.jit(lambda x: gaussian_mixture(x, meansg, sigmasg, weightsg, bal_g, a, b, l_g, True, False))
    fr = fr_025[::500000//n]
    f(x), g(x)
    sej_its = 250
    fd2_its = 30
    fd4_its = 30
    beta = 0.25
    ls_its = 2
    J_2nd = get_jacobian(n, second_order=True)
    J = get_jacobian(n, second_order=False)
    time_fw0_025, norm_fw0_025, ts_fd_2nd_025, e_fd_2nd_025, ts_fd_025, e_fd_025 = test_lower_bound(a, b, n, f, g, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, beta, ls_its)


    l_f, l_g = .1, .1
    f = jax.jit(lambda x: gaussian_mixture(x, meansf, sigmasf, weightsf, bal_f, a, b, l_f, True, False))
    g = jax.jit(lambda x: gaussian_mixture(x, meansg, sigmasg, weightsg, bal_g, a, b, l_g, True, False))
    fr = fr_01[::500000//n]
    f(x), g(x)
    sej_its = 250
    fd2_its = 50
    fd4_its = 50
    beta = 0.25
    ls_its = 2
    J_2nd = get_jacobian(n, second_order=True)
    J = get_jacobian(n, second_order=False)
    time_fw0_01, norm_fw0_01, ts_fd_2nd_01, e_fd_2nd_01, ts_fd_01, e_fd_01 = test_lower_bound(a, b, n, f, g, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, beta, ls_its)

    l_f, l_g = .025, .025
    f = jax.jit(lambda x: gaussian_mixture(x, meansf, sigmasf, weightsf, bal_f, a, b, l_f, True, False))
    g = jax.jit(lambda x: gaussian_mixture(x, meansg, sigmasg, weightsg, bal_g, a, b, l_g, True, False))
    fr = fr_0025[::500000//n]
    f(x), g(x)
    sej_its = 300
    fd2_its = 250
    fd4_its = 150
    beta = 0.1
    ls_its = 2
    J_2nd = get_jacobian(n, second_order=True)
    J = get_jacobian(n, second_order=False)
    time_fw0_0025, norm_fw0_0025, ts_fd_2nd_0025, e_fd_2nd_0025, ts_fd_0025, e_fd_0025 = test_lower_bound(a, b, n, f, g, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, beta, ls_its)

    l_f, l_g = .0, .0
    f = jax.jit(lambda x: gaussian_mixture(x, meansf, sigmasf, weightsf, bal_f, a, b, l_f, True, False))
    g = jax.jit(lambda x: gaussian_mixture(x, meansg, sigmasg, weightsg, bal_g, a, b, l_g, True, False))
    fr = fr_0[::500000//n]
    f(x), g(x)
    sej_its = 400
    fd2_its = 1000
    fd4_its = 250
    beta = 0.1
    ls_its = 3
    J_2nd = get_jacobian(n, second_order=True)
    J = get_jacobian(n, second_order=False)
    time_fw0_0, norm_fw0_0, ts_fd_2nd_0, e_fd_2nd_0, ts_fd_0, e_fd_0 = test_lower_bound(a, b, n, f, g, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, beta, ls_its)

    lb = [0, 0.025, .1, 0.25, 0.4, 0.5, 0.75, 1., 2.]
    x_ax, = [],
    for l in lb:
        f = jax.jit(lambda x: gaussian_mixture(x, meansf, sigmasf, weightsf, bal_f, a, b, l, True, False))
        g = jax.jit(lambda x: gaussian_mixture(x, meansg, sigmasg, weightsg, bal_g, a, b, l, True, False))

        M_f = jnp.amax(f(x))
        m_g = jnp.amin(g(x))

        x_ax.append(M_f / m_g)

    y_ax = [time_fw0_0, time_fw0_0025, time_fw0_01, time_fw0_025, time_fw0_04, time_fw0_05, time_fw0_075, time_fw0_1, time_fw0_2]
    y_ax2 = [ts_fd_2nd_0, ts_fd_2nd_0025, ts_fd_2nd_01, ts_fd_2nd_025, ts_fd_2nd_04, ts_fd_2nd_05, ts_fd_2nd_075, ts_fd_2nd_1, ts_fd_2nd_2]
    y_ax4 = [ts_fd_0, ts_fd_0025, ts_fd_01, ts_fd_025, ts_fd_04, ts_fd_05, ts_fd_075, ts_fd_1, ts_fd_2]

    y_ax_sej, y_ax_fd2, y_ax_fd4 = [], [], []

    for lst in y_ax:
        y_ax_sej.append(sum(lst))
        # y_ax_sej.append(len(lst))

    for lst in y_ax2:
        y_ax_fd2.append(sum(lst))
        # y_ax_fd2.append(len(lst))

    for lst in y_ax4:
        y_ax_fd4.append(sum(lst))
        # y_ax_fd4.append(len(lst))

    plt.plot(x_ax, y_ax_sej, label='FW')
    plt.plot(x_ax, y_ax_fd2, label='FD2')
    plt.plot(x_ax, y_ax_fd4, label='FD4')
    plt.xlabel('$ \\frac{M_f}{m_g}$', fontsize=14)
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Seconds', fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.show()