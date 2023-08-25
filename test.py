from uot import uot, uot_values
from fd import get_jacobian
from fw import sej
from utils import gaussian_mixture
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

def test(a, b, n, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, init = False, beta = 0.25, ls_its=0, return_all=False, plot = True):
    u_s0, _, _, _, _, _, _, _, time_fw0, _, norm_fw0 = sej(a, b, n, f, g, lamb, sej_its, False, fr)
    second_order = True
    _, res, ts_fd_2nd, e_fd_2nd = uot(a, b, n, f, g, lamb, fd2_its, init, fr, J_2nd, beta, ls_its, second_order=second_order)
    u_fd_2nd = res[1]
    u_prime_fd_2nd = res[2]
    second_order = False
    _, res, ts_fd, e_fd = uot(a, b, n, f, g, lamb, fd4_its, init, fr, J, beta, ls_its, second_order=second_order)
    u_fd = res[1]

    v = res[4]
    f_tilde = res[5]
    g_tilde = res[6]
    u_prime_fd = res[2]
    dx = (b - a) / n
    x = jnp.linspace(a, b - dx, n)
    if plot:
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

    if return_all:
        return time_fw0, norm_fw0, ts_fd_2nd, e_fd_2nd, ts_fd, e_fd, u_fd, v, f_tilde, g_tilde
    return time_fw0, norm_fw0, ts_fd_2nd, e_fd_2nd, ts_fd, e_fd


def make_plots(i, t_sej, t_fd2, t_fd4, err_sej, err_fd2, err_fd4):
    lw = 2.5
    if i == 0:
        label1 = 'FW($\lambda = 10^{-0}$)'
        label2 = 'FD2($\lambda = 10^{-0}$)'
        label3 = 'FD4($\lambda = 10^{-0}$)'
    elif i == 1:
        label1 = 'FW($\lambda = 10^{-1}$)'
        label2 = 'FD2($\lambda = 10^{-1}$)'
        label3 = 'FD4($\lambda = 10^{-1}$)'
    else:
        label1 = 'FW($\lambda = 10^{-3}$)'
        label2 = 'FD2($\lambda = 10^{-3}$)'
        label3 = 'FD4($\lambda = 10^{-3}$)'

    plt.plot(np.cumsum(t_sej[i]),  100 * np.array(err_sej[i]), linewidth=lw, label = label1, color = 'C0',linestyle='solid')
    plt.plot(np.cumsum(t_fd2[i]), 100 * np.array(err_fd2[i]), lw=lw, label = label2, color = 'C3', linestyle='solid')
    plt.plot(np.cumsum(t_fd4[i]), 100 * np.array(err_fd4[i]), lw=lw, label = label3, color = 'C2', linestyle='solid')
    plt.xlabel('Seconds', fontsize=12)
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('$ \\frac{\Vert u-u^*\Vert_\infty}{\Vert u^*\Vert_\infty}$ (%)', fontsize=15)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Similar density
    a, b = 0, 1
    lamb = 1.
    l_f, l_g = .5, .5
    bal_f, bal_g = 3., 1.

    # For the f distribution with 10 components
    meansf = jnp.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    sigmasf = np.array([0.03, 0.02, 0.018, 0.02, 0.03, 0.02, 0.018, 0.02, 0.015, 0.03])
    weightsf = jnp.array([0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.05, 0.1, 0.15, 0.05])

    # For the g distribution with 12 components
    meansg = jnp.array([0.03, 0.10, 0.2, 0.26, 0.35, 0.45, 0.52, 0.6, 0.68, 0.76, 0.84, 0.92])
    sigmasg = jnp.array([0.02, 0.02, 0.012, 0.02, 0.015, 0.02, 0.012, 0.02, 0.012, 0.02, 0.012, 0.06])
    weightsg = jnp.array([0.04, 0.08, 0.12, 0.04, 0.08, 0.12, 0.04, 0.08, 0.12, 0.04, 0.08, 0.08])

    f = jax.jit(lambda x: gaussian_mixture(x, meansf, sigmasf, weightsf, bal_f, a, b, l_f, True, False))
    g = jax.jit(lambda x: gaussian_mixture(x, meansg, sigmasg, weightsg, bal_g, a, b, l_g, True, False))
    n = 5000

    x = jnp.arange(a, b, (b-a)/n)
    plt.plot(x, f(x), label='f')
    plt.plot(x, g(x), label='g')
    plt.grid()
    plt.legend()
    plt.show()

    # n = 500000
    # lamb = _
    # u_s0, v_s0, _, _, _, _, _, _, time_fw0, _, norm_fw0 = sej(a, b, n, f, g, lamb, 50000, False, np.ones(n))
    # np.save('ref0', u_s0)

    # load numpy arrays fr
    ref0 = jnp.load('ref/ref0.npy')
    ref_1 = jnp.load('ref/ref-1.npy')
    ref_2 = jnp.load('ref/ref-2.npy')
    ref_3 = jnp.load('ref/uref-3.npy')
    ref_4 = jnp.load('ref/uref-4.npy')

    t_sej, err_sej, t_fd2, err_fd2, t_fd4, err_fd4  = [], [], [], [], [], []

    n = 5000
    # can comment out after initializing
    J_2nd = get_jacobian(n, second_order=True)
    J = get_jacobian(n, second_order=False)

    # Run once to avoid compilation time
    its = 5
    test(a, b, n, lamb, jnp.ones(n), J_2nd, J, its, its, its, True, 0.25, 0, False, False)

    # 1e0
    lamb = 1e0
    fr = ref0[::500000//n]
    sej_its = 200
    fd2_its = 30
    fd4_its = 35

    time_fw0_0, norm_fw0_0, ts_fd_2nd_0, e_fd_2nd_0, ts_fd_0, e_fd_0, u, v, f_tilde, g_tilde = test(a, b, n, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, return_all=True)

    # 1e-1
    lamb = 1e-1
    fr = ref_1[::500000//n]
    sej_its = 200
    fd2_its = 15
    fd4_its = 15
    time_fw0_1, norm_fw0_1, ts_fd_2nd_1, e_fd_2nd_1, ts_fd_1, e_fd_1, u, v, f_tilde, g_tilde = test(a, b, n, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, return_all=True)

    # 1e-2
    lamb = 1e-2
    fr = ref_2[::500000//n]
    sej_its = 2000
    fd2_its = 10
    fd4_its = 10
    time_fw0_2, norm_fw0_2, ts_fd_2nd_2, e_fd_2nd_2, ts_fd_2, e_fd_2, u, v, f_tilde, g_tilde = test(a, b, n, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, return_all=True)

    plt.plot(x, u, label='u')
    plt.plot(x, v, label='v')
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()

    plt.plot(x, f_tilde, label='$\\tilde{f}$')
    plt.plot(x, g_tilde, label='$\\tilde{g}$')
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()

    # 1e-3
    lamb = 1e-3
    fr = ref_3[::500000//n]
    sej_its = 20000
    fd2_its = 15
    fd4_its = 15
    time_fw0_3, norm_fw0_3, ts_fd_2nd_3, e_fd_2nd_3, ts_fd_3, e_fd_3 = test(a, b, n, lamb, fr, J_2nd, J, sej_its, fd2_its, fd4_its, True)

    # Append the lamb =0, 1e-1, 1e-3 cases
    t_sej.append(time_fw0_0)
    err_sej.append(norm_fw0_0)
    t_fd2.append(ts_fd_2nd_0)
    err_fd2.append(e_fd_2nd_0)
    t_fd4.append(ts_fd_0)
    err_fd4.append(e_fd_0)

    t_sej.append(time_fw0_1)
    err_sej.append(norm_fw0_1)
    t_fd2.append(ts_fd_2nd_1)
    err_fd2.append(e_fd_2nd_1)
    t_fd4.append(ts_fd_1)
    err_fd4.append(e_fd_1)

    t_sej.append(time_fw0_3)
    err_sej.append(norm_fw0_3)
    t_fd2.append(ts_fd_2nd_3)
    err_fd2.append(e_fd_2nd_3)
    t_fd4.append(ts_fd_3)
    err_fd4.append(e_fd_3)

    make_plots(0, t_sej, t_fd2, t_fd4, err_sej, err_fd2, err_fd4)
    make_plots(1, t_sej, t_fd2, t_fd4, err_sej, err_fd2, err_fd4)
    make_plots(2, t_sej, t_fd2, t_fd4, err_sej, err_fd2, err_fd4)

    # Sample uot values
    uot_cost, d_uot = uot_values(u, x, f, g, lamb, f_tilde, g_tilde)
    print("UOT cost:", uot_cost, "UOT derivative:", d_uot)