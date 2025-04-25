"""
********************************************************************************
compare the convergence of damped Jacobi and SOR methods
********************************************************************************
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmread

import utils
import solvers

parser = argparse.ArgumentParser()
parser.add_argument("-t", type=float, default=1e-9, help="tolerance")
parser.add_argument("-i", type=int, default=int(1e3), help="max iteration")
args = parser.parse_args()


def plot_setting():
    """
    see: https://matplotlib.org/stable/users/explain/customizing.html
    """
    plt.style.use("default")
    plt.style.use("seaborn-v0_8-deep")
    plt.style.use("seaborn-v0_8-talk")   # paper / notebook / talk / poster
    # plt.style.use("classic")
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = True
    plt.rcParams['axes.axisbelow'] = True   # background grid
    plt.rcParams["grid.alpha"] = .3
    plt.rcParams["legend.framealpha"] = .8
    plt.rcParams["legend.facecolor"] = "w"
    plt.rcParams["savefig.dpi"] = 300


def main():
    # Sherman1: Sherman's oil reservoir problem
    A = mmread("sherman1.mtx").toarray()
    b = mmread("sherman1_rhs1.mtx").squeeze()
    x0 = np.zeros_like(b)

    # # Damped Jacobi method
    # # x00, it_log00, res_log00, rho00 = solvers.DampedJacobi(A, b, x0, omega=.0, tol=args.t, max_iter=args.i)
    # x02, it_log02, res_log02, rho02 = solvers.DampedJacobi(A, b, x0, omega=.2, tol=args.t, max_iter=args.i)
    # x04, it_log04, res_log04, rho04 = solvers.DampedJacobi(A, b, x0, omega=.4, tol=args.t, max_iter=args.i)
    # x06, it_log06, res_log06, rho06 = solvers.DampedJacobi(A, b, x0, omega=.6, tol=args.t, max_iter=args.i)
    # x08, it_log08, res_log08, rho08 = solvers.DampedJacobi(A, b, x0, omega=.8, tol=args.t, max_iter=args.i)
    # x10, it_log10, res_log10, rho10 = solvers.DampedJacobi(A, b, x0, omega=1., tol=args.t, max_iter=args.i)
    # x12, it_log12, res_log12, rho12 = solvers.DampedJacobi(A, b, x0, omega=1.2, tol=args.t, max_iter=args.i)
    # x14, it_log14, res_log14, rho14 = solvers.DampedJacobi(A, b, x0, omega=1.4, tol=args.t, max_iter=args.i)
    # x16, it_log16, res_log16, rho16 = solvers.DampedJacobi(A, b, x0, omega=1.6, tol=args.t, max_iter=args.i)
    # x18, it_log18, res_log18, rho18 = solvers.DampedJacobi(A, b, x0, omega=1.8, tol=args.t, max_iter=args.i)
    # # x20, it_log20, res_log20, rho20 = solvers.DampedJacobi(A, b, x0, omega=2.0, tol=args.t, max_iter=args.i)

    # plt.figure()
    # # plt.plot(it_log00, res_log00, label=rf"Damped Jacobi ($\omega=0.0, \varrho={rho00:.4f}$)", ls="--", c="tab:blue", alpha=1.)
    # plt.plot(it_log02, res_log02, label=rf"Damped Jacobi ($\omega=0.2, \varrho={rho02:.4f}$)", ls="--", c="tab:blue", alpha=.8)
    # plt.plot(it_log04, res_log04, label=rf"Damped Jacobi ($\omega=0.4, \varrho={rho04:.4f}$)", ls="--", c="tab:blue", alpha=.6)
    # plt.plot(it_log06, res_log06, label=rf"Damped Jacobi ($\omega=0.6, \varrho={rho06:.4f}$)", ls="--", c="tab:blue", alpha=.4)
    # plt.plot(it_log08, res_log08, label=rf"Damped Jacobi ($\omega=0.8, \varrho={rho08:.4f}$)", ls="--", c="tab:blue", alpha=.2)
    # plt.plot(it_log10, res_log10, label=rf"Damped Jacobi ($\omega=1.0, \varrho={rho10:.4f}$)", ls="--", c="k", alpha=.8)
    # plt.plot(it_log12, res_log12, label=rf"Damped Jacobi ($\omega=1.2, \varrho={rho12:.4f}$)", ls="--", c="tab:red", alpha=.2)
    # plt.plot(it_log14, res_log14, label=rf"Damped Jacobi ($\omega=1.4, \varrho={rho14:.4f}$)", ls="--", c="tab:red", alpha=.4)
    # plt.plot(it_log16, res_log16, label=rf"Damped Jacobi ($\omega=1.6, \varrho={rho16:.4f}$)", ls="--", c="tab:red", alpha=.6)
    # plt.plot(it_log18, res_log18, label=rf"Damped Jacobi ($\omega=1.8, \varrho={rho18:.4f}$)", ls="--", c="tab:red", alpha=.8)
    # # plt.plot(it_log20, res_log20, label=rf"Damped Jacobi ($\omega=2.0, \varrho={rho20:.4f}$)", ls="--", c="tab:red", alpha=1.)
    # plt.yscale("log")
    # plt.ylim(1e-10, 1e0)
    # plt.xlabel("# of iteration")
    # plt.ylabel(r"$\| r \|_{\infty}$")
    # plt.legend(loc="lower left")
    # plt.title("Damped Jacobi method")
    # plt.savefig("sherman1_damped_jacobi.png")
    # plt.close()

    # # SOR method
    # # x00, it_log00, res_log00, rho00 = solvers.SOR(A, b, x0, omega=.0, tol=args.t, max_iter=args.i)
    # x02, it_log02, res_log02, rho02 = solvers.SOR(A, b, x0, omega=.2, tol=args.t, max_iter=args.i)
    # x04, it_log04, res_log04, rho04 = solvers.SOR(A, b, x0, omega=.4, tol=args.t, max_iter=args.i)
    # x06, it_log06, res_log06, rho06 = solvers.SOR(A, b, x0, omega=.6, tol=args.t, max_iter=args.i)
    # x08, it_log08, res_log08, rho08 = solvers.SOR(A, b, x0, omega=.8, tol=args.t, max_iter=args.i)
    # x10, it_log10, res_log10, rho10 = solvers.SOR(A, b, x0, omega=1., tol=args.t, max_iter=args.i)
    # x12, it_log12, res_log12, rho12 = solvers.SOR(A, b, x0, omega=1.2, tol=args.t, max_iter=args.i)
    # x14, it_log14, res_log14, rho14 = solvers.SOR(A, b, x0, omega=1.4, tol=args.t, max_iter=args.i)
    # x16, it_log16, res_log16, rho16 = solvers.SOR(A, b, x0, omega=1.6, tol=args.t, max_iter=args.i)
    # x18, it_log18, res_log18, rho18 = solvers.SOR(A, b, x0, omega=1.8, tol=args.t, max_iter=args.i)
    # # x20, it_log20, res_log20, rho20 = solvers.SOR(A, b, x0, omega=2.0, tol=args.t, max_iter=args.i)

    # plt.figure()
    # # plt.plot(it_log00, res_log00, label=rf"SOR ($\omega=0.0, \varrho={rho00:.4f}$)", ls="--", c="tab:blue", alpha=1.)
    # plt.plot(it_log02, res_log02, label=rf"SOR ($\omega=0.2, \varrho={rho02:.4f}$)", ls="--", c="tab:blue", alpha=.8)
    # plt.plot(it_log04, res_log04, label=rf"SOR ($\omega=0.4, \varrho={rho04:.4f}$)", ls="--", c="tab:blue", alpha=.6)
    # plt.plot(it_log06, res_log06, label=rf"SOR ($\omega=0.6, \varrho={rho06:.4f}$)", ls="--", c="tab:blue", alpha=.4)
    # plt.plot(it_log08, res_log08, label=rf"SOR ($\omega=0.8, \varrho={rho08:.4f}$)", ls="--", c="tab:blue", alpha=.2)
    # plt.plot(it_log10, res_log10, label=rf"SOR ($\omega=1.0, \varrho={rho10:.4f}$)", ls="--", c="k", alpha=.8)
    # plt.plot(it_log12, res_log12, label=rf"SOR ($\omega=1.2, \varrho={rho12:.4f}$)", ls="--", c="tab:red", alpha=.2)
    # plt.plot(it_log14, res_log14, label=rf"SOR ($\omega=1.4, \varrho={rho14:.4f}$)", ls="--", c="tab:red", alpha=.4)
    # plt.plot(it_log16, res_log16, label=rf"SOR ($\omega=1.6, \varrho={rho16:.4f}$)", ls="--", c="tab:red", alpha=.6)
    # plt.plot(it_log18, res_log18, label=rf"SOR ($\omega=1.8, \varrho={rho18:.4f}$)", ls="--", c="tab:red", alpha=.8)
    # # plt.plot(it_log20, res_log20, label=rf"SOR ($\omega=2.0, \varrho={rho20:.4f}$)", ls="--", c="tab:red", alpha=1.)
    # plt.yscale("log")
    # plt.ylim(1e-10, 1e0)
    # plt.xlabel("# of iteration")
    # plt.ylabel(r"$\| r \|_{\infty}$")
    # plt.legend(loc="lower left")
    # plt.title("SOR method")
    # plt.savefig("sherman1_sor.png")
    # plt.close()

    # cond = np.linalg.cond(A, p=1)
    # print(f">>> condition number of A (1-norm): {cond:.6e}")
    # cond = np.linalg.cond(A, p=2)
    # print(f">>> condition number of A (2-norm): {cond:.6e}")
    # cond = np.linalg.cond(A, p=np.inf)
    # print(f">>> condition number of A (inf-norm): {cond:.6e}")

    # # plot A
    # plt.figure()
    # plt.spy(A)
    # plt.savefig("sherman1_A.png")
    # plt.close()

    # # plot b
    # plt.figure()
    # plt.plot(b)
    # plt.savefig("sherman1_b.png")
    # plt.close()

    ############################################################################
    # compare the solvers
    ############################################################################

    # condition number
    cond = np.linalg.cond(A, p=2)
    x_j, it_log_j, res_log_j, rho_j = solvers.Jacobi(A, b, x=x0, tol=args.t, max_iter=args.i)
    x_dj, it_log_dj, res_log_dj, rho_dj = solvers.DampedJacobi(A, b, x=x0, omega=.8, tol=args.t, max_iter=args.i)
    x_gs, it_log_gs, res_log_gs, rho_gs = solvers.GaussSeidel(A, b, x=x0, tol=args.t, max_iter=args.i)
    x_sor, it_log_sor, res_log_sor, rho_sor = solvers.SOR(A, b, x=x0, omega=1.8, tol=args.t, max_iter=args.i)
    x_sd, it_log_sd, res_log_sd, alpha_sd = solvers.SteepDesc(A, b, x=x0, tol=args.t, max_iter=args.i)
    x_cg, it_log_cg, res_log_cg, alpha_cg = solvers.ConjGrad(A, b, x=x0, tol=args.t, max_iter=args.i)

    plt.figure()
    step = 10
    plt.plot(it_log_j[::step],   res_log_j[::step],   ls="-", marker="X", alpha=1., label=rf"Jacobi ($\varrho={rho_j:.4f}$)")
    plt.plot(it_log_dj[::step],  res_log_dj[::step],  ls="-", marker="X", alpha=1., label=rf"Damped Jacobi ($\omega=0.8, \varrho={rho_dj:.4f}$)")
    plt.plot(it_log_gs[::step],  res_log_gs[::step],  ls="-", marker="X", alpha=1., label=rf"Gauss-Seidel ($\varrho={rho_gs:.4f}$)")
    plt.plot(it_log_sor[::step], res_log_sor[::step], ls="-", marker="X", alpha=1., label=rf"SOR ($\omega=1.8, \varrho={rho_sor:.4f}$)")
    plt.plot(it_log_sd[::step],  res_log_sd[::step],  ls="-", marker="X", alpha=1., label=rf"Steepest Descent")
    plt.plot(it_log_cg[::step],  res_log_cg[::step],  ls="-", marker="X", alpha=1., label=rf"Conjugate Gradient")
    plt.legend(loc="best")
    plt.yscale("log")
    plt.ylim(1e-10, 1e0)
    plt.xlabel("# of iteration")
    plt.ylabel(r"$\| r \|_{\infty}$")
    plt.title(rf"Sherman1, $x \in \mathbb{{R}}^{{1000}}, \kappa_2(A)={cond:.3e}$")
    plt.savefig("sherman1_residual.png")
    plt.close()

    # plot the solution
    plt.figure()
    plt.plot(x_j,   ls="-", marker="X", alpha=1., label=rf"Jacobi")
    plt.plot(x_dj,  ls="-", marker="X", alpha=1., label=rf"Damped Jacobi ($\omega=0.8$)")
    plt.plot(x_gs,  ls="-", marker="X", alpha=1., label=rf"Gauss-Seidel")
    plt.plot(x_sor, ls="-", marker="X", alpha=1., label=rf"SOR ($\omega=1.8$)")
    plt.plot(x_sd,  ls="-", marker="X", alpha=1., label=rf"Steepest Descent")
    plt.plot(x_cg,  ls="-", marker="X", alpha=1., label=rf"Conjugate Gradient")
    plt.legend(loc="best")
    plt.xlabel(r"index")
    plt.ylabel(r"$x$")
    plt.title(r"Sherman1, $x \in \mathbb{R}^{1000}$")
    plt.savefig("sherman1_solution.png")
    plt.close()


if __name__ == "__main__":
    plot_setting()
    main()
