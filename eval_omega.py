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
    # from Gilbert Strang's Final Lecture
    A = np.array(
        [
            [2., 3., 4.],
            [4., 11., 14.],
            [2., 8., 17.]
        ]
    )
    b = np.array([19., 55., 50.])
    x0 = np.zeros_like(b)

    # Damped Jacobi method
    x02, it_log02, res_log02, rho02 = solvers.DampedJacobi(A, b, x0, omega=.2, tol=args.t, max_iter=args.i)
    x04, it_log04, res_log04, rho04 = solvers.DampedJacobi(A, b, x0, omega=.4, tol=args.t, max_iter=args.i)
    x06, it_log06, res_log06, rho06 = solvers.DampedJacobi(A, b, x0, omega=.6, tol=args.t, max_iter=args.i)
    x08, it_log08, res_log08, rho08 = solvers.DampedJacobi(A, b, x0, omega=.8, tol=args.t, max_iter=args.i)
    x10, it_log10, res_log10, rho10 = solvers.DampedJacobi(A, b, x0, omega=1., tol=args.t, max_iter=args.i)
    x12, it_log12, res_log12, rho12 = solvers.DampedJacobi(A, b, x0, omega=1.2, tol=args.t, max_iter=args.i)
    x14, it_log14, res_log14, rho14 = solvers.DampedJacobi(A, b, x0, omega=1.4, tol=args.t, max_iter=args.i)
    x16, it_log16, res_log16, rho16 = solvers.DampedJacobi(A, b, x0, omega=1.6, tol=args.t, max_iter=args.i)
    x18, it_log18, res_log18, rho18 = solvers.DampedJacobi(A, b, x0, omega=1.8, tol=args.t, max_iter=args.i)

    plt.figure()
    plt.plot(it_log02, res_log02, label=rf"Damped Jacobi ($\omega=0.2, \varrho={rho02:.2f}$)", ls="--", c="tab:blue", alpha=.8)
    plt.plot(it_log04, res_log04, label=rf"Damped Jacobi ($\omega=0.4, \varrho={rho04:.2f}$)", ls="--", c="tab:blue", alpha=.6)
    plt.plot(it_log06, res_log06, label=rf"Damped Jacobi ($\omega=0.6, \varrho={rho06:.2f}$)", ls="--", c="tab:blue", alpha=.4)
    plt.plot(it_log08, res_log08, label=rf"Damped Jacobi ($\omega=0.8, \varrho={rho08:.2f}$)", ls="--", c="tab:blue", alpha=.2)
    plt.plot(it_log10, res_log10, label=rf"Damped Jacobi ($\omega=1.0, \varrho={rho10:.2f}$)", ls="--", c="k", alpha=1.)
    plt.plot(it_log12, res_log12, label=rf"Damped Jacobi ($\omega=1.2, \varrho={rho12:.2f}$)", ls="--", c="tab:red", alpha=.2)
    plt.plot(it_log14, res_log14, label=rf"Damped Jacobi ($\omega=1.4, \varrho={rho14:.2f}$)", ls="--", c="tab:red", alpha=.4)
    plt.plot(it_log16, res_log16, label=rf"Damped Jacobi ($\omega=1.6, \varrho={rho16:.2f}$)", ls="--", c="tab:red", alpha=.6)
    plt.plot(it_log18, res_log18, label=rf"Damped Jacobi ($\omega=1.8, \varrho={rho18:.2f}$)", ls="--", c="tab:red", alpha=.8)
    plt.yscale("log")
    plt.ylim(1e-10, 1e0)
    plt.xlabel("# of iteration")
    plt.ylabel(r"$\| r \|_{\infty}$")
    plt.legend(loc="upper right")
    plt.title("Damped Jacobi method")
    plt.savefig("damped_jacobi.png")
    plt.close()

    # SOR method
    x02, it_log02, res_log02, rho02 = solvers.SOR(A, b, x0, omega=.2, tol=args.t, max_iter=args.i)
    x04, it_log04, res_log04, rho04 = solvers.SOR(A, b, x0, omega=.4, tol=args.t, max_iter=args.i)
    x06, it_log06, res_log06, rho06 = solvers.SOR(A, b, x0, omega=.6, tol=args.t, max_iter=args.i)
    x08, it_log08, res_log08, rho08 = solvers.SOR(A, b, x0, omega=.8, tol=args.t, max_iter=args.i)
    x10, it_log10, res_log10, rho10 = solvers.SOR(A, b, x0, omega=1., tol=args.t, max_iter=args.i)
    x12, it_log12, res_log12, rho12 = solvers.SOR(A, b, x0, omega=1.2, tol=args.t, max_iter=args.i)
    x14, it_log14, res_log14, rho14 = solvers.SOR(A, b, x0, omega=1.4, tol=args.t, max_iter=args.i)
    x16, it_log16, res_log16, rho16 = solvers.SOR(A, b, x0, omega=1.6, tol=args.t, max_iter=args.i)
    x18, it_log18, res_log18, rho18 = solvers.SOR(A, b, x0, omega=1.8, tol=args.t, max_iter=args.i)

    plt.figure()
    plt.plot(it_log02, res_log02, label=rf"SOR ($\omega=0.2, \varrho={rho02:.2f}$)", ls="--", c="tab:blue", alpha=.8)
    plt.plot(it_log04, res_log04, label=rf"SOR ($\omega=0.4, \varrho={rho04:.2f}$)", ls="--", c="tab:blue", alpha=.6)
    plt.plot(it_log06, res_log06, label=rf"SOR ($\omega=0.6, \varrho={rho06:.2f}$)", ls="--", c="tab:blue", alpha=.4)
    plt.plot(it_log08, res_log08, label=rf"SOR ($\omega=0.8, \varrho={rho08:.2f}$)", ls="--", c="tab:blue", alpha=.2)
    plt.plot(it_log10, res_log10, label=rf"SOR ($\omega=1.0, \varrho={rho10:.2f}$)", ls="--", c="k", alpha=1.)
    plt.plot(it_log12, res_log12, label=rf"SOR ($\omega=1.2, \varrho={rho12:.2f}$)", ls="--", c="tab:red", alpha=.2)
    plt.plot(it_log14, res_log14, label=rf"SOR ($\omega=1.4, \varrho={rho14:.2f}$)", ls="--", c="tab:red", alpha=.4)
    plt.plot(it_log16, res_log16, label=rf"SOR ($\omega=1.6, \varrho={rho16:.2f}$)", ls="--", c="tab:red", alpha=.6)
    plt.plot(it_log18, res_log18, label=rf"SOR ($\omega=1.8, \varrho={rho18:.2f}$)", ls="--", c="tab:red", alpha=.8)
    plt.yscale("log")
    plt.ylim(1e-10, 1e0)
    plt.xlabel("# of iteration")
    plt.ylabel(r"$\| r \|_{\infty}$")
    plt.legend(loc="upper right")
    plt.title("SOR method")
    plt.savefig("sor.png")
    plt.close()

    cond = np.linalg.cond(A, p=1)
    print(f">>> condition number of A (1-norm): {cond:.6e}")
    cond = np.linalg.cond(A, p=2)
    print(f">>> condition number of A (2-norm): {cond:.6e}")
    cond = np.linalg.cond(A, p=np.inf)
    print(f">>> condition number of A (inf-norm): {cond:.6e}")


if __name__ == "__main__":
    plot_setting()
    main()




