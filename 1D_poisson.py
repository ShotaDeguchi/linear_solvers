"""
********************************************************************************
1D Poisson equation with Dirichlet boundary conditions
********************************************************************************
"""

import os
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import utils
import solvers

parser = argparse.ArgumentParser()
parser.add_argument("-o", type=float, default=np.inf, help="order of norm")
parser.add_argument("-t", type=float, default=1e-9, help="tolerance")
parser.add_argument("-i", type=int, default=int(1e3), help="max iteration")
parser.add_argument("-n", type=int, default=4, help="problem scale")
parser.add_argument("-l", type=float, default=1., help="domain length")
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
    # 1D Poisson equation with Dirichlet boundary conditions
    N = args.n
    L = args.l
    x = np.linspace(0., args.l, N+2)
    A = utils.get_band_matrix(N, diag_val=2., off_diag_val=-1.)
    b = np.zeros(shape=(N, ))
    b[-1] = 1.
    x0 = np.zeros(shape=(N, ))
    dx = L / (N + 1)
    # A *= 1. / dx**2

    # plot the band matrix
    plt.figure()
    # plt.spy(A)
    plt.imshow(A, cmap="coolwarm", vmin=-1., vmax=1.)
    plt.savefig("band_matrix.png")
    plt.close()


    # condition number
    eigvals = np.linalg.eigvals(A)
    print(f">>> eigenvalues: {eigvals}")
    # cond = np.max(eigvals) / np.min(eigvals)
    cond = np.linalg.cond(A, p=2)
    print(f">>> condition number: {cond:.6e}")

    u1, it1, res1, rho1 = solvers.Jacobi(A, b, x=x0, ord=args.o, tol=args.t, max_iter=args.i)
    u2, it2, res2, rho2 = solvers.DampedJacobi(A, b, x=x0, ord=args.o, tol=args.t, max_iter=args.i)
    u3, it3, res3, rho3 = solvers.GaussSeidel(A, b, x=x0, ord=args.o, tol=args.t, max_iter=args.i)
    u4, it4, res4, rho4 = solvers.SOR(A, b, x=x0, ord=args.o, tol=args.t, max_iter=args.i)
    u5, it5, res5, alpha5 = solvers.SteepDesc(A, b, x=x0, ord=args.o, tol=args.t, max_iter=args.i)
    u6, it6, res6, alpha6 = solvers.ConjGrad(A, b, x=x0, ord=args.o, tol=args.t, max_iter=args.i)

    # plot
    plt.figure()
    plt.plot(it1, res1, ls="-", marker="", label=rf"Jacobi ($\varrho={rho1:.2f}$)")
    plt.plot(it2, res2, ls="--", marker="", label=rf"Damped Jacobi ($\omega=0.6, \varrho={rho2:.2f}$)")
    plt.plot(it3, res3, ls="--", marker="", label=rf"Gauss-Seidel ($\varrho={rho3:.2f}$)")
    plt.plot(it4, res4, ls="--", marker="", label=rf"SOR ($\omega=1.4, \varrho={rho4:.2f}$)")
    plt.plot(it5, res5, ls="--", marker="", label=rf"Steepest Descent")
    plt.plot(it6, res6, ls="--", marker="", label=rf"Conjugate Gradient")
    plt.legend()
    plt.yscale("log")
    plt.ylim(1e-10, 1e0)
    plt.xlabel("# of iteration")
    plt.ylabel(r"$\| r \|_{\infty}$")
    plt.title(f"1D Laplace equation, Condition number: {cond:.2f}")
    plt.savefig("1D_Laplace_residual.png")
    plt.close()

    # plot the solution
    u1 = np.concatenate([[0.], u1, [1.]])
    u2 = np.concatenate([[0.], u2, [1.]])
    u3 = np.concatenate([[0.], u3, [1.]])
    u4 = np.concatenate([[0.], u4, [1.]])
    u5 = np.concatenate([[0.], u5, [1.]])
    u6 = np.concatenate([[0.], u6, [1.]])
    plt.figure()
    plt.plot(x, u1, ls="-", marker="", label=rf"Jacobi ($\varrho={rho1:.2f}$)")
    plt.plot(x, u2, ls="--", marker="", label=rf"Damped Jacobi ($\omega=0.6, \varrho={rho2:.2f}$)")
    plt.plot(x, u3, ls="--", marker="", label=rf"Gauss-Seidel ($\varrho={rho3:.2f}$)")
    plt.plot(x, u4, ls="--", marker="", label=rf"SOR ($\omega=1.4, \varrho={rho4:.2f}$)")
    plt.plot(x, u5, ls="--", marker="", label=rf"Steepest Descent")
    plt.plot(x, u6, ls="--", marker="", label=rf"Conjugate Gradient")
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x)$")
    plt.title(f"1D Laplace equation, Condition number: {cond:.2f}")
    plt.savefig("1D_Laplace_solution.png")
    plt.close()


if __name__ == "__main__":
    plot_setting()
    main()


