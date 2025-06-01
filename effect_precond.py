"""
********************************************************************************
verify the effect of preconditioning
********************************************************************************
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.io import mmread

import utils
import solvers

parser = argparse.ArgumentParser()
parser.add_argument("-t", type=float, default=1e-9, help="tolerance")
parser.add_argument("-o", type=int, default=np.inf, help="norm order")
parser.add_argument("-i", type=int, default=int(1e3), help="max iteration")
args = parser.parse_args()

################################################################################

def main():
    # Sherman1: Sherman's oil reservoir problem
    A = mmread("sherman1.mtx").toarray()
    b = mmread("sherman1_rhs1.mtx").squeeze()

    # verify the effect of preconditioning
    x1, it_log1, res_log1, alpha_log1 = solvers.ConjGrad(
        A, b, x=np.zeros_like(b),
        tol=args.t, ord=args.o, maxiter=args.i, precond=None
    )
    x2, it_log2, res_log2, alpha_log2 = solvers.ConjGrad(
        A, b, x=np.zeros_like(b),
        tol=args.t, ord=args.o, maxiter=args.i, precond="diag"
    )
    print(f"all close: {np.allclose(x1, x2)}")

    # also verify with scipy solver
    it_log3 = []
    res_log3 = []
    def callback(x):
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=args.o)
        res_log3.append(res)
        it_log3.append(len(res_log3))
    x3, info = sp.sparse.linalg.cg(
        A, b, x0=np.zeros_like(b),
        rtol=args.t, maxiter=args.i,
        callback=callback, M=None
    )

    it_log4 = []
    res_log4 = []
    def callback(x):
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=args.o)
        res_log4.append(res)
        it_log4.append(len(res_log4))
    Dinv = 1. / np.diag(A)
    M = sp.sparse.linalg.LinearOperator(
        A.shape, matvec=lambda x: Dinv * x
    )  # preconditioner as a linear operator
    x4, info = sp.sparse.linalg.cg(
        A, b, x0=np.zeros_like(b),
        rtol=args.t, maxiter=args.i,
        callback=callback, M=M
    )

    fig, ax = plt.subplots()
    step = 10
    ax.plot(it_log1[::step], res_log1[::step], ls="-", c="b", marker="o", label="Impl CG w/o precond")
    ax.plot(it_log2[::step], res_log2[::step], ls="-", c="r", marker="s", label="Impl CG w/ precond (diag)")

    ax.plot(it_log3[::step], res_log3[::step], ls=":", c="c", marker="v", label="SciPy CG w/o precond")
    ax.plot(it_log4[::step], res_log4[::step], ls=":", c="m", marker="^", label="SciPy CG w/ precond (diag)")
    ax.legend(loc="best")
    ax.set(
        xscale="linear",
        yscale="log",
        # xlim=(0, args.i),
        # ylim=(1e-10, 1e2),
        xlabel="# of iteration",
        ylabel=r"$\| r \|_{\infty}$",
        title=r"Sherman1, $x \in \mathbb{R}^{1000}$",
    )
    fig.tight_layout()
    fig.savefig("sherman1_precond.png")
    plt.close(fig)

    print(f"it_log1[-1]: {it_log1[-1]:03d}, res_log1[-1]: {res_log1[-1]:.6e}")
    print(f"it_log2[-1]: {it_log2[-1]:03d}, res_log2[-1]: {res_log2[-1]:.6e}")
    print(f"it_log3[-1]: {it_log3[-1]:03d}, res_log3[-1]: {res_log3[-1]:.6e}")
    print(f"it_log4[-1]: {it_log4[-1]:03d}, res_log4[-1]: {res_log4[-1]:.6e}")

################################################################################

def plot_setting():
    plt.style.use("default")
    # plt.style.use("seaborn-v0_8-deep")
    plt.style.use("seaborn-v0_8-talk")   # paper / notebook / talk / poster
    # plt.style.use("classic")
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = True
    plt.rcParams['axes.axisbelow'] = True   # background grid
    plt.rcParams["grid.alpha"] = .3
    plt.rcParams["legend.framealpha"] = .8
    plt.rcParams["legend.facecolor"] = "w"
    plt.rcParams["savefig.dpi"] = 300

################################################################################

if __name__ == "__main__":
    plot_setting()
    main()
