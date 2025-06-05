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
    x3, it_log3, res_log3, alpha_log3 = solvers.ConjGrad(
        A, b, x=np.zeros_like(b),
        tol=args.t, ord=args.o, maxiter=args.i, precond="ilu"
    )

    # also verify with scipy solver
    it_log4 = []
    res_log4 = []
    def callback(x):
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=args.o)
        res_log4.append(res)
        it_log4.append(len(res_log4))
    x4, info = sp.sparse.linalg.cg(
        A, b, x0=np.zeros_like(b),
        rtol=args.t, maxiter=args.i,
        callback=callback, M=None
    )

    it_log5 = []
    res_log5 = []
    def callback(x):
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=args.o)
        res_log5.append(res)
        it_log5.append(len(res_log5))
    Dinv = 1. / np.diag(A)
    M = sp.sparse.linalg.LinearOperator(
        A.shape, matvec=lambda x: Dinv * x
    )  # preconditioner as a linear operator
    x5, info = sp.sparse.linalg.cg(
        A, b, x0=np.zeros_like(b),
        rtol=args.t, maxiter=args.i,
        callback=callback, M=M
    )

    it_log6 = []
    res_log6 = []
    def callback(x):
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=args.o)
        res_log6.append(res)
        it_log6.append(len(res_log6))
    A_sparse = sp.sparse.csc_matrix(A)
    ilu = sp.sparse.linalg.spilu(A_sparse)
    M = sp.sparse.linalg.LinearOperator(
        A.shape, matvec=lambda x: ilu.solve(x)
    )
    x6, info = sp.sparse.linalg.cg(
        A, b, x0=np.zeros_like(b),
        rtol=args.t, maxiter=args.i,
        callback=callback, M=M
    )

    fig, ax = plt.subplots()
    step = 1
    ax.plot(it_log4[::step], res_log4[::step], ls="-", c="b", marker="", alpha=.5, label="SciPy CG w/o precond")
    ax.plot(it_log5[::step], res_log5[::step], ls="-", c="g", marker="", alpha=.5, label="SciPy CG w/ precond (diag)")
    ax.plot(it_log6[::step], res_log6[::step], ls="-", c="r", marker="", alpha=.5, label="SciPy CG w/ precond (ilu)")

    ax.plot(it_log1[::step], res_log1[::step], ls=":", c="b", marker="", alpha=1., label="Impl CG w/o precond")
    ax.plot(it_log2[::step], res_log2[::step], ls=":", c="g", marker="", alpha=1., label="Impl CG w/ precond (diag)")
    ax.plot(it_log3[::step], res_log3[::step], ls=":", c="r", marker="", alpha=1., label="Impl CG w/ precond (ilu)")
    ax.legend(loc="best")
    ax.set(
        xscale="linear",
        yscale="log",
        xlim=(0-1e2, args.i+1e2),
        ylim=(1e-12, 1e0),
        xlabel=r"# of iteration",
        ylabel=r"residual norm",
        title=r"Sherman1, $x \in \mathbb{R}^{1000}$",
    )
    fig.tight_layout()
    fig.savefig("sherman1_precond.png")
    plt.close(fig)

    print(f"np.allclose(x1, x2): {np.allclose(x1, x2)}")
    print(f"np.allclose(x2, x3): {np.allclose(x2, x3)}")

    print(f"np.allclose(x4, x5): {np.allclose(x4, x5)}")
    print(f"np.allclose(x5, x6): {np.allclose(x5, x6)}")

    print(f"it_log1[-1]: {it_log1[-1]:03d}, res_log1[0]: {res_log1[0]:.6e}, res_log1[-1]: {res_log1[-1]:.6e}")
    print(f"it_log2[-1]: {it_log2[-1]:03d}, res_log2[0]: {res_log2[0]:.6e}, res_log2[-1]: {res_log2[-1]:.6e}")
    print(f"it_log3[-1]: {it_log3[-1]:03d}, res_log3[0]: {res_log3[0]:.6e}, res_log3[-1]: {res_log3[-1]:.6e}")
    print(f"it_log4[-1]: {it_log4[-1]:03d}, res_log4[0]: {res_log4[0]:.6e}, res_log4[-1]: {res_log4[-1]:.6e}")
    print(f"it_log5[-1]: {it_log5[-1]:03d}, res_log5[0]: {res_log5[0]:.6e}, res_log5[-1]: {res_log5[-1]:.6e}")
    print(f"it_log6[-1]: {it_log6[-1]:03d}, res_log6[0]: {res_log6[0]:.6e}, res_log6[-1]: {res_log6[-1]:.6e}")

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
