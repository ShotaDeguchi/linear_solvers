"""
********************************************************************************
sherman problems
1: symmetric
2-5: asymetric
********************************************************************************
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmread


def plot_setting():
    plt.style.use("default")
    plt.style.use("seaborn-v0_8-deep")
    plt.style.use("seaborn-v0_8-talk")   # paper / notebook / talk / poster
    # plt.style.use("classic")
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = True
    plt.rcParams['axes.axisbelow'] = True   # background grid
    plt.rcParams["grid.alpha"] = .3
    plt.rcParams["legend.framealpha"] = .8
    plt.rcParams["legend.facecolor"] = "w"
    plt.rcParams["savefig.dpi"] = 300


def main():
    path_sherman = Path("sherman_data")
    names = [
        "sherman1",
        "sherman2",
        "sherman3",
        "sherman4",
        "sherman5",
    ]
    for name in names:
        # read matrix A and vector b
        path = path_sherman / name
        A = mmread(path / f"{name}.mtx").toarray()
        b = mmread(path / f"{name}_rhs1.mtx").squeeze()

        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_subplot(121)
        ax.spy(A)
        ax.set_title(rf"{name}: $\bm{{A}}$")

        ax = fig.add_subplot(122)
        ax.plot(b)
        ax.set_title(rf"{name}: $\bm{{b}}$")

        fig.tight_layout()
        fig.savefig(path / f"{name}_A_b.png", bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    plot_setting()
    main()
