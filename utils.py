"""
********************************************************************************
utilities
********************************************************************************
"""

import numpy as np


def get_band_matrix(N, diag_val, off_diag_val):
    diag = np.full(N, diag_val)
    off_diag = np.full(N-1, off_diag_val)
    D = np.diag(diag)
    U = np.diag(off_diag, k=1)
    L = np.diag(off_diag, k=-1)
    B = D + U + L
    return B


def get_adjacency_matrix(M, diag_val, off_diag_val):
    B = get_band_matrix(M, 4., -1.)
    I = np.eye(M)
    Z = np.zeros((M, M))
    A = np.block(
        [
            [B if i == j else I if i == j + 1 or i == j - 1 else Z for j in range(M)]
            for i in range(M)
        ]
    )
    return A


def get_rhs(M):
    # use np.kron
    b = np.kron(np.ones(M), np.array([1., 0., 0.]))
    return b

