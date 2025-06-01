"""
********************************************************************************
iterative solvers for linear systems
********************************************************************************
"""

import numpy as np
import scipy as sp


def Jacobi(A, b, x, tol=1e-9, ord=np.inf, maxiter=int(1e3)):
    print("\n=== Jacobi method ===")

    # use A = D + L + U
    # D = np.diag(np.diag(A))
    # LU = A - D
    # it_log = []
    # res_log = []
    # for it in range(0, maxiter+1):
    #     x_old = np.copy(x)
    #     x = np.dot(np.linalg.inv(D), b - np.dot(LU, x))
    #     # res = np.linalg.norm(b - np.dot(A, x))
    #     res = np.linalg.norm(x - x_old, ord=ord)
    #     print(f">>> it: {it:d}, res: {res:.6e}")
    #     it_log.append(it)
    #     res_log.append(res)
    #     if res < tol:
    #         break
    # print(f">>> converged to x: {x}")
    # return x, it_log, res_log, rho

    # alternative form: use A = D - (L + U)
    D = np.diag(np.diag(A))
    LU = D - A
    M = np.dot(np.linalg.inv(D), LU)
    c = np.dot(np.linalg.inv(D), b)

    cond = np.linalg.cond(A, p=2)
    rho = np.max(np.abs(np.linalg.eigvals(M)))
    print(f">>> condition number of A: {cond:.6e}")
    print(f">>> spectral radius of M: {rho:.6e}")

    it_log = []
    res_log = []
    for it in range(0, maxiter+1):
        # x_old = np.copy(x)
        # x = np.dot(M, x) + c
        # res = np.linalg.norm(x - x_old, ord=ord)

        x = np.dot(M, x) + c
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=ord) / np.linalg.norm(b, ord=ord)
        print(f">>> it: {it:d}, res: {res:.6e}")
        it_log.append(it)
        res_log.append(res)
        if res < tol:
            break
    print(f">>> res: {res:.6e} / tol: {tol:.6e}")
    # print(f">>> converged to x: {x}")
    return x, it_log, res_log, rho


def DampedJacobi(A, b, x, omega=.6, tol=1e-9, ord=np.inf, maxiter=int(1e3)):
    print("\n=== Damped Jacobi method ===")
    # alternative form: use A = D - (L + U)
    D = np.diag(np.diag(A))
    LU = D - A
    M = omega * np.dot(np.linalg.inv(D), LU) + (1. - omega) * np.eye(x.shape[0])
    c = omega * np.dot(np.linalg.inv(D), b)

    cond = np.linalg.cond(A, p=2)
    rho = np.max(np.abs(np.linalg.eigvals(M)))
    print(f">>> condition number of A: {cond:.6e}")
    print(f">>> spectral radius of M: {rho:.6e}")

    it_log = []
    res_log = []
    for it in range(0, maxiter+1):
        x = np.dot(M, x) + c
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=ord)
        print(f">>> it: {it:d}, res: {res:.6e}")
        it_log.append(it)
        res_log.append(res)
        if res < tol:
            break
    print(f">>> res: {res:.6e} / tol: {tol:.6e}")
    # print(f">>> converged to x: {x}")
    return x, it_log, res_log, rho


def GaussSeidel(A, b, x, tol=1e-9, ord=np.inf, maxiter=int(1e3)):
    print("\n=== Gauss-Seidel method ===")
    # use A = D + L + U
    # D = np.diag(np.diag(A))
    # L = np.tril(A, k=-1)
    # U = np.triu(A, k=1)
    # it_log = []
    # res_log = []
    # for it in range(0, maxiter+1):
    #     x_old = np.copy(x)
    #     x = np.dot(np.linalg.inv(D + L), b - np.dot(U, x))
    #     res = np.linalg.norm(x - x_old, ord=ord)
    #     print(f">>> it: {it:d}, res: {res:.6e}")
    #     it_log.append(it)
    #     res_log.append(res)
    #     if res < tol:
    #         break
    # print(f">>> converged to x: {x}")
    # return x, it_log, res_log, rho

    # alternative form: use A = D - (L + U)
    D = np.diag(np.diag(A))
    L = - np.tril(A, k=-1)
    U = - np.triu(A, k=1)
    M = np.dot(np.linalg.inv(D - L), U)
    c = np.dot(np.linalg.inv(D - L), b)

    cond = np.linalg.cond(A, p=2)
    rho = np.max(np.abs(np.linalg.eigvals(M)))
    print(f">>> condition number of A: {cond:.6e}")
    print(f">>> spectral radius of M: {rho:.6e}")

    it_log = []
    res_log = []
    for it in range(0, maxiter+1):
        x = np.dot(M, x) + c
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=ord)
        print(f">>> it: {it:d}, res: {res:.6e}")
        it_log.append(it)
        res_log.append(res)
        if res < tol:
            break
    print(f">>> res: {res:.6e} / tol: {tol:.6e}")
    # print(f">>> converged to x: {x}")
    return x, it_log, res_log, rho


def SOR(A, b, x, omega=1.4, tol=1e-9, ord=np.inf, maxiter=int(1e3)):
    print("\n=== Successive Over-Relaxation method ===")
    # use A = D + L + U
    # D = np.diag(np.diag(A))
    # L = np.tril(A, k=-1)
    # U = np.triu(A, k=1)
    # it_log = []
    # res_log = []
    # for it in range(0, maxiter+1):
    #     x_old = np.copy(x)
    #     x = np.dot(np.linalg.inv(D + omega * L), (1. - omega) * np.dot(D, x) - omega * np.dot(U, x) + omega * b)
    #     res = np.linalg.norm(x - x_old, ord=ord)
    #     print(f">>> it: {it:d}, res: {res:.6e}")
    #     it_log.append(it)
    #     res_log.append(res)
    #     if res < tol:
    #         break
    # print(f">>> converged to x: {x}")
    # return x, it_log, res_log, rho

    # alternative form: use A = D - (L + U)
    D = np.diag(np.diag(A))
    L = - np.tril(A, k=-1)
    U = - np.triu(A, k=1)
    M = np.dot(np.linalg.inv(D - omega * L), (1. - omega) * D + omega * U)
    c = omega * np.dot(np.linalg.inv(D - omega * L), b)

    cond = np.linalg.cond(A, p=2)
    rho = np.max(np.abs(np.linalg.eigvals(M)))
    print(f">>> condition number of A: {cond:.6e}")
    print(f">>> spectral radius of M: {rho:.6e}")

    it_log = []
    res_log = []
    for it in range(0, maxiter+1):
        x = np.dot(M, x) + c
        r = b - np.dot(A, x)
        res = np.linalg.norm(r, ord=ord)
        print(f">>> it: {it:d}, res: {res:.6e}")
        it_log.append(it)
        res_log.append(res)
        if res < tol:
            break
    print(f">>> res: {res:.6e} / tol: {tol:.6e}")
    # print(f">>> converged to x: {x}")
    return x, it_log, res_log, rho


def SteepDesc(A, b, x, tol=1e-9, ord=np.inf, maxiter=int(1e3)):
    print("\n=== Steepest Descent method ===")
    r = b - np.dot(A, x)
    it_log = []
    res_log = []
    alpha_log = []
    for it in range(0, maxiter+1):
        Ar = np.dot(A, r)
        alpha = np.dot(r, r) / np.dot(r, Ar)
        x += alpha * r
        r -= alpha * Ar
        res = np.linalg.norm(r, ord=ord)
        print(f">>> it: {it:d}, res: {res:.6e}")
        it_log.append(it)
        res_log.append(res)
        alpha_log.append(alpha)
        if res < tol:
            break
    print(f">>> res: {res:.6e} / tol: {tol:.6e}")
    # print(f">>> converged to x: {x}")
    return x, it_log, res_log, alpha_log


def ConjDir(A, b, x, tol=1e-9, ord=np.inf, maxiter=int(1e3)):
    print("\n=== Conjugate Direction method ===")
    raise NotImplementedError


def ConjGrad(
        A, b, x,
        tol=1e-9, ord=np.inf, maxiter=int(1e3), precond=None
):
    print("\n=== Conjugate Gradient method ===")

    if precond is None:
        print(f">>> preconditioner not used")
        apply_precond = lambda r: r

    elif precond == "diag":
        print(f">>> using diagonal preconditioner (Jacobi)")
        M_inv = 1.0 / np.diag(A)
        apply_precond = lambda r: M_inv * r

    elif precond == "ilu":
        print(f">>> using Incomplete LU preconditioner")
        A_sparse = sp.sparse.csc_matrix(A)
        ilu = sp.sparse.linalg.spilu(A_sparse)
        apply_precond = lambda r: ilu.solve(r)

    else:
        raise ValueError(f"{precond} is not a valid preconditioner")

    r0 = b - A @ x
    z0 = apply_precond(r0)
    p = np.copy(z0)

    it_log = []
    res_log = []
    alpha_log = []

    for it in range(0, maxiter + 1):
        Ap = A @ p
        rz0 = np.dot(r0, z0)
        alpha = rz0 / np.dot(p, Ap)
        x += alpha * p
        r1 = r0 - alpha * Ap
        res = np.linalg.norm(r1, ord=ord)
        print(f">>> it: {it:d}, res: {res:.6e}")
        it_log.append(it)
        res_log.append(res)
        alpha_log.append(alpha)
        if res < tol:
            break
        z1 = apply_precond(r1)
        beta = np.dot(r1, z1) / rz0
        p = z1 + beta * p
        r0 = r1
        z0 = z1

    print(f">>> res: {res:.6e} / tol: {tol:.6e}")
    return x, it_log, res_log, alpha_log
