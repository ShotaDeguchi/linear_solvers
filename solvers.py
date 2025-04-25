"""
********************************************************************************
iterative solvers for linear systems
********************************************************************************
"""

import numpy as np


def Jacobi(A, b, x, tol=1e-9, ord=np.inf, max_iter=int(1e3)):
    print("\n=== Jacobi method ===")

    # use A = D + L + U
    # D = np.diag(np.diag(A))
    # LU = A - D
    # it_log = []
    # res_log = []
    # for it in range(0, max_iter+1):
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
    for it in range(0, max_iter+1):
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


def DampedJacobi(A, b, x, omega=.6, tol=1e-9, ord=np.inf, max_iter=int(1e3)):
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
    for it in range(0, max_iter+1):
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


def GaussSeidel(A, b, x, tol=1e-9, ord=np.inf, max_iter=int(1e3)):
    print("\n=== Gauss-Seidel method ===")
    # use A = D + L + U
    # D = np.diag(np.diag(A))
    # L = np.tril(A, k=-1)
    # U = np.triu(A, k=1)
    # it_log = []
    # res_log = []
    # for it in range(0, max_iter+1):
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
    for it in range(0, max_iter+1):
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


def SOR(A, b, x, omega=1.4, tol=1e-9, ord=np.inf, max_iter=int(1e3)):
    print("\n=== Successive Over-Relaxation method ===")
    # use A = D + L + U
    # D = np.diag(np.diag(A))
    # L = np.tril(A, k=-1)
    # U = np.triu(A, k=1)
    # it_log = []
    # res_log = []
    # for it in range(0, max_iter+1):
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
    for it in range(0, max_iter+1):
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


def SteepDesc(A, b, x, tol=1e-9, ord=np.inf, max_iter=int(1e3)):
    print("\n=== Steepest Descent method ===")
    r = b - np.dot(A, x)
    it_log = []
    res_log = []
    alpha_log = []
    for it in range(0, max_iter+1):
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


def ConjDir(A, b, x, tol=1e-9, ord=np.inf, max_iter=int(1e3)):
    print("\n=== Conjugate Direction method ===")
    raise NotImplementedError


def ConjGrad(A, b, x, tol=1e-9, ord=np.inf, max_iter=int(1e3)):
    print("\n=== Conjugate Gradient method ===")
    r0 = b - np.dot(A, x)
    p = np.copy(r0)
    it_log = []
    res_log = []
    alpha_log = []
    for it in range(0, max_iter+1):
        Ap = np.dot(A, p)
        alpha = np.dot(r0, r0) / np.dot(p, Ap)
        x += alpha * p
        r1 = r0 - alpha * Ap
        res = np.linalg.norm(r1, ord=ord)
        print(f">>> it: {it:d}, res: {res:.6e}")
        it_log.append(it)
        res_log.append(res)
        alpha_log.append(alpha)
        if res < tol:
            break
        beta = np.dot(r1, r1) / np.dot(r0, r0)
        p = r1 + beta * p
        r0 = np.copy(r1)
    print(f">>> res: {res:.6e} / tol: {tol:.6e}")
    # print(f">>> converged to x: {x}")
    return x, it_log, res_log, alpha_log
