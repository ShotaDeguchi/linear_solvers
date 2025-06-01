"""
********************************************************************************
iterative solvers for linear systems (asymmetric)
********************************************************************************
"""

import numpy as np

# ==============================================================================

def ConjGradSq(A, b, x, tol=1e-9, ord=np.inf, maxiter=int(1e3)):
    """
    Conjugate Gradient Squared
    """
    raise NotImplementedError

# ==============================================================================

def BiConjGrad(A, b, x, tol=1e-9, ord=np.inf, maxiter=int(1e3)):
    """
    Bi-Conjugate Gradient
    """
    raise NotImplementedError

# ==============================================================================

def BiConjGradStab(A, b, x, tol=1e-9, ord=np.inf, maxiter=int(1e3)):
    """
    Bi-Conjugate Gradient Stabilized
    """
    raise NotImplementedError

