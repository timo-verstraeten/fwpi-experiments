import itertools
import logging
import numpy as np
import pyDOE
import scipy as sp

def toggle_logger(f):
    def wrapper(*args, **kwargs):
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        f(*args, **kwargs)
        logging.getLogger().setLevel(level)
    return wrapper

def chol_inverse(L):
    return chol_solve(L, np.identity(L.shape[0]))

def chol_solve(L, Y):
    L_inv = sp.linalg.solve_triangular(L, Y, lower=True)
    X = sp.linalg.solve_triangular(L.T, L_inv, lower=False)
    return X

def generate_support_mesh(bounds, N):
    lhc = pyDOE.lhs(bounds.shape[0], N, 'c')
    mesh = np.array([lhc[:,d] * (b_max - b_min) + b_min for d, (b_min, b_max) in enumerate(bounds)]).T
    return mesh

def generate_full_support_mesh(bounds):
    support = np.array([np.linspace(*b) for b in bounds])
    mesh = np.array(list(itertools.product(*support)))
    return mesh