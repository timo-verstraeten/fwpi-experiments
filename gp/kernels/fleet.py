from GPy.kern import Coregionalize
from GPy.util.config import config  # for assesing whether to use cython

import numpy as np
import warnings

try:
    from . import coregionalize_cython
    use_coregionalize_cython = config.getboolean('cython', 'working')
except ImportError:
    print('warning in coregionalize: failed to import cython module: falling back to numpy')
    use_coregionalize_cython = False

def FleetConstant(input_dim, num_outputs, kernel, name='fleet_const'):
    if kernel.input_dim != input_dim:
        kernel.input_dim = input_dim
        warnings.warn("kernel's input dimension overwritten to fit input_dim parameter.")

    K = kernel.prod(CoregionalizeSparse(1, num_outputs, active_dims=[input_dim], name='B'), name=name)
    return K

class CoregionalizeSparse(Coregionalize):

    def __init__(self, input_dim, output_dim, active_dims=None, name='coreg_sparse'):
        rank = output_dim - 1
        E = np.identity(rank)
        E = np.vstack((np.ones(rank), E))
        W = 0.5*np.random.randn(output_dim, rank)/np.sqrt(rank)
        W *= E
        super(CoregionalizeSparse, self).__init__(input_dim, output_dim, rank=rank, W=W, active_dims=active_dims, name=name)

    def update_gradients_full(self, dL_dK, X, X2=None):
        index = np.asarray(X, dtype=np.int)
        if X2 is None:
            index2 = index
        else:
            index2 = np.asarray(X2, dtype=np.int)

        # attempt to use cython for a nasty double indexing loop: fall back to numpy
        if use_coregionalize_cython:
            dL_dK_small = self._gradient_reduce_cython(dL_dK, index, index2)
        else:
            dL_dK_small = self._gradient_reduce_numpy(dL_dK, index, index2)

        dkappa = np.diag(dL_dK_small).copy()
        dL_dK_small += dL_dK_small.T
        dW = (self.W[:, None, :] * dL_dK_small[:, :, None]).sum(0)
        dW_fleet = np.diag(np.diag(dW, k=-1), k=-1)[:,:-1]
        dW_single = np.zeros_like(dW)
        dW_single[0] = dW[0]
        dW = dW_fleet + dW_single

        self.W.gradient = dW
        self.kappa.gradient = dkappa