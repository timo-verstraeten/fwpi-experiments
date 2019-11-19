import logging
import matplotlib.pyplot as plt
import numpy as np

from utils.matrix import toggle_logger
from gp.kernels.fleet import FleetConstant

import GPy

class GPDM:

    def __init__(self, noise_var, sparse):
        self.noise_var = noise_var
        self.gps = None
        self.sparse = sparse

    def predict(self, S, A):
        M = np.zeros((S.shape[0], 1), dtype=np.int64)
        X = np.hstack((S, A, M))

        # Predict state deltas
        Mu, Var = [], []
        for gp in self.gps:
            mu, var = gp.predict(X, Y_metadata={"output_index": M})
            Mu.append(mu.flatten())
            Var.append(var.flatten())
        Mu, Var = np.array(Mu).T, np.array(Var).T  # (Ns x Na, Ds)

        return Mu + S, Var

    @toggle_logger
    def fit(self, S, A, S_):
        X_list = [np.hstack((s, a)) for s, a in zip(S, A)]
        Y_list = [s_ - s for s, s_ in zip(S, S_)]
        Dx, Dy, M = X_list[0].shape[1], Y_list[0].shape[1], len(Y_list)

        # GP dynamic model specification
        #   - one GP per state
        self.gps = []
        for d in range(Dy):
            kernel = GPy.kern.RBF(Dx, ARD=True)
            if self.sparse:
                kernel = FleetConstant(input_dim=Dx, num_outputs=M, kernel=kernel, name='coreg')
            else:
                W_rank = 1
                kernel = GPy.util.multioutput.ICM(input_dim=Dx, num_outputs=M, kernel=kernel, W_rank=W_rank, name='coreg')
            gp = GPy.models.GPCoregionalizedRegression(X_list=X_list, Y_list=[Y[:,[d]] for Y in Y_list], kernel=kernel)
            for m in range(M):
                eval('gp.mixed_noise.Gaussian_noise_%i.constrain_fixed(self.noise_var)' % m)
            gp.coreg.rbf.lengthscale.constrain_bounded(1e-5, 1e5)
            gp.coreg.rbf.variance.constrain_fixed(1)
            gp.optimize()
            self.gps.append(gp)

    def print_parameters(self):
        for d_out, gp in enumerate(self.gps):
            # Construct correlation matrix
            W = gp.coreg.B.W
            M = np.zeros((W.shape[0], W.shape[0]))
            for r in range(W.shape[1]):
                w = W[:,[r]]
                M += np.dot(w, w.T)
            M += np.diag(gp.coreg.B.kappa)
            M_diag = np.diag(1 / np.sqrt(np.diag(M)))
            M_corr = np.dot(M_diag, np.dot(M, M_diag))

            # Build output string
            strs = ["GPDM output %i: var - %.6f" % (d_out, gp.coreg.rbf.variance)]
            strs += ['ls%i - %.6f' % (d_in, ls) for d_in, ls in enumerate(gp.coreg.rbf.lengthscale)]
            strs += ['corr%i - %.6f' % (d_in, c) for d_in, c in enumerate(M_corr[0,:])]
            logging.info(', '.join(strs))

class ValueGP:

    def __init__(self, noise_var):
        self.noise_var = noise_var

        self.gp = None
        self.var = None  # Signal variance
        self.ls = None  # Length scales
        self.K_inv = None
        self.V_kern = None  # Values in kernel space

    def predict(self, S):
        MU_star, S2_star = self.gp.predict(S)
        return MU_star, S2_star

    @toggle_logger
    def fit(self, S, V):
        self.support_points, self.support_values = S, V

        kernel = GPy.kern.RBF(S.shape[1], ARD=True)
        self.gp = GPy.models.GPRegression(X=S, Y=V, kernel=kernel, noise_var=self.noise_var, normalizer=True)
        self.gp.Gaussian_noise.variance.constrain_fixed(self.noise_var)
        self.gp.optimize()

        self.K_inv = self.gp.posterior.woodbury_inv
        self.V_kern = self.gp.posterior.woodbury_vector
        self.ls = self.gp.rbf.lengthscale
        self.var = self.gp.rbf.variance[0]

    def print_parameters(self):
        strs = ["Value GP: var - %.6f" % self.var]
        strs += ['ls%i - %.6f' % (d_in, ls) for d_in, ls in enumerate(self.ls)]
        strs += ['noise - %.6f' % self.noise_var]
        logging.info(', '.join(strs))