import numpy as np

def standardize(X):
    X_mu, X_std = np.mean(X, axis=0), np.std(X, axis=0)
    return (X - X_mu)/X_std, X_mu, X_std

def standardize_batch(batch):
    S, A, S_ = batch
    S, S_mu, S_std = standardize(S)
    A, A_mu, A_std = standardize(A)
    S_ = (S_ - S_mu) / S_std
    return (S, A, S_), (S_mu, S_std), (A_mu, A_std)