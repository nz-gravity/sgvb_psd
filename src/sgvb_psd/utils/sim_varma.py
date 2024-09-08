import numpy as np
import matplotlib.pyplot as plt

'''
def rmvnorm(n, d, mu=None, Sigma=None, **kwargs):
    """
    Simulate from a Multivariate Normal Distribution.

    Parameters:
    - n: sample size
    - d: dimensionality
    - mu: mean vector
    - Sigma: covariance matrix
    - kwargs: additional arguments

    Returns:
    - An n by d matrix with one sample in each row.
    """
    if mu is None:
        mu = np.zeros(d)
    if Sigma is None:
        Sigma = np.eye(d)
    return np.random.multivariate_normal(mu, Sigma, n)

def sim_varma(model, coeffs, n, d, burnin=10000, **kwargs):
    """
    Simulate from a VARMA model.

    Parameters:
    - model: 'ar' or 'ma'
    - coeffs: coefficients for the VAR or VMA model
    - n: sample size
    - d: dimensionality
    - burnin: length of burn-in period
    - kwargs: additional arguments

    Returns:
    - An n by d matrix with one sample in each row.
    """
    if model not in ['ar', 'ma']:
        raise ValueError("Model must be either 'ar' or 'ma'")

    if model == 'ar':
        ar = np.array(coeffs)
        ma = np.empty((d, 0))
    else:
        ma = np.array(coeffs)
        ar = np.empty((d, 0))

    if ar.shape[0] != d or (ar.shape[1] % d != 0 and ar.shape[1] != 0):
        raise ValueError("AR coefficients must have correct dimensions")
    if ma.shape[0] != d or (ma.shape[1] % d != 0 and ma.shape[1] != 0):
        raise ValueError("MA coefficients must have correct dimensions")

    p = ar.shape[1] // d
    q = ma.shape[1] // d
    if burnin < max(p, q):
        raise ValueError("Burn-in period must be at least max(p, q)")

    X_sim = np.full((n + burnin, d), np.nan)
    epsilon_sim = np.full((n + burnin, d), np.nan)

    if max(p, q) > 0:
        X_sim[:max(p, q), :] = rmvnorm(max(p, q), d, **kwargs)
        epsilon_sim[:max(p, q), :] = rmvnorm(max(p, q), d, **kwargs)

    for t in range(max(p, q), n + burnin):
        epsilon_sim[t, :] = rmvnorm(1, d, **kwargs).flatten()
        X_sim[t, :] = epsilon_sim[t, :]

        for j in range(1, p + 1):
            X_sim[t, :] += ar[:, (j - 1) * d:j * d] @ X_sim[t - j, :]

        for j in range(1, q + 1):
            X_sim[t, :] += ma[:, (j - 1) * d:j * d] @ epsilon_sim[t - j, :]

    return X_sim[burnin:]
'''

def sim_varma(n, varCoef, vmaCoef, sigma=np.array([1.])):
    
    dim = vmaCoef.shape[1]
    lag_ma = vmaCoef.shape[0]
    lag_ar = varCoef.shape[0]
    
    if sigma.shape[0] == 1:
        Sigma = np.identity(dim) * sigma
    else:
        Sigma = sigma
    
    x_init = np.array(np.zeros(shape = [lag_ar+1, dim]))
    x = np.empty((n+101, dim))
    x[:] = np.NaN
    x[:lag_ar+1, ] = x_init
    epsilon = np.random.multivariate_normal(np.repeat(0., dim), Sigma, size=[lag_ma,])
    for i in np.arange(lag_ar+1, x.shape[0]):
        epsilon = np.concatenate([np.random.multivariate_normal(np.repeat(0., dim), Sigma, size=[1,]), epsilon[:-1]])
        x[i,] = np.sum(np.matmul(varCoef, x[i-1:i-lag_ar-1:-1][...,np.newaxis]), axis=(0, -1)) + \
                np.sum(np.matmul(vmaCoef, epsilon[...,np.newaxis]), axis=(0, -1)) 
    x = x[101: ]        
    return x















