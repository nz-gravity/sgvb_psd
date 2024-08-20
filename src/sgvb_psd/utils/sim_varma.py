import numpy as np
import matplotlib.pyplot as plt

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

# Example usage
if __name__ == "__main__":
    # Define VAR(2) model parameters
    ar = np.array([[0.5, 0, 0, 0], [0, -0.3, 0, -0.5]])
    n = 256
    d = 2

    # Simulate the VAR(2) process
    x = sim_varma(model='ar', coeffs=ar, n=n, d=d)

    # Plot the time series
    plt.figure(figsize=(10, 6))
    plt.plot(x)
    plt.title('Simulated VAR(2) Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend([f'Series {i+1}' for i in range(d)])
    plt.grid(True)
    plt.show()
