from scipy import signal
import numpy as np

def get_periodogram(x, fs):
    """Given a multivariate time series, return the periodogram."""
    n, p = x.shape
    periodogram = np.zeros((1+n//2, p, p))
    for row_i in range(p):
        for col_j in range(p):
            if row_i == col_j:
                f, periodogram[..., row_i, col_j] = signal.periodogram(x[:, row_i], fs=fs)
            else:
                f, periodogram[..., row_i, col_j] = signal.csd(x[:, row_i], x[:, col_j], fs=fs)
    return f, periodogram
