from scipy import signal
import numpy as np

def get_periodogram(x, fs):
    """Given a multivariate time series, return the periodogram."""
    
    n, p = x.shape
    periodogram = np.zeros((n//2, p, p), dtype=complex)
    for row_i in range(p):
        for col_j in range(p):
            if row_i == col_j:
                f, Pxx_den0 = signal.periodogram(x[:, row_i], fs=fs)
                f = f[1:]
                Pxx_den0 = Pxx_den0[1:] 
                periodogram[...,row_i,col_j] = Pxx_den0
            else:
                
                y = np.apply_along_axis(np.fft.fft, 0, x)
                if np.mod(n, 2) == 0:
                    # n is even
                    y = y[0:int(n/2)]
                else:
                    # n is odd
                    y = y[0:int((n-1)/2)]
                y = y / np.sqrt(n)
                cross_spectrum_fij = y[:, row_i] * np.conj(y[:, col_j])
                cross_spectrum_fij = cross_spectrum_fij / fs
                periodogram[..., row_i, col_j] = cross_spectrum_fij
                
    return periodogram, f
