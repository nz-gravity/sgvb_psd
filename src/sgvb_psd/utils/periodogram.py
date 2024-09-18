from typing import Tuple

import numpy as np
from scipy import signal
from scipy.signal import csd


def get_periodogram(x, fs, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    n, p = x.shape
    x = x
    periodogram = np.zeros((n // 2, p, p), dtype=complex)
    for row_i in range(p):
        for col_j in range(p):
            if row_i == col_j:
                # TODO: the default is 'density' -- does that make sense
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
                f, Pxx_den0 = signal.periodogram(
                    x[:, row_i], fs=fs, scaling="density", detrend=False
                )
                f = f[1:]
                Pxx_den0 = Pxx_den0[1:]
                periodogram[..., row_i, col_j] = Pxx_den0 / 2
            else:

                y = np.apply_along_axis(np.fft.fft, 0, x)
                if np.mod(n, 2) == 0:
                    # n is even
                    y = y[0 : int(n / 2)]
                else:
                    # n is odd
                    y = y[0 : int((n - 1) / 2)]
                y = y / np.sqrt(n)
                cross_spectrum_fij = y[:, row_i] * np.conj(y[:, col_j])
                cross_spectrum_fij = cross_spectrum_fij / fs
                periodogram[..., row_i, col_j] = cross_spectrum_fij

    return periodogram, f


def get_welch_periodogram(x, fs, n_chunks=1) -> Tuple[np.ndarray, np.ndarray]:
    """Given a multivariate time series, return the periodogram."""
    n, p = x.shape
    nperseg = n // n_chunks
    n, p = x.shape
    n_freq = nperseg // 2
    periodogram = np.zeros((n_freq, p, p), dtype=complex)
    for row_i in range(p):
        for col_j in range(p):
            xi = x[:, row_i]
            xj = x[:, col_j]
            f, Pxx_den0 = signal.csd(
                xi,
                xj,
                fs=fs,
                nperseg=nperseg,
                return_onesided=True,
                scaling="density",
                average="median",
                detrend=False,
            )
            periodogram[..., row_i, col_j] = Pxx_den0[1:]
    return periodogram, f[1:]
