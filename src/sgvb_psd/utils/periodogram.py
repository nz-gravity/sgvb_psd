import numpy as np
from scipy import signal


def get_chunked_median_periodogram(x, fs, n_chunks=1) -> np.ndarray:
    """Given a multivariate time series, return the periodogram."""
    chunked_x = np.array(np.array_split(x, n_chunks))
    _, n, p = chunked_x.shape

    assert (
        n > p
    ), "The number of samples must be greater than the number of variables."

    f = None
    pdgm = np.zeros((n_chunks, n // 2, p, p))
    for i in range(n_chunks):
        pdgm[i], f = get_one_periodogram(x, fs)
    pdgm = np.median(pdgm, axis=0)

    assert pdgm.shape == (n // 2, p, p)
    return pdgm, f


def get_periodogram(x, fs, psd_scaling, **kwargs):
    n, p = x.shape
    x = x/psd_scaling
    periodogram = np.zeros((n // 2, p, p), dtype=complex)
    for row_i in range(p):
        for col_j in range(p):
            if row_i == col_j:
                f, Pxx_den0 = signal.periodogram(x[:, row_i], fs=fs)
                f = f[1:]
                Pxx_den0 = Pxx_den0[1:]
                periodogram[..., row_i, col_j] = Pxx_den0/2
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
