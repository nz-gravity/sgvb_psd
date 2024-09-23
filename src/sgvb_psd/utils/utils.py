import numpy as np


def get_freq(fs: float, n_time_samples: int, fmax=None) -> np.ndarray:
    n = n_time_samples
    dt = 1 / fs
    freq = np.fft.fftfreq(n, d=dt)
    if np.mod(n, 2) == 0:  # the length per chunk is even
        freq = freq[0: int(n / 2)]
    else:  # the length per chunk is odd
        freq = freq[0: int((n - 1) / 2)]

    if fmax is not None:
        fmax_idx = np.searchsorted(freq, fmax)
        freq = freq[0:fmax_idx]
    return freq
