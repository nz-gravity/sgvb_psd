import numpy as np


def get_freq(
    fs: float, n_time_samples: int, fmax=None, fmin=None
) -> np.ndarray:
    n = n_time_samples
    dt = 1 / fs
    freq = np.fft.fftfreq(n, d=dt)
    if np.mod(n, 2) == 0:  # the length per chunk is even
        freq = freq[1 : int(n / 2)]
    else:  # the length per chunk is odd
        freq = freq[1 : int((n - 1) / 2)]

    fmin_idx = 0
    fmax_idx = len(freq)

    if fmin is not None:
        fmin_idx = np.searchsorted(freq, fmin)
    if fmax is not None:
        fmax_idx = np.searchsorted(freq, fmax)

    fmin_idx = int(np.clip(fmin_idx, 0, len(freq)))
    fmax_idx = int(np.clip(fmax_idx, 0, len(freq)))
    if fmax_idx < fmin_idx:
        fmax_idx = fmin_idx

    return freq[fmin_idx:fmax_idx]
