import numpy as np
from typing import Tuple


class AnalysisData:  # Parent used to create BayesianModel object
    def __init__(
            self,
            x: np.ndarray,
            nchunks: int = 128,
            fmax_for_analysis: float = 128,
            fs: float = 2048.,
            N_theta: int = 15,
            N_delta: int = 15
    ):
        # x:      N-by-p, multivariate timeseries with N samples and p dimensions
        # ts:     time series
        # y_ft:   fourier transformed time series
        # freq:   frequencies w/ y_ft
        # p_dim:  dimension of ts
        # Xmat:   basis matrix
        # Zar:    arry of design matrix Z_k for every freq k
        self.ts = x
        if x.shape[1] < 2:
            raise Exception("Time series should be at least 2 dimensional.")

        self.p_dim = x.shape[1]
        self.nchunks = nchunks
        self.N_theta = N_theta
        self.N_delta = N_delta

        self.fs = fs
        self.fmax_for_analysis = fmax_for_analysis

        # Compute the required datasets
        self.y_ft, self.freq = compute_chunked_fft(self.ts, self.nchunks, self.fmax_for_analysis, self.fs)
        self.Zar = _compute_chunked_Zmatrix(self.y_ft)
        self.Xmat_delta, self.Xmat_theta = _compute_Xmatrices(self.freq, N_delta, N_theta)

    def __repr__(self):
        x = self.ts.shape
        y = self.y_ft.shape
        Xd = self.Xmat_delta.shape
        Xt = self.Xmat_theta.shape
        Z = self.Zar.shape
        return f"AnalysisDataset(x(t)={x}, y(f)={y}, Xmat_delta={Xd}, Xmat_theta={Xt}, Z={Z})"


def _compute_Xmatrices(freq, N_delta: int = 15, N_theta: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the Xmatrices for delta and theta

    DR_basis(y_ft$fq_y, N=10)
    cbinded X matrix

    #TODO: jianan please document this..
    """
    fstack = np.column_stack([np.repeat(1, freq.shape[0]), freq])
    Xd = np.concatenate([fstack, DR_basis(freq, N=N_delta)], axis=1)
    Xt = np.concatenate([fstack, DR_basis(freq, N=N_theta)], axis=1)
    return Xd, Xt


def _compute_Zmatrix(y_k: np.ndarray):
    """
    #TODO: jianan please document this..
    """
    n, p = y_k.shape
    Z_k = np.zeros(
        [n, p, int(p * (p - 1) / 2)], dtype=complex
    )

    for j in range(n):
        count = 0
        for i in np.arange(1, p):
            Z_k[j, i, count: count + i] = y_k[j, :i]  # .flatten()
            count += i
    return Z_k


def _compute_chunked_Zmatrix(y_ft: np.ndarray) -> np.ndarray[np.complex128]:
    """
    Compute the design matrix Z, 3d array (The design matrix Z_k for every freq k)
    #TODO: jianan please document this..
    """
    chunks, n_per_chunk, p = y_ft.shape
    if p == 1:
        return 0

    if chunks == 1:
        y_ls = np.squeeze(y_ft, axis=0)
        Z_ = _compute_Zmatrix(y_ls)
    else:
        y_ls = np.squeeze(np.split(y_ft, chunks))
        Z_ = np.array([_compute_Zmatrix(x) for x in y_ls])

    return Z_


def DR_basis(freq: np.ndarray, N=10):
    """
    Return the basis matrix for the Demmler-Reinsch basis
    for linear smoothing splines (Eubank,1999)

            # freq: vector of frequences
    # N:  amount of basis used
    # return a len(freq)-by-N matrix
    """
    return np.array(
        [
            np.sqrt(2) * np.cos(x * np.pi * freq * 2)
            for x in np.arange(1, N + 1)
        ]
    ).T


def compute_chunked_fft(x: np.ndarray, nchunks: int, fmax_for_analysis: float, fs: float) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Scaled fft and get the elements of freq = 1:[Nquist] (or 1:[fmax_for_analysis] if specified)
    discarding the rest of freqs
    """
    orig_n, p = x.shape
    if orig_n < p:
        raise ValueError(f"Number of samples {orig_n} is less than number of dimensions {p}.")
    # split x into chunks
    n_per_chunk = x.shape[0] // nchunks
    chunked_x = np.array(
        np.split(x[0: n_per_chunk * nchunks, :], nchunks)
    )
    assert chunked_x.shape == (nchunks, n_per_chunk, p)

    # compute fft for each chunk
    y_ft = np.apply_along_axis(np.fft.fft, 1, chunked_x)
    #
    # y = []
    # for i in range(nchunks):
    #     y_fft = np.apply_along_axis(np.fft.fft, 0, chunked_x[i])
    #     y.append(y_fft)
    # y = np.array(y)

    # scale it
    y_ft = y_ft / np.sqrt(n_per_chunk)
    Ts = 1  # for VB backend we use Duration of 1.0 (rescale later)
    fq_y = np.fft.fftfreq(np.size(chunked_x, axis=1), Ts)
    ftrue_y = np.fft.fftfreq(n_per_chunk, d=1 / fs)

    # Truncate the FFT'd data
    if np.mod(n_per_chunk, 2) == 0:  # n is even
        idx = int(n_per_chunk / 2)
    else:  # n is odd
        idx = int((n_per_chunk - 1) / 2)

    y_ft = y_ft[:, 0:idx, :]
    fq_y = fq_y[0:idx]
    ftrue_y = ftrue_y[0:idx]

    if fmax_for_analysis is None:
        fmax_for_analysis = ftrue_y[-1]
    fmax_idx = np.searchsorted(ftrue_y, fmax_for_analysis)
    y_ft = y_ft[:, 0:fmax_idx, :]
    fq_y = fq_y[0:fmax_idx]
    return y_ft, fq_y
