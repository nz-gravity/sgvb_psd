import numpy as np
from typing import Tuple
import tensorflow as tf
from ..logging import logger


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
        # y_ft:   fourier transformed time series
        # freq:   frequencies w/ y_ft
        # p:  dimension of x
        # Xmat:   basis matrix
        # Zar:    arry of design matrix Z_k for every freq k
        self.x = x
        if x.shape[1] < 2:
            raise Exception("Time series should be at least 2 dimensional.")
        self.p = x.shape[1]
        self.nchunks = nchunks
        self.N_theta = N_theta
        self.N_delta = N_delta

        self.fs = fs
        self.fmax_for_analysis = fmax_for_analysis

        # Compute the required datasets
        self.y_ft, self.freq = compute_chunked_fft(self.x, self.nchunks, self.fmax_for_analysis, self.fs)
        self.Zar = _compute_chunked_Zmatrix(self.y_ft)
        Xmat_delta, Xmat_theta = _compute_Xmatrices(self.freq, N_delta, N_theta)

        # Setup tensors
        y_ft = tf.convert_to_tensor(self.y_ft, dtype=tf.complex64)
        self.y_re = tf.math.real(y_ft)
        self.y_im = tf.math.imag(y_ft)
        self.Xmat_delta = tf.convert_to_tensor(
            Xmat_delta, dtype=tf.float32
        )
        self.Xmat_theta = tf.convert_to_tensor(
            Xmat_theta, dtype=tf.float32
        )

        Zar = tf.convert_to_tensor(self.Zar, dtype=tf.complex64)
        self.Z_re = tf.math.real(Zar)
        self.Z_im = tf.math.imag(Zar)

        logger.info(f"Loaded {self}")

    def __repr__(self):
        x = self.x.shape
        y = self.y_ft.shape
        Xd = self.Xmat_delta.shape
        Xt = self.Xmat_theta.shape
        Z = self.Zar.shape
        return f"AnalysisData(x(t)={x}, y(f)={y}, Xmat_delta={Xd}, Xmat_theta={Xt}, Z={Z})"


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


def _compute_Zmatrix(y_k: np.ndarray) -> np.ndarray:
    """
    Compute the design matrix Z_k for each frequency k.

    Parameters:
    y_k (np.ndarray): Fourier transformed time series data of shape (n, p).

    Returns:
    np.ndarray: Design matrix Z_k of shape (n, p, p*(p-1)/2).
    """
    n, p = y_k.shape
    Z_k = np.zeros((n, p, int(p * (p - 1) / 2)), dtype=np.complex64)

    for j in range(n):
        count = 0
        for i in range(1, p):
            Z_k[j, i, count: count + i] = y_k[j, :i]
            count += i

    return Z_k


def _compute_chunked_Zmatrix(y_ft: np.ndarray) -> np.ndarray:
    """
    Compute the design matrix Z, a 3D array (The design matrix Z_k for every frequency k).

    Parameters:
    y_ft (np.ndarray): Fourier transformed time series data of shape (chunks, n_per_chunk, p).

    Returns:
    np.ndarray: 3D array of design matrices Z_k for each frequency k.
    """
    chunks, n_per_chunk, p = y_ft.shape
    if p == 1:
        return np.zeros((chunks, n_per_chunk, 0), dtype=np.complex64)

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
