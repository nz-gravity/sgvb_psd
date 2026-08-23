from typing import Tuple

import numpy as np
import tensorflow as tf

from ..logging import logger


class EigenbasisAnalysisData:
    def __init__(
        self,
        x: np.ndarray,
        nchunks: int = 128,
        fmax_for_analysis: float = 128,
        fs: float = 2048.0,
        N_theta: int = 15,
        N_delta: int = 15,
        fmin_for_analysis: float = None,
        fmin_idx_extension: int = 0,
        fmax_idx_extension: int = 32,
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
        self.fmin_for_analysis = fmin_for_analysis
        self.fmin_idx_extension = fmin_idx_extension
        self.fmax_idx_extension = fmax_idx_extension

        # Compute the required datasets
        (
            self.y_ft,
            self.freq,
            self.u,
            self.output_keep_mask,
        ) = compute_chunked_fft(
            self.x,
            self.nchunks,
            self.fmax_for_analysis,
            self.fs,
            self.fmin_for_analysis,
            self.fmin_idx_extension,
            self.fmax_idx_extension,
        )
        self.Zar = _compute_Zmatrix_from_u(self.u)
        Xmat_delta, Xmat_theta = _compute_Xmatrices(
            self.freq, N_delta, N_theta
        )

        # Setup tensors
        y_ft = tf.convert_to_tensor(self.y_ft, dtype=tf.complex64)
        self.y_re = tf.math.real(y_ft)
        self.y_im = tf.math.imag(y_ft)
        self.Xmat_delta = tf.convert_to_tensor(Xmat_delta, dtype=tf.float32)
        self.Xmat_theta = tf.convert_to_tensor(Xmat_theta, dtype=tf.float32)

        Zar = tf.convert_to_tensor(self.Zar, dtype=tf.complex64)
        self.Z_re = tf.math.real(Zar)
        self.Z_im = tf.math.imag(Zar)

        u = tf.convert_to_tensor(self.u, dtype=tf.complex64)
        self.u_re = tf.math.real(u)
        self.u_im = tf.math.imag(u)

        logger.info(f"Loaded {self}")

    def __repr__(self):
        x = self.x.shape
        y = self.y_ft.shape
        Xd = self.Xmat_delta.shape
        Xt = self.Xmat_theta.shape
        Z = self.Zar.shape
        return f"EigenbasisAnalysisData(x(t)={x}, y(f)={y}, Xmat_delta={Xd}, Xmat_theta={Xt}, Z={Z})"


def _compute_Xmatrices(
    freq, N_delta: int = 15, N_theta: int = 15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the X matrices for delta and theta based on the provided frequencies.

    Parameters:
    freq (np.ndarray): vector of frequencies
    N_delta (int): The number of basis functions to use for delta (default is 15).
    N_theta (int): The number of basis functions to use for theta (default is 15).

    Returns:
    Xd (np.ndarray): The design matrix for Demmler-Reinsch basis functions of delta,
                     the shape is (n, 2 + N_delta).
    Xt (np.ndarray): The design matrix for Demmler-Reinsch basis functions of theta,
                     the shape is (n, 2 + N_theta).

    """
    fstack = np.column_stack([np.repeat(1, freq.shape[0]), freq])
    Xd = np.concatenate([fstack, DR_basis(freq, N=N_delta)], axis=1)
    Xt = np.concatenate([fstack, DR_basis(freq, N=N_theta)], axis=1)
    return Xd, Xt


def _compute_Zmatrix_from_uk(u_k: np.ndarray) -> np.ndarray:
    """
    Compute the design matrix Z_k from u_k for one frequency.

    Parameters
    ----------
    u_k : np.ndarray
        Array of shape (p, p), where u_k[:, nu] is the nu-th vector
        at the current frequency.

    Returns
    -------
    np.ndarray
        Design matrix of shape (p, p, p*(p-1)//2).
        Z_k[nu, j, :] is built from u_k[:, nu].
    """
    p, p2 = u_k.shape
    if p != p2:
        raise ValueError(f"Expected u_k.shape = (p, p), got {u_k.shape}")

    n_theta = p * (p - 1) // 2
    Z_k = np.zeros((p, p, n_theta), dtype=np.complex64)

    for nu in range(p):
        vec = u_k[:, nu]
        count = 0
        for j in range(1, p):
            Z_k[nu, j, count : count + j] = vec[:j]
            count += j

    return Z_k


def _compute_Zmatrix_from_u(u: np.ndarray) -> np.ndarray:
    """
    Compute the design matrices for all frequencies and eigen-components.

    Parameters
    ----------
    u : np.ndarray
        Array of shape (n_freq, p, p).

    Returns
    -------
    np.ndarray
        Array of shape (n_freq, p, p, p*(p-1)//2), interpreted as
        Z[freq, nu, j, theta].
    """
    return np.array([_compute_Zmatrix_from_uk(u_k) for u_k in u])


def DR_basis(freq: np.ndarray, N=10):
    """
    Return the basis matrix for the Demmler-Reinsch basis
    for linear smoothing splines (Eubank,1999)

            # freq: vector of frequencies
    # N:  amount of basis used
    # return a len(freq)-by-N matrix
    """
    return np.array(
        [
            np.sqrt(2) * np.cos(x * np.pi * freq * 2)
            for x in np.arange(1, N + 1)
        ]
    ).T


def compute_chunked_fft(
    x: np.ndarray,
    nchunks: int,
    fmax_for_analysis: float,
    fs: float,
    fmin_for_analysis: float = None,
    fmin_idx_extension: int = 0,
    fmax_idx_extension: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scaled fft and get the elements of freq = 1:[Nquist] (or 1:[fmax_for_analysis] if specified)
    discarding the rest of freqs
    """

    orig_n, p = x.shape
    if orig_n < p:
        raise ValueError(
            f"Number of samples {orig_n} is less than number of dimensions {p}."
        )
    # split x into chunks
    n_per_chunk = x.shape[0] // nchunks
    chunked_x = np.array(np.split(x[0 : n_per_chunk * nchunks, :], nchunks))
    assert chunked_x.shape == (nchunks, n_per_chunk, p)

    # chunked_x = chunked_x - np.mean(chunked_x, axis=1, keepdims=True)

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

    y_ft = y_ft[:, 1:idx, :]
    fq_y = fq_y[1:idx]
    ftrue_y = ftrue_y[1:idx]

    if fmax_for_analysis is None:
        fmax_idx = len(ftrue_y)
    else:
        fmax_idx = np.searchsorted(ftrue_y, fmax_for_analysis)
    fmin_idx = 0
    if fmin_for_analysis is not None:
        fmin_idx = np.searchsorted(ftrue_y, fmin_for_analysis)

    padded_fmin_idx = fmin_idx
    if fmin_for_analysis is not None:
        padded_fmin_idx = max(fmin_idx - int(fmin_idx_extension), 0)

    padded_fmax_idx = fmax_idx
    if fmax_for_analysis is not None:
        padded_fmax_idx = min(fmax_idx + int(fmax_idx_extension), len(ftrue_y))

    output_keep_mask = np.zeros(padded_fmax_idx - padded_fmin_idx, dtype=bool)
    output_start = fmin_idx - padded_fmin_idx
    output_end = output_start + (fmax_idx - fmin_idx)
    output_keep_mask[output_start:output_end] = True

    y_ft = y_ft[:, padded_fmin_idx:padded_fmax_idx, :]
    fq_y = fq_y[padded_fmin_idx:padded_fmax_idx]

    block_periodograms = y_ft[:, :, :, None] * np.conjugate(
        y_ft[:, :, None, :]
    )
    summed_periodogram = np.sum(block_periodograms, axis=0)
    diag_idx = np.arange(summed_periodogram.shape[-1])
    summed_periodogram[:, diag_idx, diag_idx] = summed_periodogram[
        :, diag_idx, diag_idx
    ].real

    eigvals, eigvecs = np.linalg.eigh(summed_periodogram)
    eigvals = np.clip(eigvals.real, a_min=0.0, a_max=None)
    u = eigvecs * np.sqrt(eigvals)[:, None, :]

    return y_ft, fq_y, u, output_keep_mask
