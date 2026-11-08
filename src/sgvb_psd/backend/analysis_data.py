from typing import Tuple

import numpy as np
import tensorflow as tf

from ..logging import logger
from .._spectral import (
    DR_basis,
    _compute_Xmatrices,
    _compute_Zmatrix,
    _compute_chunked_Zmatrix,
    compute_chunked_fft,
)


class AnalysisData:  # Parent used to create BayesianModel object
    def __init__(
        self,
        x: np.ndarray,
        nchunks: int = 128,
        fmax_for_analysis: float = 128,
        fs: float = 2048.0,
        N_theta: int = 15,
        N_delta: int = 15,
        fmin_for_analysis: float = None,
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

        # Compute the required datasets
        self.y_ft, self.freq = compute_chunked_fft(
            self.x,
            self.nchunks,
            self.fmax_for_analysis,
            self.fs,
            self.fmin_for_analysis,
        )

        # Sum the block-level Fourier outer products once.  The blocked
        # Whittle likelihood only depends on the blocks through this matrix,
        # so it can be reused at every MAP and VI iteration.
        block_periodograms = self.y_ft[:, :, :, None] * np.conjugate(
            self.y_ft[:, :, None, :]
        )
        summed_periodogram = np.sum(block_periodograms, axis=0)
        del block_periodograms

        # The diagonal is real by construction; remove round-off imaginary
        # parts before converting the matrix to a TensorFlow tensor.
        diag_idx = np.arange(summed_periodogram.shape[-1])
        summed_periodogram[:, diag_idx, diag_idx] = np.real(
            summed_periodogram[:, diag_idx, diag_idx]
        )

        self.Zar = _compute_chunked_Zmatrix(self.y_ft)
        Xmat_delta, Xmat_theta = _compute_Xmatrices(
            self.freq, N_delta, N_theta
        )

        # Setup tensors
        y_ft = tf.convert_to_tensor(self.y_ft, dtype=tf.complex64)
        self.y_re = tf.math.real(y_ft)
        self.y_im = tf.math.imag(y_ft)
        self.summed_periodogram = tf.convert_to_tensor(
            summed_periodogram, dtype=tf.complex64
        )
        self.Xmat_delta = tf.convert_to_tensor(Xmat_delta, dtype=tf.float32)
        self.Xmat_theta = tf.convert_to_tensor(Xmat_theta, dtype=tf.float32)

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
