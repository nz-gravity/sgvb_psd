import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from hyperopt import fmin, hp, tpe
from hyperopt.exceptions import AllTrialsFailed

from .backend import SpecVI
from .logging import logger
from .postproc import (
    format_axes,
    plot_coherence,
    plot_peridogram,
    plot_psdq,
    plot_single_psd,
)
from .utils.periodogram import get_periodogram, get_welch_periodogram
from .utils.tf_utils import set_seed


class OptimalPSDEstimator:
    """
    This class is used to run SGVB and estimate the posterior PSD

    This works in a few steps
    1. Optimize the learning rate for determining maximised posterior+ELBO (ie surrogate posterior == unnormlised posterior)
    2. Use optimized learning rate to estimate the posterior PSD

    The main interface is run() which returns the posterior PSD and the quantiles of the PSD.
    """

    def __init__(
        self,
        x: np.ndarray,
        N_theta: int = 30,
        nchunks: int = 1,
        duration: float = 1.0,
        ntrain_map=10000,
        N_samples: int = 500,
        fs=1.0,
        max_hyperparm_eval: int = 100,
        fmax_for_analysis=None,
        degree_fluctuate=None,
        seed=None,
        lr_range = (0.002, 0.02),
    ):
        """
        :param x: the input multivariate time series
        :param N_theta: the number of basis functions for the theta component
        :param nchunks: the number of blocks that multivariate time series is divided into
        :param duration: the total observation time
        :param ntrain_map: the number of iterations in gradient ascent for MAP
        :param N_samples: the number of parameters sampled from the surrogate distribution
        :param max_hyperparm_eval: the number of evaluations in 'Hyperopt'
        :param psd_scaling: the scale size of the input data
        :param fmax_for_analysis: the maximum frequency in the frequency domain that needs to be analyzed.
        :param degree_fluctuate: a hyperparameter from the prior,
               which should be set to a large value when dealing with a large number of basis functions.
        """

        if seed is not None:
            set_seed(seed)

        self.N_theta = N_theta
        self.N_samples = N_samples
        self.nchunks = nchunks
        self.duration = duration
        self.ntrain_map = ntrain_map
        self.fmax_for_analysis = fmax_for_analysis
        self.sampling_freq = fs
        self.lr_range = lr_range

        # normalize the data
        self.psd_scaling = np.std(x, axis=0)
        self.psd_offset = np.mean(x, axis=0)
        self.x = (x - self.psd_offset) / self.psd_scaling
        self.n, self.p = x.shape

        self.pdgrm, self.pdgrm_freq = get_periodogram(
            self.x, fs=self.sampling_freq
        )
        self.pdgrm = (self.pdgrm * self.psd_scaling**2)
        self.max_hyperparm_eval = max_hyperparm_eval
        self.degree_fluctuate = degree_fluctuate

        if self.nchunks > 1:
            logger.info(
                f"Dividing data {self.x.shape} into "
                f"({self.nchunks}, {self.nt_per_chunk}, {self.p}) chunks"
            )
            # if nchunks is not a power of 2, wil be slower
            if np.log2(nchunks) % 1 != 0:
                logger.warning("nchunks must be a power of 2 for faster FFTs")

        if self.fmax_for_analysis is not None:
            if self.fmax_for_analysis > self.sampling_freq / 2:
                raise ValueError(
                    f"fmax_for_analysis ({fmax_for_analysis}) must be less than or equal to the Nyquist frequency: {self.sampling_freq / 2}"
                )
            else:
                logger.info(
                    f"Reducing the number of frequencies to be analyzed from "
                    f"{self.nt_per_chunk//2} to {self.fmax_for_analysis}..."
                )
        logger.info(
            f"Final PSD will be of shape: {self.nfreq_per_chunk} x {self.p} x {self.p}"
        )

        # Internal variables
        self.model_info = {}
        self.psd_quantiles = None
        self.psd_all = None
        self.Spec = SpecVI(self.x)

    def __learning_rate_optimisation_objective(self, lr):
        """Objective function for the hyperopt optimisation of the learning rate for the MAP

        :param lr: learning rate to be optimised
        :return: ELBO loss
        """
        lr_map = lr["lr_map"]

        result_list = self.Spec.runModel(
            N_theta=self.N_theta,
            lr_map=lr_map,
            ntrain_map=self.ntrain_map,
            nchunks=self.nchunks,
            fmax_for_analysis=self.fmax_for_analysis,
            inference_size=self.N_samples,
            degree_fluctuate=self.degree_fluctuate,
            fs=self.sampling_freq,
        )

        losses = result_list[0]
        samp = result_list[2]

        if self.model_info == {}:
            (
                self.model_info["Xmat_delta"],
                self.model_info["Xmat_theta"],
            ) = result_list[1].Xmtrix(
                N_delta=self.N_theta, N_theta=self.N_theta
            )
            self.model_info["p_dim"] = result_list[1].p_dim

        # TODO: this is taking too much memory -- do we need to store all of this?
        # dont we just need the best point?
        current_loss = losses[-1].numpy()
        if not hasattr(self, "best_loss") or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.optimal_lr = lr_map
            self.best_samp = samp  # Update best sample
    
        return current_loss

    def find_optimal_surrogate_params(self):
        """
        This function is used to find the optimal learning rate
        :return: Best surrogate posterior parameters given the optimal learning rate
        """
        space = {"lr_map": hp.uniform("lr_map", *self.lr_range)}
        algo = tpe.suggest

        try:
            fmin(
                self.__learning_rate_optimisation_objective,
                space,
                algo=algo,
                max_evals=self.max_hyperparm_eval,
            )
            
        except AllTrialsFailed as e:
            logger.error(
                f"Hyperopt failed to find optimal learning rate: {e}. Using last tested LR:{self.optimal_lr}."
            )
        logger.info(f"Optimal learning rate: {self.optimal_lr}")
        return self.best_samp

    def _compute_spectral_density(
        self, post_sample: np.ndarray, quantiles=[0.05, 0.5, 0.95]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function is used to compute the spectral density given best surrogate posterior parameters
        :param post_sample: the surrogate posterior parameters

        Computes:
            1. self.psd_q: the quantiles of the spectral density [n-quantiles, n-freq, dim, dim]
            2. self.psd_all: Nsamp instances of the spectral density [Nsamp, n-freq, dim, dim]

        """
        Xmat_delta = self.model_info["Xmat_delta"]
        Xmat_theta = self.model_info["Xmat_theta"]
        p_dim = self.model_info["p_dim"]

        delta2_all_s = tf.exp(
            tf.matmul(Xmat_delta, tf.transpose(post_sample[0], [0, 2, 1]))
        )  # (500, #freq, p)

        theta_re_s = tf.matmul(
            Xmat_theta, tf.transpose(post_sample[2], [0, 2, 1])
        )  # (500, #freq, p(p-1)/2)
        theta_im_s = tf.matmul(
            Xmat_theta, tf.transpose(post_sample[4], [0, 2, 1])
        )

        theta_all_s = -(
            tf.complex(theta_re_s, theta_im_s)
        )  # (500, #freq, p(p-1)/2)
        theta_all_np = theta_all_s.numpy()

        D_all = tf.map_fn(
            lambda x: tf.linalg.diag(x), delta2_all_s
        ).numpy()  # (500, #freq, p, p)

        num_slices, num_freq, num_elements = theta_all_np.shape
        row_indices, col_indices = np.tril_indices(p_dim, k=-1)
        diag_matrix = np.eye(p_dim, dtype=np.complex64)
        T_all = np.tile(diag_matrix, (num_slices, num_freq, 1, 1))
        T_all[:, :, row_indices, col_indices] = theta_all_np.reshape(
            num_slices, num_freq, -1
        )

        T_all_conj_trans = np.conj(np.transpose(T_all, axes=(0, 1, 3, 2)))
        D_all_inv = np.linalg.inv(D_all)

        spectral_density_inverse_all = T_all_conj_trans @ D_all_inv @ T_all
        psd_all = (
            np.linalg.inv(spectral_density_inverse_all) * self.psd_scaling**2
        )

        psd_q = np.zeros((3, num_freq, p_dim, p_dim), dtype=complex)

        diag_indices = np.diag_indices(p_dim)
        psd_q[:, :, diag_indices[0], diag_indices[1]] = np.quantile(
            np.real(psd_all[:, :, diag_indices[0], diag_indices[1]]),
            quantiles,
            axis=0,
        )

        triu_indices = np.triu_indices(p_dim, k=1)
        real_part = np.real(psd_all[:, :, triu_indices[1], triu_indices[0]])
        imag_part = np.imag(psd_all[:, :, triu_indices[1], triu_indices[0]])

        for i, q in enumerate(quantiles):
            psd_q[i, :, triu_indices[1], triu_indices[0]] = (
                np.quantile(real_part, q, axis=0)
                + 1j * np.quantile(imag_part, q, axis=0)
            ).T

        psd_q[:, :, triu_indices[0], triu_indices[1]] = np.conj(
            psd_q[:, :, triu_indices[1], triu_indices[0]]
        )

        # changing freq from [0, 1/2] to [0, samp_freq/2] (and applying scaling)
        true_fmax = self.sampling_freq / 2
        self.psd_quantiles = psd_q / (true_fmax / 0.5)
        self.psd_all = psd_all / (true_fmax / 0.5)

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Running hyperopt to find optimal learning rate")
        t0 = time.time()
        best_samp = self.find_optimal_surrogate_params()
        t1 = time.time()
        logger.info(f"Optimal learning rate found in {t1 - t0:.2f}s")
        logger.info("Computing optimal PSD estimation")
        t0 = time.time()
        self._compute_spectral_density(best_samp)
        t1 = time.time()
        logger.info(f"Optimal PSD estimation complete in {t1 - t0:.2f}s")
        return self.psd_all, self.psd_quantiles

    @property
    def freq(self):
        """Return the freq per chunk of the PSD estimate"""
        if hasattr(self, "_freq"):
            return self._freq

        fmax_true = self.sampling_freq / 2
        dt = 1 / self.sampling_freq
        self._freq = np.fft.fftfreq(self.nt_per_chunk, d=dt)
        n = self.nt_per_chunk
        if np.mod(n, 2) == 0:
            # the length per chunk is even
            self._freq = self._freq[0 : int(n / 2)]
        else:
            # the length per chunk is odd
            self._freq = self._freq[0 : int((n - 1) / 2)]

        if self.fmax_for_analysis != None:
            fmax_idx = np.searchsorted(self._freq, self.fmax_for_analysis)
            self._freq = self._freq[0:fmax_idx]

        return self._freq

    @property
    def nt_per_chunk(self):
        """Return the number of points per chunk"""
        return self.n // self.nchunks

    @property
    def nfreq_per_chunk(self):
        """Return the number of frequencies per chunk"""
        return len(self.freq)

    def plot(self, true_psd=None, plot_periodogram=True, **kwargs) -> np.ndarray[plt.Axes]:
        axes = plot_psdq(self.psd_quantiles, self.freq, **kwargs)
        if plot_periodogram:
            axes = plot_peridogram(self.pdgrm, self.pdgrm_freq, axs=axes, **kwargs)

        if true_psd is not None:
            plot_single_psd(*true_psd, axes, **kwargs)

        format_axes(axes, **kwargs)

        return axes

    def plot_coherence(self, true_psd=None, **kwargs) -> np.ndarray[plt.Axes]:
        labels = kwargs.pop("labels", "123456789")
        ax = plot_coherence(self.psd_all, self.freq, **kwargs, labels=labels)
        if true_psd is not None:
            ax = plot_coherence(
                true_psd[0], true_psd[1], **kwargs, ax=ax, ls="--", color="k"
            )
        return ax
