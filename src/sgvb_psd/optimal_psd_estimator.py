from typing import Tuple

import numpy as np
import tensorflow as tf
from hyperopt import fmin, hp, tpe
from hyperopt.exceptions import AllTrialsFailed
import time


from .backend import SpecVI
from .logging import logger
from .postproc import format_axes, plot_peridogram, plot_psdq, plot_single_psd
from .utils.periodogram import get_periodogram
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
        max_hyperparm_eval: int = 100,
        psd_scaling: float = 1.0,
        fmax_for_analysis = None,
        degree_fluctuate=None,
        seed=None,
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


        """

        if seed is not None:
            set_seed(seed)
            
        if fmax_for_analysis is None:
           fmax_for_analysis = x.shape[0] / 2    

        self.N_theta = N_theta
        self.N_samples = N_samples
        self.nchunks = nchunks
        self.duration = duration
        self.ntrain_map = ntrain_map
        self.fmax_for_analysis = fmax_for_analysis
        self.x = x
        self.n, self.p = x.shape
        self.psd_scaling = psd_scaling
        self.pdgrm = get_periodogram(
            x, fs=self.sampling_freq, psd_scaling = self.psd_scaling)
        self.max_hyperparm_eval = max_hyperparm_eval
        self.degree_fluctuate = degree_fluctuate

        if self.nchunks > 1:
            logger.info(
                f"Dividing data {x.shape} into "
                f"({nchunks}, {self.n // self.nchunks}, {self.p}) chunks"
            )
            # if nchunks is not a power of 2, wil be slower
            if np.log2(nchunks) % 1 != 0:
                logger.warning("nchunks must be a power of 2 for faster FFTs")

        # Internal variables
        self.lr_map_values = []
        self.loss_values = []
        self.all_samp = []
        self.model_info = {}
        self.psd_quantiles = None
        self.psd_all = None

    def __learning_rate_optimisation_objective(self, lr):
        """Objective function for the hyperopt optimisation of the learning rate for the MAP

        :param lr: learning rate to be optimised
        :return: ELBO loss
        """
        lr_map = lr["lr_map"]
        Spec = SpecVI(self.x)
        result_list = Spec.runModel(
            N_delta=self.N_theta,  # N_delta is set to N_theta -- they must be the same
            N_theta=self.N_theta,
            lr_map=lr_map,
            ntrain_map=self.ntrain_map,
            nchunks=self.nchunks,
            duration=self.duration,
            fmax_for_analysis=self.fmax_for_analysis,
            inference_size=self.N_samples,
            degree_fluctuate=self.degree_fluctuate
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
        self.lr_map_values.append(lr_map)
        self.loss_values.append(losses[-1].numpy())
        self.all_samp.append(samp)

        return losses[-1].numpy()

    def find_optimal_surrogate_params(self):
        """
        This function is used to find the optimal learning rate
        :return: Best surrogate posterior parameters given the optimal learning rate
        """
        space = {"lr_map": hp.uniform("lr_map", 0.002, 0.02)}
        algo = tpe.suggest

        try:
            fmin(
                self.__learning_rate_optimisation_objective,
                space,
                algo=algo,
                max_evals=self.max_hyperparm_eval,

            )
            min_loss_index = self.loss_values.index(min(self.loss_values))
            self.optimal_lr = self.lr_map_values[min_loss_index]
            best_samp = self.all_samp[min_loss_index]
        except AllTrialsFailed as e:
            self.optimal_lr = self.lr_map_values[-1]
            best_samp = self.all_samp[-1]
            logger.error(f"Hyperopt failed to find optimal learning rate: {e}. Using last tested LR:{self.optimal_lr}.")
        return best_samp

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
        self.psd_all = np.linalg.inv(spectral_density_inverse_all)

        self.psd_q = np.zeros((3, num_freq, p_dim, p_dim), dtype=complex)

        diag_indices = np.diag_indices(p_dim)
        self.psd_q[:, :, diag_indices[0], diag_indices[1]] = np.quantile(
            np.real(self.psd_all[:, :, diag_indices[0], diag_indices[1]]),
            quantiles,
            axis=0,
        )

        triu_indices = np.triu_indices(p_dim, k=1)
        real_part = np.real(
            self.psd_all[:, :, triu_indices[1], triu_indices[0]]
        )
        imag_part = np.imag(
            self.psd_all[:, :, triu_indices[1], triu_indices[0]]
        )

        for i, q in enumerate(quantiles):
            self.psd_q[i, :, triu_indices[1], triu_indices[0]] = (
                np.quantile(real_part, q, axis=0)
                + 1j * np.quantile(imag_part, q, axis=0)
            ).T

        self.psd_q[:, :, triu_indices[0], triu_indices[1]] = np.conj(
            self.psd_q[:, :, triu_indices[1], triu_indices[0]]
        )

        self.psd_quantiles = self.psd_q
        self.psd_all = self.psd_all

        # changing freq from [0, 1/2] to [0, samp_freq/2] (and applying scaling)
        true_fmax = self.sampling_freq / 2
        self.psd_quantiles = (
            self.psd_quantiles / self.psd_scaling**2 / (true_fmax / 0.5)
        )
        self.psd_all = self.psd_all / self.psd_scaling**2 / (true_fmax / 0.5)

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
        self._freq = np.fft.fftfreq(self.n_per_chunk, d=1 / self.sampling_freq)

        n = self.n_per_chunk
        if np.mod(n, 2) == 0:
            # the length per chunk is even
            self._freq = self._freq[0 : int(n / 2)]
        else:
            # the length per chunk is odd
            self._freq = self._freq[0 : int((n - 1) / 2)]

        # Check if duration == 1
        if self.duration == 1:
            fmax_idx = int(self.fmax_for_analysis)  # Set fmax_idx to fmax_for_analysis
        else:
            # use fftshift to get the freq in the correct order
            fmax_idx = int(
                self.fmax_for_analysis / fmax_true * (self.n_per_chunk / 2)
            )
        return self._freq[0:fmax_idx]

    @property
    def n_per_chunk(self):
        """Return the number of points per chunk"""
        return self.x.shape[0] // self.nchunks

    @property
    def sampling_freq(self):
        """Return the sampling frequency"""
        if self.duration == 1:
            self._sampling_freq = (
                2 * np.pi
            )  # this is for the duration time is unit 1, the situation like simulation study
        else:
            self._sampling_freq = self.x.shape[0] / self.duration

        return self._sampling_freq

    def plot(self, true_psd=None, **kwargs) -> "matplotlib.pyplot.figure":
        axes = plot_psdq(self.psd_quantiles, self.freq, **kwargs)
        axes = plot_peridogram(*self.pdgrm, axs=axes, **kwargs)
        
        if true_psd is not None:
            plot_single_psd(*true_psd, axes, **kwargs)

        format_axes(axes, **kwargs)

        return axes
