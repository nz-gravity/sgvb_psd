import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from hyperopt import fmin, hp, tpe

from .backend import ViRunner
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


class PSDEstimator:
    """
    PSDEstimator: A class for estimating the posterior Power Spectral Density (PSD) using Stochastic Gradient Variational Bayes (SGVB).

    This class implements a two-step process:
    1. Optimize the learning rate to maximize the posterior and Evidence Lower Bound (ELBO).
    2. Use the optimized learning rate to estimate the posterior PSD.

    The main interface is the run() method, which returns the posterior PSD and its quantiles.

    Attributes:
        N_theta (int): Number of basis functions for the theta component.
        N_samples (int): Number of parameters sampled from the surrogate distribution.
        nchunks (int): Number of blocks the multivariate time series is divided into.
        ntrain_map (int): Number of iterations in gradient ascent for Maximum A Posteriori (MAP) estimation.
        fs (float): Sampling frequency of the input data.
        lr_range (tuple): Range of learning rates to consider during optimization.
        psd_scaling (np.ndarray): Scaling factor for the input data.
        psd_offset (np.ndarray): Offset for the input data.
        x (np.ndarray): Normalized input multivariate time series.
        n (int): Number of time points in the input data.
        p (int): Number of variables in the multivariate time series.
        fmax_for_analysis (int): Maximum frequency in the frequency domain to be analyzed.
        pdgrm (np.ndarray): Periodogram of the input data.
        pdgrm_freq (np.ndarray): Frequencies corresponding to the periodogram.
        max_hyperparm_eval (int): Number of evaluations in hyperparameter optimization.
        degree_fluctuate (float): Hyperparameter from the prior, used when dealing with a large number of basis functions.
        model (object): Trained model object.
        samps (np.ndarray): Samples drawn from the posterior distribution.
        vi_losses (np.ndarray): Variational Inference losses during training.
        psd_quantiles (np.ndarray): Quantiles of the estimated PSD.
        psd_all (np.ndarray): All estimated PSDs.
        inference_runner (ViRunner): Object for running the variational inference.
        optimal_lr (float): Optimized learning rate.
    """

    def __init__(
        self,
        x: np.ndarray,
        N_theta: int = 30,
        nchunks: int = 1,
        ntrain_map=10000,
        N_samples: int = 500,
        fs=1.0,
        max_hyperparm_eval: int = 100,
        fmax_for_analysis=None,
        degree_fluctuate=None,
        seed=None,
        lr_range=(0.002, 0.02),
    ):
        """
        Initialize the PSDEstimator.

        Args:
            x (np.ndarray): Input multivariate time series.
            N_theta (int): Number of basis functions for the theta component.
            nchunks (int): Number of blocks to divide the multivariate time series into.
            ntrain_map (int): Number of iterations in gradient ascent for MAP.
            N_samples (int): Number of parameters sampled from the surrogate distribution.
            fs (float): Sampling frequency.
            max_hyperparm_eval (int): Number of evaluations in hyperparameter optimization.
            fmax_for_analysis (int): Maximum frequency to analyze in the frequency domain.
            degree_fluctuate (float): Hyperparameter from the prior.
            seed (int): Random seed for reproducibility.
            lr_range (tuple): Range of learning rates to consider during optimization.
        """

        if seed is not None:
            set_seed(seed)

        self.N_theta = N_theta
        self.N_samples = N_samples
        self.nchunks = nchunks
        self.ntrain_map = ntrain_map

        self.fs = fs
        self.lr_range = lr_range

        # normalize the data
        self.psd_scaling = np.std(x, axis=0)
        self.psd_offset = np.mean(x, axis=0)
        self.x = (x - self.psd_offset) / self.psd_scaling
        self.n, self.p = x.shape
        if fmax_for_analysis is None:
            fmax_for_analysis = self.n // 2

        self.fmax_for_analysis = fmax_for_analysis

        self.pdgrm, self.pdgrm_freq = get_periodogram(self.x, fs=self.fs)
        self.pdgrm = self.pdgrm * self.psd_scaling**2
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

        if self.fmax_for_analysis < self.nt_per_chunk // 2:
            logger.info(
                f"Reducing the number of frequencies to be analyzed from "
                f"{self.nt_per_chunk // 2} to {self.fmax_for_analysis}..."
            )

        logger.info(
            f"Final PSD will be of shape: {self.nfreq_per_chunk} x {self.p} x {self.p}"
        )

        # Internal variables
        self.model = None
        self.samps = None
        self.vi_losses = None
        self.psd_quantiles = None
        self.psd_all = None
        self.inference_runner = ViRunner(
            self.x,
            N_theta=self.N_theta,
            nchunks=self.nchunks,
            fmax_for_analysis=self.fmax_for_analysis,
            degree_fluctuate=self.degree_fluctuate,
            fs=self.fs,
        )

    def __learning_rate_optimisation_objective(self, lr):
        """
        Objective function for hyperparameter optimization of the learning rate for MAP.

        Args:
            lr (dict): Dictionary containing the learning rate to be optimized.

        Returns:
            float: ELBO loss.
        """
        vi_losses, _, _ = self.inference_runner.runModel(
            lr_map=lr["lr_map"],
            ntrain_map=self.ntrain_map,
            inference_size=self.N_samples,
        )
        return vi_losses[-1].numpy()

    def __find_optimal_learing_rate(self):
        """
        Find the optimal learning rate using hyperopt.

        This method uses the TPE algorithm to optimize the learning rate.
        """
        self.optimal_lr = fmin(
            self.__learning_rate_optimisation_objective,
            space={"lr_map": hp.uniform("lr_map", *self.lr_range)},
            algo=tpe.suggest,
            max_evals=self.max_hyperparm_eval,
        )["lr_map"]
        logger.info(f"Optimal learning rate: {self.optimal_lr}")

    def __train_model(self):
        """
        Train the model using the optimal learning rate.

        This method runs the variational inference to estimate the posterior PSD.
        """
        vi_losses, model, samples = self.inference_runner.runModel(
            lr_map=self.optimal_lr,
            ntrain_map=self.ntrain_map,
            inference_size=self.N_samples,
        )
        self.model = model
        self.samps = samples
        self.vi_losses = vi_losses.numpy()

    def run(self, lr=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the SGVB algorithm to estimate the posterior PSD.

        This method either uses a provided learning rate or finds the optimal one,
        then trains the model and computes the posterior PSD.

        Args:
            lr (float, optional): Learning rate for MAP. If None, optimal rate is found.

        Returns:
            tuple: (posterior PSD, quantiles of the PSD)
        """
        if lr:
            logger.info(f"Using provided learning rate: {lr}")
            self.optimal_lr = lr
        else:
            logger.info("Running hyperopt to find optimal learning rate")
            t0 = time.time()
            self.__find_optimal_learing_rate()
            t1 = time.time()
            logger.info(f"Optimal learning rate found in {t1 - t0:.2f}s")

        logger.info("Training model")
        t0 = time.time()
        self.__train_model()
        t1 = time.time()
        logger.info(f"Model trained in {t1 - t0:.2f}s")

        logger.info("Computing posterior PSDs")
        t0 = time.time()
        self.psd_all, self.psd_quantiles = self.model.compute_psd(
            self.samps, psd_scaling=self.psd_scaling, fs=self.fs
        )
        t1 = time.time()
        logger.info(f"Optimal PSD estimation complete in {t1 - t0:.2f}s")
        return self.psd_all, self.psd_quantiles

    @property
    def freq(self) -> np.ndarray:
        """
        Get the frequencies per chunk of the PSD estimate.

        Returns:
            np.ndarray: Array of frequencies.
        """
        if hasattr(self, "_freq"):
            return self._freq

        dt = 1 / self.fs
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
    def nt_per_chunk(self) -> int:
        """Return the number of time-points per chunk"""
        return self.n // self.nchunks

    @property
    def nfreq_per_chunk(self) -> int:
        """Return the number of frequencies per chunk"""
        return len(self.freq)

    def plot(
        self, true_psd=None, plot_periodogram=True, **kwargs
    ) -> np.ndarray[plt.Axes]:
        """
        Plot the estimated PSD, periodogram, and true PSD (if provided).

        Args:
            true_psd (tuple, optional): True PSD to plot for comparison.
            plot_periodogram (bool): Whether to plot the periodogram.
            tick_ln=5: Length of the ticks.
            diag_spline_thickness=2: Thickness of the diagonal spline.
            xlims=None: Limits for the x-axis.
            diag_ylims=None: Limits for the diagonal.
            off_ylims=None: Limits for the off-diagonal.
            diag_log=True: Whether to use a log scale for the diagonal.
            off_symlog=True: Whether to use a symlog scale for the off-diagonal.
            sylmog_thresh=1e-49: Threshold for symlog.

        """
        axes = plot_psdq(self.psd_quantiles, self.freq, **kwargs)
        if plot_periodogram:
            axes = plot_peridogram(
                self.pdgrm, self.pdgrm_freq, axs=axes, **kwargs
            )

        if true_psd is not None:
            plot_single_psd(*true_psd, axes, **kwargs)

        format_axes(axes, **kwargs)

        return axes

    def plot_coherence(self, true_psd=None, **kwargs) -> np.ndarray[plt.Axes]:
        """
        Plot the coherence of the estimated PSD.

        Args:
            true_psd (tuple, optional): True PSD to plot for comparison.
            **kwargs: Additional keyword arguments for plotting.

        Returns:
            plt.Axes: Matplotlib Axes object.
        """
        labels = kwargs.pop("labels", "123456789")
        ax = plot_coherence(self.psd_all, self.freq, **kwargs, labels=labels)
        if true_psd is not None:
            ax = plot_coherence(
                true_psd[0], true_psd[1], **kwargs, ax=ax, ls="--", color="k"
            )
        return ax

    def plot_vi_losses(self) -> plt.Axes:
        """
        Plot the variational inference losses.

        Returns:
            plt.Axes: Matplotlib Axes object.
        """
        plt.plot(self.vi_losses)
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        # use exponential offset  y-axis
        plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        return plt.gca()
