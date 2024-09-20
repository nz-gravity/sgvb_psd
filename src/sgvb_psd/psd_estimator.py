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
from .postproc.plot_psd import plot_psd
from .utils.periodogram import get_periodogram
from .utils.tf_utils import set_seed


class PSDEstimator:
    """
    A class for estimating the posterior Power Spectral Density (PSD) using Stochastic Gradient Variational Bayes (SGVB).

    This class implements a two-step process:
    1. Optimize the learning rate to maximize the posterior and Evidence Lower Bound (ELBO).
    2. Use the optimized learning rate to estimate the posterior PSD.

    The main interface is the run() method, which returns the posterior PSD and its quantiles.

    :ivar N_theta: Number of basis functions for the theta component.
    :vartype N_theta: int
    :ivar N_samples: Number of parameters sampled from the surrogate distribution.
    :vartype N_samples: int
    :ivar nchunks: Number of blocks the multivariate time series is divided into.
    :vartype nchunks: int
    :ivar ntrain_map: Number of iterations in gradient ascent for Maximum A Posteriori (MAP) estimation.
    :vartype ntrain_map: int
    :ivar fs: Sampling frequency of the input data.
    :vartype fs: float
    :ivar lr_range: Range of learning rates to consider during optimization.
    :vartype lr_range: tuple
    :ivar psd_scaling: Scaling factor for the input data.
    :vartype psd_scaling: numpy.ndarray
    :ivar psd_offset: Offset for the input data.
    :vartype psd_offset: numpy.ndarray
    :ivar x: Normalized input multivariate time series.
    :vartype x: numpy.ndarray
    :ivar n: Number of time points in the input data.
    :vartype n: int
    :ivar p: Number of variables in the multivariate time series.
    :vartype p: int
    :ivar fmax_for_analysis: Maximum frequency in the frequency domain to be analyzed.
    :vartype fmax_for_analysis: int
    :ivar pdgrm: Periodogram of the input data.
    :vartype pdgrm: numpy.ndarray
    :ivar pdgrm_freq: Frequencies corresponding to the periodogram.
    :vartype pdgrm_freq: numpy.ndarray
    :ivar max_hyperparm_eval: Number of evaluations in hyperparameter optimization.
    :vartype max_hyperparm_eval: int
    :ivar degree_fluctuate: Hyperparameter from the prior, used when dealing with a large number of basis functions.
    :vartype degree_fluctuate: float
    :ivar model: Trained model object.
    :vartype model: object
    :ivar samps: Samples drawn from the posterior distribution.
    :vartype samps: numpy.ndarray
    :ivar vi_losses: Variational Inference losses during training.
    :vartype vi_losses: numpy.ndarray
    :ivar psd_quantiles: Quantiles of the estimated PSD.
    :vartype psd_quantiles: numpy.ndarray
    :ivar psd_all: All estimated PSDs.
    :vartype psd_all: numpy.ndarray
    :ivar inference_runner: Object for running the variational inference.
    :vartype inference_runner: ViRunner
    :ivar optimal_lr: Optimized learning rate.
    :vartype optimal_lr: float
    """

    def __init__(
        self,
        x,
        N_theta=30,
        nchunks=1,
        ntrain_map=10000,
        N_samples=500,
        fs=1.0,
        max_hyperparm_eval=100,
        fmax_for_analysis=None,
        degree_fluctuate=None,
        seed=None,
        lr_range=(0.002, 0.02),
    ):
        """
        Initialize the PSDEstimator.

        :param x: Input multivariate time series.
        :type x: numpy.ndarray
        :param N_theta: Number of basis functions for the theta component, defaults to 30.
        :type N_theta: int, optional
        :param nchunks: Number of blocks to divide the multivariate time series into, defaults to 1.
        :type nchunks: int, optional
        :param ntrain_map: Number of iterations in gradient ascent for MAP, defaults to 10000.
        :type ntrain_map: int, optional
        :param N_samples: Number of parameters sampled from the surrogate distribution, defaults to 500.
        :type N_samples: int, optional
        :param fs: Sampling frequency, defaults to 1.0.
        :type fs: float, optional
        :param max_hyperparm_eval: Number of evaluations in hyperparameter optimization, defaults to 100.
        :type max_hyperparm_eval: int, optional
        :param fmax_for_analysis: Maximum frequency to analyze in the frequency domain, defaults to None.
        :type fmax_for_analysis: int, optional
        :param degree_fluctuate: Hyperparameter from the prior, defaults to None.
        :type degree_fluctuate: float, optional
        :param seed: Random seed for reproducibility, defaults to None.
        :type seed: int, optional
        :param lr_range: Range of learning rates to consider during optimization, defaults to (0.002, 0.02).
        :type lr_range: tuple, optional
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

        :param lr: Learning rate for MAP. If None, optimal rate is found, defaults to None.
        :type lr: float, optional
        :return: Tuple containing the posterior PSD and quantiles of the PSD.
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
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
        self,
        true_psd=None,
        plot_periodogram=True,
        tick_ln=5,
        diag_spline_thickness=2,
        xlims=None,
        diag_ylims=None,
        off_ylims=None,
        diag_log=True,
        off_symlog=True,
        sylmog_thresh=1e-49,
        **kwargs,
    ) -> np.ndarray[plt.Axes]:
        """
        Plot the estimated PSD, periodogram, and true PSD (if provided).

        :param true_psd: True PSD and freq to plot for comparison (true_psd, true_freq)
        :type true_psd: tuple, optional
        :param plot_periodogram: Whether to plot the periodogram
        :type plot_periodogram: bool
        :param tick_ln: Length of the ticks, defaults to 5
        :type tick_ln: int, optional
        :param diag_spline_thickness: Thickness of the diagonal spline, defaults to 2
        :type diag_spline_thickness: int, optional
        :param xlims: Limits for the x-axis
        :type xlims: tuple, optional
        :param diag_ylims: Limits for the diagonal
        :type diag_ylims: tuple, optional
        :param off_ylims: Limits for the off-diagonal
        :type off_ylims: tuple, optional
        :param diag_log: Whether to use a log scale for the diagonal, defaults to True
        :type diag_log: bool, optional
        :param off_symlog: Whether to use a symlog scale for the off-diagonal, defaults to True
        :type off_symlog: bool, optional
        :param sylmog_thresh: Threshold for symlog, defaults to 1e-49
        :type sylmog_thresh: float, optional
        :return: Matplotlib Axes object
        :rtype: numpy.ndarray
        """
        all_kwargs = dict(
            tick_ln=tick_ln,
            diag_spline_thickness=diag_spline_thickness,
            xlims=xlims,
            diag_ylims=diag_ylims,
            off_ylims=off_ylims,
            diag_log=diag_log,
            off_symlog=off_symlog,
            sylmog_thresh=sylmog_thresh,
            **kwargs,
        )
        pdgrm = [self.pdgrm, self.pdgrm_freq] if plot_periodogram else None
        return plot_psd(
            psdq=[self.psd_quantiles, self.freq],
            pdgrm=pdgrm,
            true_psd=true_psd,
            **all_kwargs,
        )

    def plot_coherence(self, true_psd=None, **kwargs) -> np.ndarray[plt.Axes]:
        """
        Plot the coherence of the estimated PSD.

        :param true_psd: True PSD to plot for comparison
        :type true_psd: tuple, optional
        :param kwargs: Additional keyword arguments for plotting
        :return: Matplotlib Axes object
        :rtype: numpy.ndarray[plt.Axes]
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

        :return: Matplotlib Axes object
        :rtype: plt.Axes
        """
        plt.plot(self.vi_losses)
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        # use exponential offset  y-axis
        plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        return plt.gca()
