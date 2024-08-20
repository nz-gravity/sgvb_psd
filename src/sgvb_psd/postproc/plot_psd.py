import matplotlib.pyplot as plt
import numpy as np
from ..utils.periodogram import get_periodogram


def plot_psdq(psd_q, freqs, axs=None, col="C0"):
    """
    This is a utility class to plot the estimated multivariate PSDs.
    """
    nquantiles = psd_q.shape[0]
    nfreqs = psd_q.shape[1]
    p = psd_q.shape[-1]

    if nfreqs != len(freqs):
        raise ValueError("The number of frequencies does not match the number of PSDs")

    if axs is None:
        # generate a square figure with pxp subplots
        fig, axs = plt.subplots(p, p, figsize=(p * 2.2, p * 2.2))

    for row_i in range(p):
        for col_j in range(p):
            psd = psd_q[..., row_i, col_j]
            if row_i == col_j:
                psd = np.real(psd)
            elif row_i < col_j:  # upper triangular
                psd = np.real(psd)
            else:  # lower triangular
                psd = np.imag(psd)

            ax = axs[row_i, col_j]
            ax.fill_between(freqs, psd[0], psd[2], color=col, alpha=0.3)
            ax.plot(freqs, psd[1], color=col)

    return axs




def plot_peridogram(x, col="lightgray", axs=None, fs=1.0):
    """
    This is a utility function to plot the periodogram of a time series.
    """
    n, p = x.shape
    f, pdgrm = get_periodogram(x, fs)


    if axs is None:
        fig, axs = _generate_fig(p)


    for row_i in range(p):
        for col_j in range(p):
            psd = pdgrm[..., row_i, col_j]
            if row_i == col_j:
                psd = np.real(psd)
            elif row_i < col_j:  # upper triangular
                psd = np.real(psd)
            else:  # lower triangular
                psd = np.imag(psd)

            ax = axs[row_i, col_j]
            ax.plot(f, psd[1], color=col)

    return axs




def _generate_fig(p):
    """
    This is a utility function to generate a figure with pxp subplots.
    """
    fig, axs = plt.subplots(p, p, figsize=(p * 2.2, p * 2.2))
    return fig, axs

