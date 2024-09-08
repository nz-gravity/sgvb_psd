import matplotlib.pyplot as plt
import numpy as np
from ..utils.periodogram import get_periodogram
from sgvb_psd import true_var


def plot_psdq(psd_q, freqs, axs=None, col="C0"):
    """
    This is a utility class to plot the estimated multivariate PSDs.
    """
    nquantiles = psd_q.shape[0]
    nfreqs = psd_q.shape[1]
    p = psd_q.shape[-1]

    if nfreqs != len(freqs):
        raise ValueError(
            f"The len of frequencies {len(freqs)} does not match the len of PSD {nfreqs}")

    if axs is None:
        # generate a square figure with pxp subplots
        fig, axs = plt.subplots(p, p, figsize=(p * 2.2, p * 2.2))

    for row_i in range(p):
        for col_j in range(p):
            psd = psd_q[..., row_i, col_j]
            if row_i == col_j:
                psd = np.log(np.real(psd))
            elif row_i < col_j:  # upper triangular
                psd = np.real(psd)
            else:  # lower triangular
                psd = np.imag(psd)

            ax = axs[row_i, col_j]
            ax.fill_between(freqs, psd[0], psd[2], color=col, alpha=0.3)
            ax.plot(freqs, psd[1], color=col)

    return axs

#n = var2_data().n
#varCoef=var2_data().varCoef
#vmaCoef=var2_data().vmaCoef
#sigma=var2_data().sigma

def plot_psd(fs = 1.0, axs=None, col="C3"):
    """Plot a true PSD"""
    Simulation = true_var.VarmaSim(n=n)
    freq = (np.arange(0,np.floor_divide(n, 2), 1) / (n))
    freq = freq* fs
    spec_true = Simulation.calculateSpecMatrix(freq, varCoef, vmaCoef, sigma)
    p = spec_true.shape[-1]

    if axs is None:
        fig, axs = plt.subplots(p, p, figsize=(p * 2.2, p * 2.2))

    for row_i in range(p):
        for col_j in range(p):
            psd = spec_true[:, row_i, col_j]

            if row_i == col_j:
                psd = np.log(np.real(psd) / (2*np.pi))
            elif row_i < col_j:
                psd = np.real(psd) / fs
            else:
                psd = np.imag(psd) / fs

            ax = axs[row_i, col_j]
            ax.plot(freq, psd, color=col)

    return axs



def plot_peridogram(x, axs=None, fs=1.0, **kwargs):
    """
    This is a utility function to plot the periodogram of a time series.
    """
    n, p = x.shape
    f, pdgrm = get_periodogram(x, fs)

    # setting some default values
    kwargs['color'] = kwargs.pop("color", "lightgray")
    kwargs['zorder'] = kwargs.pop("zorder", -10)


    if axs is None:
        fig, axs = _generate_fig(p)


    for row_i in range(p):
        for col_j in range(p):
            psd = pdgrm[..., row_i, col_j]
            if row_i == col_j:
                psd = np.log(np.real(psd)/2)
            elif row_i < col_j:  # upper triangular
                psd = np.real(psd)
            else:  # lower triangular
                psd = np.imag(psd)

            ax = axs[row_i, col_j]
            ax.plot(f, psd, **kwargs)

    return axs




def _generate_fig(p):
    """
    This is a utility function to generate a figure with pxp subplots.
    """
    fig, axs = plt.subplots(p, p, figsize=(p * 2.2, p * 2.2))
    return fig, axs

