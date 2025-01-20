import itertools
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
import numpy as np

from ..logging import logger


def matrix_combinations(p):
    indices = range(p)
    combinations = itertools.combinations(indices, 2)
    result = [((i, j), (i, i), (j, j)) for i, j in combinations]
    return result


def uniformmax_help(sample):
    return np.abs(sample - np.median(sample)) / median_abs_deviation(sample)


def uniformmax_multi(coh_all):
    N_sample, N = coh_all.shape
    C_help = np.zeros((N_sample, N))

    for j in range(N):
        C_help[:, j] = uniformmax_help(coh_all[:, j])

    return np.max(C_help, axis=0)



def compute_coherence(pxx, pyy, pxy):
    coh_q = np.real(np.abs(pxy) ** 2) / np.real(pxx) / np.real(pyy)
    coh_uniform = None
    if len(coh_q.shape) > 1:
        coh_all = coh_q
        coh_q = np.quantile(coh_all, [0.05, 0.5, 0.95], axis=0)

        coh_median = coh_q[1]
        mad = median_abs_deviation(coh_all, axis=0, nan_policy='omit')
        mad[mad == 0] = 1e-10
        max_std_abs_dev = uniformmax_multi(coh_all)
        threshold = np.quantile(max_std_abs_dev, 0.9)
        coh_lower = coh_median - threshold * mad
        coh_upper = coh_median + threshold * mad
        coh_uniform = np.stack([coh_lower, coh_median, coh_upper], axis=0)

    return coh_q, coh_uniform





def plot_coherence(psd, freq, labels=None, ax=None, color=None, ls="-"):
    p = psd.shape[-1]
    combinations = matrix_combinations(p)
    if ax is None:
        _, ax = plt.subplots(1, 1)

    for idx, comb in enumerate(combinations):
        (i, j), (ii, _), (jj, _) = comb
        coh_q, coh_uniform = compute_coherence(
            psd[..., ii, ii], psd[..., jj, jj], psd[..., i, j]
        )
        if labels is not None:
            l = r"$C_{" + f"{labels[i]}{labels[j]}" + "}$"
        else:
            l = None
        _plot_one_coherence(
            coh_q,
            coh_uniform,
            freq,
            label=l,
            color=f"C{idx}" if color is None else color,
            ax=ax,
            ls=ls,
        )
    ax.legend()
    return ax


def _plot_one_coherence(coh, coh_uniform, freq, label, ax, color, ls="-"):
    """
    Plot the coherence between two signals.
    """
    nqt = len(coh.shape)
    if nqt > 1:
        ax.fill_between(freq, coh[0], coh[2], alpha=0.3, lw=0, color=color)
        ax.fill_between(freq, coh_uniform[0], coh_uniform[2], alpha=0.5, lw=0, color=color)
        coh_median = coh[1]
    else:
        coh_median = coh
    ax.plot(freq, coh_median, color=color, label=label, ls=ls)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$C_{xy}$")
    ax.set_xlim([freq[0], freq[-1]])
    return ax
