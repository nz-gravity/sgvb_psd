import matplotlib.pyplot as plt
import numpy as np
import itertools
from ..logging import logger


def matrix_combinations(p):
    indices = range(p)
    combinations = itertools.combinations(indices, 2)
    result = [((i, j), (i, i), (j, j)) for i, j in combinations]
    return result


def compute_coherence(pxx, pyy, pxy):
    coh = np.abs(np.abs(pxy) ** 2 / (pxx * pyy))
    if len(coh.shape) > 1:
        coh = np.quantile(coh, [0.025, 0.5, 0.975], axis=0)
    return coh


def plot_coherence(psd, freq, labels=None, ax=None, color=None, ls="-"):
    p = psd.shape[-1]
    combinations = matrix_combinations(p)
    if ax is None:
        _, ax = plt.subplots(1, 1)

    for idx, comb in enumerate(combinations):
        (i, j), (ii, _), (jj, _) = comb
        cohs = compute_coherence(
            psd[..., ii, ii], psd[..., jj, jj], psd[..., i, j]
        )
        if labels is not None:
            l = r"$C_{" + f"{labels[i]}{labels[j]}" + "}$"
        else:
            l = None
        _plot_one_coherence(
            cohs,
            freq,
            label=l,
            color=f"C{idx}" if color is None else color,
            ax=ax,
            ls=ls,
        )
    ax.legend()
    return ax


def _plot_one_coherence(coh, freq, label, ax, color, ls="-"):
    """
    Plot the coherence between two signals.
    """
    nqt = len(coh.shape)
    if nqt > 1:
        ax.fill_between(freq, coh[0], coh[2], alpha=0.3, lw=0, color=color)
        coh_median = coh[1]
    else:
        coh_median = coh
    ax.plot(freq, coh_median, color=color, label=label, ls=ls)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$C_{xy}$")
    ax.set_xlim([freq[0], freq[-1]])
    return ax
