import itertools
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation


def plot_triangle_psd_coherence(
    psd_q: np.ndarray,
    freqs: np.ndarray,
    true_psd: Optional[np.ndarray] = None,
    channel_labels: Optional[List[str]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Create a triangle plot with PSDs on diagonal and coherence on off-diagonal.

    Parameters
    ----------
    psd_q : np.ndarray
        Quantiles of estimated PSDs with shape (n_quantiles, n_freqs, p, p)
    freqs : np.ndarray
        Frequency vector
    true_psd : np.ndarray, optional
        True PSD for comparison
    channel_labels : list, optional
        Labels for channels (e.g., ['H1', 'L1', 'V1'])

    Returns
    -------
    np.ndarray
        Array of axes objects
    """
    p = psd_q.shape[-1]

    # Create figure and axes
    fig, axs = plt.subplots(p, p, figsize=(p * 3, p * 2.5))
    if p == 1:
        axs = np.array([[axs]])

    # Plot diagonal elements (PSDs)
    for i in range(p):
        ax = axs[i, i]
        _plot_diagonal_psd(psd_q, freqs, i, ax, true_psd, **kwargs)

    # Plot off-diagonal elements (coherence)
    for i in range(p):
        for j in range(p):
            if i != j:
                ax = axs[i, j]
                _plot_off_diagonal_coherence(psd_q, freqs, i, j, ax, **kwargs)

    # Format the plot
    _format_triangle_plot(axs, channel_labels, **kwargs)

    return axs


def _plot_diagonal_psd(psd_q, freqs, idx, ax, true_psd=None, **kwargs):
    """Plot PSD on diagonal elements."""
    psd_ii = psd_q[..., idx, idx]

    # Default plotting parameters
    color = kwargs.get("color", "C0")

    # Plot quantiles if available
    if psd_q.shape[0] > 1:
        ax.fill_between(
            freqs, psd_ii[0], psd_ii[2], alpha=0.3, color=color, lw=0
        )
        ax.plot(freqs, psd_ii[1], color=color, lw=1.5)
    else:
        ax.plot(freqs, psd_ii.ravel(), color=color, lw=1.5)

    # Plot true PSD if provided
    if true_psd is not None:
        ax.plot(freqs, true_psd[..., idx, idx], "k--", alpha=0.7, lw=1)

    # Set log scale for PSD
    ax.set_yscale("log")
    ax.set_xscale("log")


def _plot_off_diagonal_coherence(psd_q, freqs, i, j, ax, **kwargs):
    """Plot coherence on off-diagonal elements."""
    # Compute coherence
    pxx = psd_q[..., i, i]
    pyy = psd_q[..., j, j]
    pxy = psd_q[..., i, j]

    coh_q, coh_uniform = compute_coherence(pxx, pyy, pxy)

    # Default color
    color = kwargs.get("coherence_color", "tab:blue")

    # Plot coherence
    if len(coh_q.shape) > 1:
        # Plot confidence bands
        ax.fill_between(
            freqs, coh_q[0], coh_q[2], alpha=0.3, color=color, lw=0
        )
        if coh_uniform is not None:
            ax.fill_between(
                freqs,
                coh_uniform[0],
                coh_uniform[2],
                alpha=0.5,
                color=color,
                lw=0,
            )
        # Plot median
        ax.plot(freqs, coh_q[1], color=color, lw=1.5)
    else:
        ax.plot(freqs, coh_q, color=color, lw=1.5)

    # Set coherence limits
    ax.set_ylim(0, 1)
    ax.set_xscale("log")


def _format_triangle_plot(axs, channel_labels=None, **kwargs):
    """Format the triangle plot."""
    p = axs.shape[0]

    # Default channel labels
    if channel_labels is None:
        channel_labels = [f"Ch{i + 1}" for i in range(p)]

    # Remove spines and ticks for upper triangle
    for i in range(p):
        for j in range(p):
            ax = axs[i, j]

            if i < j:  # Upper triangle - remove these plots
                ax.set_visible(False)
                continue

            # Format remaining plots
            if i == j:  # Diagonal
                # Add channel label
                label = f"${channel_labels[i]}$ PSD"
                ax.text(
                    0.05,
                    0.95,
                    label,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    fontweight="bold",
                )
                ax.set_ylabel("PSD [Hz$^{-1}$]")
            else:  # Lower triangle
                # Add coherence label
                label = f"$C_{{{channel_labels[i]}{channel_labels[j]}}}$"
                ax.text(
                    0.05,
                    0.95,
                    label,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    fontweight="bold",
                )
                ax.set_ylabel("Coherence")

            # X-axis labels only on bottom row
            if i == p - 1:
                ax.set_xlabel("Frequency [Hz]")
            # else:
            # ax.set_xticklabels([])

    # Adjust layout
    plt.tight_layout()


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
        mad = median_abs_deviation(coh_all, axis=0, nan_policy="omit")
        mad[mad == 0] = 1e-10
        max_std_abs_dev = uniformmax_multi(coh_all)
        threshold = np.quantile(max_std_abs_dev, 0.9)
        coh_lower = coh_median - threshold * mad
        coh_upper = coh_median + threshold * mad
        coh_uniform = np.stack([coh_lower, coh_median, coh_upper], axis=0)

    return coh_q, coh_uniform
