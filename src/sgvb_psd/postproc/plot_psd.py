import warnings

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

warnings.filterwarnings(
    "ignore", message="Attempt to set non-positive ylim on a log-scaled axis"
)


def plot_psd(
        psdq: List[np.ndarray],
        true_psd: Optional[List[np.ndarray]] = None,
        pdgrm: Optional[List[np.ndarray]] = None,
        axes=None,
        **kwargs
) -> np.ndarray[plt.Axes]:
    """
    This is a utility function to plot the estimated PSDs.

    Parameters
    ----------
    psdq : tuple
        A tuple containing the estimated PSDs and the frequency vector.
    pdgrm : tuple, optional
        A tuple containing the periodogram and the frequency vector.
    true_psd : tuple, optional
        A tuple containing the true PSD and the frequency vector.
    **kwargs : dict
        Additional arguments to pass to the plotting functions.

    Returns
    -------
    np.ndarray[plt.Axes]
    """
    axes = plot_psdq(*psdq, **kwargs, axs=axes)
    if pdgrm:
        axes = plot_peridogram(*pdgrm, axs=axes, **kwargs)
    if true_psd:
        plot_single_psd(*true_psd, axes, **kwargs)
    format_axes(axes, **kwargs)
    return axes


def plot_psdq(psd_q, freqs, axs=None, **kwargs):
    """
    This is a utility class to plot the estimated multivariate PSDs.
    """
    nquantiles = psd_q.shape[0]
    nfreqs = psd_q.shape[1]
    p = psd_q.shape[-1]

    # setting some default values
    plt_kwargs = dict(
        color=kwargs.get("color", "C0"),
        zorder=kwargs.get("zorder", 10),
    )

    if nfreqs != len(freqs):
        raise ValueError(
            f"The len of frequencies {len(freqs)} does not match the len of PSD {nfreqs}"
        )

    if axs is None:
        # generate a square figure with pxp subplots
        fig, axs = _generate_fig(p)

    fill_kwargs = plt_kwargs.copy()
    fill_kwargs["alpha"] = 0.3
    fill_kwargs["lw"] = 0

    for row_i in range(p):
        for col_j in range(p):
            psd_ij = psd_q[..., row_i, col_j]
            psd_ij = _fmt_ij_elements(psd_ij, row_i, col_j)

            ax = axs[row_i, col_j]
            if nquantiles > 1:

                ax.fill_between(freqs, psd_ij[0], psd_ij[2], **fill_kwargs)
                ax.plot(freqs, psd_ij[1], **plt_kwargs)
            else:
                ax.plot(freqs.ravel(), psd_ij.ravel(), **plt_kwargs)

    return axs


def plot_single_psd(psd, freqs, axs=None, **kwargs):
    psd = np.array([psd])
    kwargs["color"] = kwargs.get("color", "k")
    kwargs["zorder"] = kwargs.get("zorder", -1)
    return plot_psdq(psd, freqs, axs, **kwargs)


def plot_peridogram(pdgrm, freq, axs=None, **kwargs):
    """
    This is a utility function to plot the periodogram of a time series.
    """
    p = pdgrm.shape[1]

    # setting some default values
    plt_kwargs = dict(
        color="lightgray",
        zorder=kwargs.get("zorder", -10),
        alpha=kwargs.get("alpha", 0.5),
    )

    if axs is None:
        fig, axs = _generate_fig(p)

    for row_i in range(p):
        for col_j in range(p):
            psd_ij = pdgrm[..., row_i, col_j]
            psd_ij = _fmt_ij_elements(psd_ij, row_i, col_j)

            ax = axs[row_i, col_j]
            ax.plot(freq, psd_ij, **plt_kwargs)

    if len(kwargs) > 0:
        format_axes(axs, **kwargs)

    return axs


def _fmt_ij_elements(psd, i, j):
    """
    This is a utility function to get the real and imaginary parts of the PSD.
    """
    if i > j:  # lower triangular
        return np.imag(psd)
    else:  # diag and upper triangular
        return np.real(psd)


def _generate_fig(p):
    """
    This is a utility function to generate a figure with pxp subplots.
    """
    fig, axs = plt.subplots(p, p, figsize=(p * 2.2, p * 2.2), sharex=True)
    return fig, axs


def format_axes(axes, **kwargs):
    """
    This is a utility function to format the figure.
    """
    _format_spines(axes, **kwargs)
    _format_text(axes, **kwargs)


def _format_spines(
        axes,
        tick_ln=5,
        diag_spline_thickness=2,
        xlims=None,
        diag_ylims=None,
        off_ylims=None,
        diag_log=True,
        off_symlog=True,
        sylmog_thresh=1e-49,
        **kwargs,
):
    p = axes.shape[0]
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    if diag_ylims is None:
        diag_ylims = (
            min([axes[i, i].get_ylim()[0] for i in range(p)]),
            max([axes[i, i].get_ylim()[1] for i in range(p)]),
        )

    if off_ylims is None:
        off_ylims = (
            min(
                [
                    axes[i, j].get_ylim()[0]
                    for i in range(p)
                    for j in range(p)
                    if i != j
                ]
            ),
            max(
                [
                    axes[i, j].get_ylim()[1]
                    for i in range(p)
                    for j in range(p)
                    if i != j
                ]
            ),
        )

    if xlims is None:
        xlims = axes[0, 0].get_xlim()

    # formatting for spline axes
    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            # turn off minor ticks
            ax.tick_params(
                "both",
                length=0,
                width=0,
                which="minor",
                bottom=False,
                top=False,
                left=False,
                right=False,
            )
            ax.tick_params(axis="y", direction="in", pad=-10)
            ax.yaxis.set_tick_params(which="both", labelleft=True, zorder=3)
            for label in ax.get_yticklabels():
                label.set_horizontalalignment("left")
            ax.set_xlim(xlims)

            if i == j:
                # increase ax-spine linewidth
                for spine in ax.spines.values():
                    spine.set_linewidth(diag_spline_thickness)
                    spine.set_zorder(10)
                ax.tick_params(
                    "both",
                    length=tick_ln,
                    width=diag_spline_thickness,
                    which="major",
                )
                if diag_log:
                    ax.set_yscale("log")
                ax.set_ylim(diag_ylims)
            else:
                ax.set_ylim(off_ylims)
                ax.patch.set_color("lightgray")  # or whatever color you like
                ax.patch.set_alpha(0.3)
                ax.tick_params("both", length=tick_ln, width=1, which="major")

                if off_symlog:
                    ax.set_yscale("symlog", linthresh=sylmog_thresh)


def _format_text(axes, channel_labels=None, **kwargs):
    p = axes.shape[0]
    if channel_labels is None:
        channel_labels = "".join([f"{i + 1}" for i in range(p)])
    assert len(channel_labels) == p

    add_chnl_lbls = kwargs.get("add_channel_labels", True)
    if add_chnl_lbls:
        # formatting text associated with the plot
        for i in range(p):
            for j in range(p):
                ax = axes[i, j]
                lbl = f"{channel_labels[i]}{channel_labels[j]}"
                lbl = "\mathbf{S}_{" + lbl + "}"

                if i < j:  # upper triangular
                    lbl = "$\Re(" + lbl + ")$"
                elif i > j:  # lower triangular
                    lbl = "$\Im(" + lbl + ")$"
                else:
                    lbl = "$" + lbl + "$"

                ax.text(
                    0.95,
                    0.95,
                    lbl,
                    transform=ax.transAxes,
                    horizontalalignment="right",
                    verticalalignment="top",
                    fontsize="small",
                )

    add_axes_labels = kwargs.get("add_axes_labels", True)
    if add_axes_labels:
        fig = axes[0, 0].get_figure()
        # yaxis label
        fig.text(0.075, 0.5, "PSD [1/Hz]", va="center", rotation="vertical")
        # xaxis label
        fig.text(0.5, 0.0325, "Frequency [Hz]", ha="center")
