from sgvb_psd.utils.sim_varma import SimVARMA
from sgvb_psd.utils.tf_utils import set_seed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sgvb_psd.utils.periodogram import get_welch_periodogram


def test_data_generation(plot_dir):
    sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
    varCoef = np.array([[[0.5, 0.0], [0.0, -0.3]], [[0.0, 0.0], [0.0, -0.5]]])
    vmaCoef = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    n = 4096
    set_seed(0)
    var = SimVARMA(
        n_samples=n, var_coeffs=varCoef, vma_coeffs=vmaCoef, sigma=sigma
    )

    set_seed(0)
    var2 = SimVARMA(
        n_samples=n, var_coeffs=varCoef, vma_coeffs=vmaCoef, sigma=sigma
    )

    # Gridspec
    # 4 colums and 2 rows
    # 2nd column is blank
    # arrange the plots in the following order
    # ax0     ax[0,0], ax[0,1],
    # ax1     ax[1,0], ax[1,1]

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 4)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    ax02 = fig.add_subplot(gs[0, 2])
    ax03 = fig.add_subplot(gs[0, 3])
    ax12 = fig.add_subplot(gs[1, 2])
    ax13 = fig.add_subplot(gs[1, 3])

    axes = np.array([[ax02, ax03], [ax12, ax13]])

    ax0.plot(var.data[:, 0])
    ax1.plot(var.data[:, 1])

    assert np.all(var.data == var2.data)
    var2.plot(
        axs=axes, add_axes_labels=False, xlims=[0, np.pi], welch_nchunks=8
    )

    for ax in [ax12, ax13]:
        ax.set_xlabel("Frequency [Hz]", fontsize=12)
    for ax in [ax02, ax12]:
        ax.set_ylabel("PSD [1/Hz]", fontsize=12)

    ax1.set_xlabel("Time [s]", fontsize=12)
    for ax in [ax0, ax1]:
        ax.set_ylabel("Amplitude", fontsize=12)
        ax.set_xlim([0, n])

    # draw an arrow from [ax0, ax1] to [ax02, ax03]

    # add annotationn to top right of ax0, ax1
    ax0.annotate("Channel 1", xy=(0.6, 0.9), xycoords="axes fraction")
    ax1.annotate("Channel 2", xy=(0.6, 0.9), xycoords="axes fraction")
    ax0.set_xticklabels([])

    # Draw an arrow from ax0 (bottom right corner) to ax02 (bottom left corner)
    ax0.annotate(
        "",
        xy=(0.4, 0.5),  # end point
        xytext=(0.3, 0.5),  # start point
        xycoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )

    plt.savefig(f"{plot_dir}/pdgrm.png")

    assert var._repr_html_() is not None
