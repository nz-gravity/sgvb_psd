import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from sgvb_psd.postproc import format_axes, plot_psdq, plot_single_psd
from sgvb_psd.psd_estimator import PSDEstimator
from sgvb_psd.utils.sim_varma import SimVARMA


def test_all_modes(plot_dir):
    sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
    var_coeffs = np.array(
        [
            [[0.5, 0.0], [0.0, -0.3]],
            [[0.0, 0.0], [0.0, -0.5]],
        ]
    )
    vma_coeffs = np.array([[[1.0, 0.0], [0.0, 1.0]]])

    simulation = SimVARMA(
        n_samples=8192,
        var_coeffs=var_coeffs,
        vma_coeffs=vma_coeffs,
        sigma=sigma,
        seed=0,
    )

    estimator_options = dict(
        x=simulation.data,
        N_theta=30,
        nchunks=8,
        ntrain_map=1000,
        n_elbo_maximisation_steps=500,
        fs=simulation.fs,
        seed=0,
    )
    modes = [
        ("Original joint", "C0", {}),
        (
            "Eigenbasis joint",
            "C1",
            {"use_eigenbasis": True, "posterior_mode": "joint"},
        ),
        (
            "Eigenbasis factorized",
            "C2",
            {"use_eigenbasis": True, "posterior_mode": "factorized"},
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    plot_single_psd(
        simulation.psd,
        simulation.freq,
        axs=axes,
        color="black",
        ls="--",
    )

    for label, color, mode_options in modes:
        estimator = PSDEstimator(**estimator_options, **mode_options)
        estimator.run(lr=0.003)

        assert estimator.pointwise_ci.shape == (3, 511, 2, 2)
        assert np.all(np.isfinite(estimator.pointwise_ci))

        plot_psdq(
            estimator.pointwise_ci,
            estimator.freq,
            axs=axes,
            color=color,
        )

    format_axes(axes, xlims=[0, np.pi], off_symlog=False)
    axes[0, 0].legend(
        handles=[
            Line2D([0], [0], color="black", ls="--", label="True PSD"),
            Line2D([0], [0], color="C0", label="Original joint"),
            Line2D([0], [0], color="C1", label="Eigenbasis joint"),
            Line2D([0], [0], color="C2", label="Eigenbasis factorized"),
        ],
        frameon=False,
    )

    output_file = os.path.join(plot_dir, "all_modes_var2.png")
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert os.path.exists(output_file)
