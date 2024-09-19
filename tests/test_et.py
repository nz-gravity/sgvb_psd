import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np

from sgvb_psd.psd_estimator import PSDEstimator

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(HERE, "..", "docs")
FNAME = os.path.join(DOCS_DIR, "examples", "et_data.h5")


def load_et_data(npts):
    with h5py.File(FNAME, "r") as f:
        channels = np.column_stack(
            [f["X"][:npts], f["Y"][:npts], f["Z"][:npts]]
        )
        return channels


def test_et(plot_dir):
    data = load_et_data(2**14)
    N_theta = 300
    optim = PSDEstimator(
        N_theta=N_theta,
        nchunks=8,
        ntrain_map=100,
        fs=2048,
        x=data,
        max_hyperparm_eval=1,
        fmax_for_analysis=128,
        degree_fluctuate=N_theta,
        seed=0,
        lr_range=(0.002, 0.003),
    )
    optim.run()

    kwargs = dict(
        channel_labels="XYZ",
        sylmog_thresh=1e-49,
        xlims=[5, 128],
    )

    optim.plot(**kwargs, plot_periodogram=True)
    plt.savefig(f"{plot_dir}/ET_psd.png")

    optim.plot_coherence(labels="XYZ")
    plt.savefig(f"{plot_dir}/ET_coherence.png")
