import os

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
    optim = PSDEstimator(
        x=load_et_data(2**14),
        N_theta=100,
        nchunks=8,
        ntrain_map=50,
        fs=2048,
        fmax_for_analysis=128,
        seed=0,
    )
    optim.run(lr=0.001)

    kwargs = dict(
        channel_labels="XYZ",
        sylmog_thresh=1e-49,
        xlims=[5, 128],
    )

    optim.plot(**kwargs, plot_periodogram=True)
    plt.savefig(f"{plot_dir}/ET_psd.png")

    optim.plot_coherence(labels="XYZ")
    plt.savefig(f"{plot_dir}/ET_coherence.png")
