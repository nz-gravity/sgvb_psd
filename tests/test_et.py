import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np

from sgvb_psd.psd_estimator import PSDEstimator

HERE = os.path.dirname(os.path.abspath(__file__))


def load_et_data(npts=None) -> np.ndarray:
    """
    Return (n_samples, 3) array of XYZ channels
    """
    files_and_keys = [
        ("../docs/data/X_ETnoise_GP.hdf5", "E1:STRAIN"),
        ("../docs/data/Y_ETnoise_GP.hdf5", "E2:STRAIN"),
        ("../docs/data/Z_ETnoise_GP.hdf5", "E3:STRAIN"),
    ]

    channels = [
        h5py.File(os.path.join(HERE, file), "r")[key][:]
        for file, key in files_and_keys
    ]
    channels = np.column_stack(channels)
    duration = 2000
    total_npts = len(channels[0])
    dt = duration / total_npts
    t = np.arange(0, duration, dt)

    if npts is not None:
        channels = channels[0:npts]
        t = t[0:npts]
    return channels, t


def test_et(plot_dir):
    # Test takes too long -- "tests" should be a few seconds.
    data, t = load_et_data(2**14)
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

    optim.plot_coherence(labels='XYZ')
    plt.savefig(f"{plot_dir}/ET_coherence.png")


