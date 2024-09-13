import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from sgvb_psd.optimal_psd_estimator import OptimalPSDEstimator

HERE = os.path.dirname(os.path.abspath(__file__))


def load_et_data() -> np.ndarray:
    """
    Return (n_samples, 3) array of XYZ channels
    """
    path = os.path.join(HERE, "../docs/data/et_data.h5")
    with h5py.File(path, "r") as f:
        return np.array([f["X"][:], f["Y"][:], f["Z"][:]]).T


def test_et(plot_dir):
    optim = OptimalPSDEstimator(
        N_theta=400,
        nchunks=16,
        duration=1,
        ntrain_map=100,
        x=load_et_data(),
        max_hyperparm_eval=3,
        seed=0,
    )
    optim.run()
    optim.plot()
    plt.savefig(f"{plot_dir}/var_psd.png")
