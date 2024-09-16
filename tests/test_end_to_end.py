import os.path
import time

import matplotlib.pyplot as plt
import numpy as np

from sgvb_psd.optimal_psd_estimator import OptimalPSDEstimator
from sgvb_psd.postproc.plot_psd import plot_peridogram
from sgvb_psd.postproc.psd_analyzer import PSDAnalyzer
from sgvb_psd.utils.periodogram import get_periodogram
from sgvb_psd.utils.sim_varma import SimVARMA
import tensorflow as tf
from sgvb_psd.utils.tf_utils import set_seed

import pytest


@pytest.fixture
def var2_data()->SimVARMA:
    set_seed(0)
    tf.config.experimental.enable_op_determinism()

    sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
    varCoef = np.array([[[0.5, 0.0], [0.0, -0.3]], [[0.0, 0.0], [0.0, -0.5]]])
    vmaCoef = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    n = 1024
    var2 = SimVARMA(
        n_samples=n, var_coeffs=varCoef, vma_coeffs=vmaCoef, sigma=sigma
    )
    return var2



def test_var_psd_generation(var2_data, plot_dir):
    start_time = time.time()
    optim = OptimalPSDEstimator(
        N_theta=30,
        nchunks=1,
        duration=1,
        ntrain_map=100,
        x=var2_data.data,
        max_hyperparm_eval=1,
        seed=0,
    )
    psd_all, psd_quantiles = optim.run()
    optim.plot(
        true_psd=[var2_data.psd, var2_data.freq],
        off_symlog=False,
        xlims=[0, np.pi]
    )
    plt.savefig(f"{plot_dir}/var_psd.png")

    end_time = time.time()
    estimation_time = end_time - start_time
    assert estimation_time < 30

    csv = f"{plot_dir}/var_psd.csv"
    psd_analyzer = PSDAnalyzer(
        spec_true=var2_data.psd,
        spectral_density_q=psd_quantiles,
        task_id=1,
        csv_file=csv,
    )
    assert isinstance(psd_analyzer.coverage_point_CI, float)
    assert isinstance(psd_analyzer.l2_error, float)
    assert os.path.exists(csv)


