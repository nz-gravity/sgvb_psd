import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf

from sgvb_psd.psd_estimator import PSDEstimator
from sgvb_psd.postproc.psd_analyzer import PSDAnalyzer
from sgvb_psd.utils.sim_varma import SimVARMA
from sgvb_psd.utils.tf_utils import set_seed


def get_var_data(npts):
    set_seed(0)
    tf.config.experimental.enable_op_determinism()

    sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
    varCoef = np.array([[[0.5, 0.0], [0.0, -0.3]], [[0.0, 0.0], [0.0, -0.5]]])
    vmaCoef = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    print(f"Generating VARMA data: {npts} samples")
    var2 = SimVARMA(
        n_samples=npts, var_coeffs=varCoef, vma_coeffs=vmaCoef, sigma=sigma
    )
    return var2


def test_var_psd_generation(plot_dir):
    var2_data = get_var_data(2**8)
    print("Starting VAR PSD generation test")
    start_time = time.time()
    optim = PSDEstimator(
        N_theta=10,
        nchunks=1,
        ntrain_map=50,
        x=var2_data.data,
        max_hyperparm_eval=1,
        fs=2 * np.pi,
        seed=0,
    )
    optim.run()
    optim.plot(
        true_psd=[var2_data.psd, var2_data.freq],
        off_symlog=False,
        xlims=[0, np.pi],
    )
    plt.savefig(f"{plot_dir}/var_psd.png")
    optim.plot_coherence(
        true_psd=[var2_data.psd, var2_data.freq],
    )
    plt.savefig(f"{plot_dir}/var_coherence.png")
    end_time = time.time()
    estimation_time = end_time - start_time
    assert estimation_time < 60


def test_chunked(plot_dir):
    var2_data = get_var_data(2**12)
    start_time = time.time()
    optim = PSDEstimator(
        N_theta=10,
        nchunks=8,
        ntrain_map=50,
        x=var2_data.data,
        max_hyperparm_eval=1,
        fs=2 * np.pi,
        seed=0,
    )
    optim.run(lr=0.003)
    optim.plot(
        true_psd=[var2_data.psd, var2_data.freq],
        off_symlog=False,
        xlims=[0, np.pi],
    )
    plt.savefig(f"{plot_dir}/var_chunked_psd.png")


def test_psd_analyser(plot_dir):
    var2_data = get_var_data(2**8)
    optim = PSDEstimator(
        N_theta=10,
        nchunks=1,
        ntrain_map=50,
        x=var2_data.data,
        max_hyperparm_eval=1,
        fs=2 * np.pi,
        seed=0,
    )
    psd_all, psd_quantiles = optim.run()
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
