import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sgvb_psd.backend import BayesianModel, AnalysisData
from sgvb_psd.postproc.psd_analyzer import PSDAnalyzer
from sgvb_psd.psd_estimator import PSDEstimator
from sgvb_psd.utils.sim_varma import SimVARMA
from sgvb_psd.utils.tf_utils import set_seed
from sgvb_psd.logging import logger
from sgvb_psd.postproc.plot_psd import plot_psdq


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
    var2_data = get_var_data(2**7)
    print("Starting VAR PSD generation test")
    start_time = time.time()
    optim = PSDEstimator(
        N_theta=5,
        nchunks=1,
        ntrain_map=5,
        x=var2_data.data,
        max_hyperparm_eval=1,
        n_elbo_maximisation_steps=5,
        fs=2 * np.pi,
        seed=0,
    )
    optim.run()

    ## Run is done, lets make some plots
    ax = optim.plot(
        true_psd=[var2_data.psd, var2_data.freq],
        off_symlog=False,
        xlims=[0, np.pi],
        quantiles='pointwise',
    )
    plot_psdq(
        optim.uniform_ci,
        freqs=optim.freq,
        axs=ax,
        color='red',
        ls='--',
    )

    plt.savefig(f"{plot_dir}/var_psd.png")
    optim.plot_coherence(
        true_psd=[var2_data.psd, var2_data.freq],
    )
    plt.savefig(f"{plot_dir}/var_coherence.png")
    optim.plot_vi_losses()
    plt.savefig(f"{plot_dir}/var_vi_losses.png")
    csv = f"{plot_dir}/var_psd.csv"
    psd_analyzer = PSDAnalyzer(
        spec_true=var2_data.psd,
        spectral_density_q=optim.pointwise_ci,
        idx=1,
        csv_file=csv,
    )
    assert isinstance(psd_analyzer.coverage_point_CI, float)
    assert isinstance(psd_analyzer.l2_error, float)
    assert os.path.exists(csv)


    end_time = time.time()
    estimation_time = end_time - start_time
    assert estimation_time < 50, f"Estimation time {estimation_time} is too long"
    logger.info(f"Test passed in {estimation_time} seconds")


def test_one_train_step_with_chunks():
    t0 = time.time()
    var2_data = get_var_data(2**7)
    analysis_data = AnalysisData(
        x=var2_data.data,
        nchunks=2,
        fmax_for_analysis=128,
        fs=2 * np.pi,
        N_theta=10,
        N_delta=10,
    )
    model = BayesianModel(analysis_data)
    opts = tf.keras.optimizers.Adam(0.01)
    model.map_train_step(opts)
    estimation_time = time.time() - t0
    assert estimation_time < 2, f"Estimation time {estimation_time} is too long"
    logger.info(f"Test passed in {estimation_time} seconds")
