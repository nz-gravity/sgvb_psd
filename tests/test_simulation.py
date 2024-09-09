import matplotlib.pyplot as plt
import numpy as np

from sgvb_psd.optimal_psd_estimator import OptimalPSDEstimator
from sgvb_psd.postproc.plot_psd import plot_peridogram
from sgvb_psd.postproc.psd_analyzer import PSDAnalyzer
from sgvb_psd.utils.periodogram import get_periodogram


def test_var_psd_generation(var2_data, plot_dir):
    np.random.seed(0)
    optim = OptimalPSDEstimator(
        N_theta=30,
        nchunks=1,
        duration=1,
        ntrain_map=100,
        x=var2_data.data,
        max_hyperparm_eval=1,
    )
    optim.run()
    optim.plot(true_psd=[var2_data.psd, var2_data.freq], off_symlog=False)
    plt.savefig(f"{plot_dir}/var_psd.png")

    ## TODO make siomethig like the following work
    # we need to ensure that we are getting correct range of
    # l2 error and coverage

    # run_statistics = PSDAnalyzer(optim).
    # assert run_statistics.coverage > 0.95
    # assert run_statistics.l2_error < 0.3


def test_pdgmr(var2_data, plot_dir):
    pdgrm = var2_data.periodogram
    # assert pdgrm.shape == (128, 2, 2)
    var2_data.plot()
    plt.savefig(f"{plot_dir}/pdgrm.png")
