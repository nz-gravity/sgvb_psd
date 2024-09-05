import matplotlib.pyplot as plt
import numpy as np

from sgvb_psd.optimal_psd_estimator import OptimalPSDEstimator
from sgvb_psd.utils.periodogram import get_periodogram
from sgvb_psd.postproc.plot_psd import plot_peridogram
from sgvb_psd.postproc.psd_analyzer import PSDAnalyzer

def test_simulated_datasets(var2_data, plot_dir):
    plt.figure(figsize=(3, 2.5))
    plt.plot(var2_data.x)
    plt.title('Simulated VAR(2) Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend([f'Series {i + 1}' for i in range(var2_data.d)])
    plt.grid(True)
    plt.savefig(f'{plot_dir}/var2_data.png')


def test_var_psd_generation(var2_data, plot_dir):
    #FIXME: although this runs, the PSD scales are off
    optim = OptimalPSDEstimator(
        N_theta=30, nchunks=1, duration=1,
        ntrain_map=100, x=var2_data.x, max_hyperparm_eval=1
    )
    optim.run()
    optim.plot()
    plt.savefig(f'{plot_dir}/var_psd.png')

    ## TODO
    # run_statistics = PSDAnalyzer(optim).
    # assert run_statistics.coverage > 0.95
    # assert run_statistics.l2_error < 0.3



def test_pdgmr(var2_data, plot_dir):
    f, pdgrm = get_periodogram(var2_data.x, fs=2 * np.pi)
    assert pdgrm.shape == (129, 2, 2)
    plot_peridogram(var2_data.x, fs=2 * np.pi)
    plt.savefig(f'{plot_dir}/pdgrm.png')
