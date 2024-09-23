"""
Run a simulation study to compare the performance of the PSD estimator

(currently uses lower settings than in the paper for faster execution).
"""

from sgvb_psd.psd_estimator import PSDEstimator
from sgvb_psd.postproc.psd_analyzer import PSDAnalyzer
from sgvb_psd.utils.sim_varma import SimVARMA
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sgvb_psd.logging import logger

N_EXPERIMENTS = 10

SIM_KWGS = dict(
    sigma=np.array([[1.0, 0.9], [0.9, 1.0]]),
    var_coeffs=np.array([[[0.5, 0.0], [0.0, -0.3]], [[0.0, 0.0], [0.0, -0.5]]]),
    vma_coeffs=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
    n_samples=512,
)
VI_KWGS = dict(
    N_theta=40,
    nchunks=1,
    ntrain_map=100,
    max_hyperparm_eval=1,
    fs=2 * np.pi,
    seed=0,
)

logger.setLevel('ERROR')


def run_simulation(idx):
    sim = SimVARMA(**SIM_KWGS, seed=idx)
    optim = PSDEstimator(**VI_KWGS, x=sim.data)
    optim.run(lr=0.003)
    summary = PSDAnalyzer(
        sim.psd,
        optim.psd_quantiles,
        idx=idx,
    ).__dict__()
    summary['time'] = optim.runtimes['train']
    return {k: summary[k] for k in ['idx', 'L2_errors_VI', 'time']}


def run_all_simulations(n_experiments=N_EXPERIMENTS):
    results = []
    for idx in tqdm(range(n_experiments)):
        results.append(run_simulation(idx))
    return pd.DataFrame(results)


def plot_results(results: pd.DataFrame):
    """Ploot violin plots of L2 errors and training times."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    results.boxplot('L2_errors_VI', ax=ax[0])
    results.boxplot('time', ax=ax[1])
    ax[0].set_ylabel('L2 error')
    ax[1].set_ylabel('time')
    # add a title
    fig.suptitle(f"{N_EXPERIMENTS} simulations")
    return fig


def main():
    results = run_all_simulations()
    plot_results(results)
    plt.savefig('simulation_results.png')


if __name__ == '__main__':
    main()
