import numpy as np
from sgvb_psd.utils.sim_varma import SimVARMA
from sgvb_psd.psd_estimator import PSDEstimator
import timeit
from typing import Tuple
from tqdm.auto import trange
from sgvb_psd.logging import logger
import matplotlib.pyplot as plt

logger.setLevel("ERROR")

np.random.seed(0)

DATA_KWGS = dict(
    var_coeffs=np.array(
        [[[0.5, 0.0], [0.0, -0.3]], [[0.0, 0.0], [0.0, -0.5]]]
    ),
    vma_coeffs=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
    sigma=np.array([[1.0, 0.9], [0.9, 1.0]]),
)

PSD_KWGS = dict(
    N_theta=50,
    nchunks=1,
    ntrain_map=1000,
    max_hyperparm_eval=1,
    fs=2 * np.pi,
)
OPTIM_LR = 0.003


def run_simulation(log2n: int = 5, nrep:int = 5) -> Tuple[float, float]:
    sim_data = SimVARMA(
        n_samples=2 ** log2n,
        **DATA_KWGS,
    )
    func = lambda: PSDEstimator(
        x=sim_data.data,
        **PSD_KWGS
    ).run(lr=OPTIM_LR)
    times = np.array([timeit.repeat(func, repeat=nrep, number=1)])
    return np.median(times), np.std(times)


def run_simulations(minLog2n: int = 3, maxLog2n: int = 4, fname='timings.txt', nrep:int=5) -> None:
    log2ns = np.arange(minLog2n, maxLog2n + 1)
    times = np.zeros((len(log2ns), 2))
    # run in reverse order to start with the largest n
    for i in trange(len(log2ns), desc="Running simulations"):
        times[i, :] = run_simulation(log2ns[i], nrep=nrep)

    np.savetxt(
        fname,
        np.array([log2ns, times[:, 0], times[:, 1]]).T,
        header="log2n median_time std_time",
        fmt="%d %.4f %.4f",
    )


def plot_timings(fname: str = 'timings.txt') -> None:
    data = np.loadtxt(fname)
    log2n, median_time, std_time = data.T
    n = 2 ** log2n

    plt_fname = fname.replace('.txt', '.png')
    fig, ax = plt.subplots()
    ax.fill_between(
        n,
        median_time - std_time,
        median_time + std_time,
        alpha=0.2,
        color="tab:blue",
    )
    ax.loglog(n, median_time, color="tab:blue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("Time (s)")
    plt.savefig(plt_fname, bbox_inches="tight")


def benchmark(minLog2n: int = 3, maxLog2n: int = 4, fname='timings.txt', nrep:int=5) -> None:
    run_simulations(minLog2n, maxLog2n, fname, nrep=nrep)
    plot_timings(fname)
