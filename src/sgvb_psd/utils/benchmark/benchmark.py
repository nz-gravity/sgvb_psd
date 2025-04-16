import numpy as np
from sgvb_psd.utils.sim_varma import SimVARMA
from sgvb_psd.psd_estimator import PSDEstimator
import timeit

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


def run_simulation(log2n=10):
    sim_data = SimVARMA(
        n_samples=2 ** log2n,
        **DATA_KWGS,
    )
    func = lambda: PSDEstimator(
        x=sim_data.data,
        **PSD_KWGS
    ).run(lr=OPTIM_LR)
    times = timeit.repeat(func, repeat=5, number=1)








