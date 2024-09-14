import os
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np

from sgvb_psd.optimal_psd_estimator import OptimalPSDEstimator

HERE = os.path.dirname(os.path.abspath(__file__))


def load_et_data() -> np.ndarray:
    """
    Return (n_samples, 3) array of XYZ channels
    """
    files_and_keys = [
        ("../studies/et_study/X_ETnoise_GP.hdf5", 'E1:STRAIN'),
        ("../studies/et_study/Y_ETnoise_GP.hdf5", 'E2:STRAIN'),
        ("../studies/et_study/Z_ETnoise_GP.hdf5", 'E3:STRAIN')
    ]
    
    channels = [h5py.File(os.path.join(HERE, file), "r")[key][:] for file, key in files_and_keys]
    
    return np.column_stack(channels)#[0:32768,:]

def test_et(plot_dir):
    psd_scaling = 10.0 ** 23
    data = load_et_data()
    N_theta = 400
    start_time = time.time()
    optim = OptimalPSDEstimator(
        N_theta=N_theta,
        nchunks=125,
        duration=2000,
        ntrain_map=1000,
        psd_scaling=psd_scaling,
        x=data * psd_scaling,
        max_hyperparm_eval=1,
        fmax_for_analysis=128,
        degree_fluctuate=N_theta,
        seed=0,
    )
    psd_all, psd_quantiles = optim.run()
    optim.plot(off_symlog=True)
    plt.savefig(f"{plot_dir}/ET_psd.png")

    end_time = time.time()
    estimation_time = end_time - start_time
    print('The estimation time is', estimation_time, 'seconds')
