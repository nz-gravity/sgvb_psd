import os
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np

from src.sgvb_psd.optimal_psd_estimator import OptimalPSDEstimator

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
    fmax_for_analysis = 128
    start_time = time.time()
    optim = OptimalPSDEstimator(
        N_theta=N_theta,
        nchunks=125,
        duration=2000,
        ntrain_map=1000,
        psd_scaling=psd_scaling,
        x=data * psd_scaling,
        max_hyperparm_eval=1,
        fmax_for_analysis=fmax_for_analysis,
        degree_fluctuate=N_theta,
        seed=0,
    )
    psd_all, psd_quantiles = optim.run()
    optim.plot(off_symlog=True)
    plt.savefig(f"{plot_dir}/ET_psd.png")

    end_time = time.time()
    estimation_time = end_time - start_time
    print('The estimation time is', estimation_time, 'seconds')

    psd_lower, psd_median, psd_upper = psd_quantiles

    channel_pairs = [(0, 1, 'green', 'lightgreen', 'X Y'), 
                     (0, 2, 'blue', 'lightblue', 'X Z'),   
                     (1, 2, 'red', 'lightcoral', 'Y Z')]


    frequency = np.linspace(0, fmax_for_analysis, psd_median.shape[0])


    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    plt.xlim([5, 128])


    for i, j, color, fill_color, label in channel_pairs:

        squ_coh_median = np.abs(psd_median[..., i, j]) ** 2 / (psd_median[..., i, i] * psd_median[..., j, j])
        squ_coh_lower = np.abs(psd_lower[..., i, j]) ** 2 / (psd_lower[..., i, i] * psd_lower[..., j, j])
        squ_coh_upper = np.abs(psd_upper[..., i, j]) ** 2 / (psd_upper[..., i, i] * psd_upper[..., j, j])


        plt.plot(frequency, np.squeeze(squ_coh_median), color=color, linestyle="-", label=f'coherence for {label}')
        

        plt.fill_between(frequency, np.squeeze(squ_coh_lower), np.squeeze(squ_coh_upper),
                         color=fill_color, alpha=1, label=f'90% CI for {label}')


    plt.xlabel('Frequency [Hz]', fontsize=20, labelpad=10)
    plt.ylabel('Squared Coherency', fontsize=20, labelpad=10)
    plt.ylim([0, 0.7])
    plt.legend(loc='upper left', fontsize='medium')
    plt.savefig(f"{plot_dir}/ET_squared_coherence.png")
