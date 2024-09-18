import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np

from sgvb_psd.optimal_psd_estimator import OptimalPSDEstimator
from sgvb_psd.postproc import plot_peridogram

HERE = os.path.dirname(os.path.abspath(__file__))


def load_et_data(npts=None) -> np.ndarray:
    """
    Return (n_samples, 3) array of XYZ channels
    """
    files_and_keys = [
        ("../docs/data/X_ETnoise_GP.hdf5", "E1:STRAIN"),
        ("../docs/data/Y_ETnoise_GP.hdf5", "E2:STRAIN"),
        ("../docs/data/Z_ETnoise_GP.hdf5", "E3:STRAIN"),
    ]

    channels = [
        h5py.File(os.path.join(HERE, file), "r")[key][:]
        for file, key in files_and_keys
    ]
    channels = np.column_stack(channels)
    duration = 2000
    total_npts = len(channels[0])
    dt = duration / total_npts
    t = np.arange(0, duration, dt)

    if npts is not None:
        channels = channels[0:npts]
        t = t[0:npts]
    return channels, t


# Jianan -- please stop this test, it takes too long, and takes up
# too many minutes in the online pipeline (i only have limited minutes)!
def test_et(plot_dir):
    # Test takes too long -- "tests" should be a few seconds.
    data, t = load_et_data(2**14)
    dt = t[1] - t[0]

    N_theta = 10
    fmax_for_analysis = 128
    start_time = time.time()
    optim = OptimalPSDEstimator(
        N_theta=N_theta,
        nchunks=8,
        duration=t[-1],
        ntrain_map=10,
        fs=2048,
        x=data,
        max_hyperparm_eval=1,
        fmax_for_analysis=fmax_for_analysis,
        degree_fluctuate=N_theta,
        seed=0,
    )

    prev_fs = optim.x.shape[0] / 2000

    plot_peridogram(
        optim.pdgrm,
        optim.pdgrm_freq,
        channel_labels="XYZ",
        sylmog_thresh=1e-49,
        xlims=[5, 128],
    )
    plt.savefig(f"{plot_dir}/ET_peridogram.png")

    psd_all, psd_quantiles = optim.run()
    optim.plot(off_symlog=True, xlims=[5, 128])

    # plt.savefig(f"{plot_dir}/ET_psd.png")
    #
    # end_time = time.time()
    # estimation_time = end_time - start_time
    # print('The estimation time is', estimation_time, 'seconds')
    #
    # optim.plot_coherence(labels='XYZ')
    # plt.savefig(f"{plot_dir}/ET_coherence.png")
    #
    # psd_lower, psd_median, psd_upper = psd_quantiles
    #
    # channel_pairs = [(0, 1, 'green', 'lightgreen', 'X Y'),
    #                  (0, 2, 'blue', 'lightblue', 'X Z'),
    #                  (1, 2, 'red', 'lightcoral', 'Y Z')]
    #
    #
    # frequency = np.linspace(0, fmax_for_analysis, psd_median.shape[0])
    #
    #
    # fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    # plt.xlim([5, 128])
    #
    #
    # for i, j, color, fill_color, label in channel_pairs:
    #
    #     squ_coh_median = np.abs(psd_median[..., i, j]) ** 2 / (psd_median[..., i, i] * psd_median[..., j, j])
    #     squ_coh_lower = np.abs(psd_lower[..., i, j]) ** 2 / (psd_lower[..., i, i] * psd_lower[..., j, j])
    #     squ_coh_upper = np.abs(psd_upper[..., i, j]) ** 2 / (psd_upper[..., i, i] * psd_upper[..., j, j])
    #
    #
    #     plt.plot(frequency, np.squeeze(squ_coh_median), color=color, linestyle="-", label=f'coherence for {label}')
    #
    #
    #     plt.fill_between(frequency, np.squeeze(squ_coh_lower), np.squeeze(squ_coh_upper),
    #                      color=fill_color, alpha=1, label=f'90% CI for {label}')
    #
    #
    # plt.xlabel('Frequency [Hz]', fontsize=20, labelpad=10)
    # plt.ylabel('Squared Coherency', fontsize=20, labelpad=10)
    # plt.ylim([0, 0.7])
    # plt.legend(loc='upper left', fontsize='medium')
    # plt.savefig(f"{plot_dir}/ET_squared_coherence.png")
