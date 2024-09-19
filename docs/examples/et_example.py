import os
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sgvb_psd.psd_estimator import PSDEstimator

HERE = os.path.dirname(os.path.abspath(__file__))

def load_et_data() -> np.ndarray:
    """
    Return (n_samples, 3) array of XYZ channels
    """
    files_and_keys = [
        ("../data/X_ETnoise_GP.hdf5", 'E1:STRAIN'),
        ("../data/Y_ETnoise_GP.hdf5", 'E2:STRAIN'),
        ("../data/Z_ETnoise_GP.hdf5", 'E3:STRAIN')
    ]
    channels = [h5py.File(file, "r")[key][:] for file, key in files_and_keys]
    return np.column_stack(channels)

data = load_et_data()
original_len = len(data)
new_len = 2**19
print(f"Original data length: {original_len:,}, reducing to data length: {new_len:,}")
data = data[:new_len]

N_theta = 400
start_time = time.time()
optim = PSDEstimator(
    N_theta=N_theta,
    nchunks=128,
    fs=2048,
    ntrain_map=1000,
    x=data,
    fmax_for_analysis=128,
    degree_fluctuate=N_theta,
    seed=0,
)

optim.run(lr=0.003)
end_time = time.time()
estimation_time = end_time - start_time
print(f'The estimation time is {estimation_time:.2f}s')

optim.plot_coherence(labels='XYZ')
plt.savefig("ET_coherence.png")
plt.close()

optim.plot(xlims=[5,128], labels='XYZ')
plt.savefig("ET_psd.png")
plt.close()

optim.plot_vi_losses()
plt.savefig("ET_vi_losses.png")
plt.close()