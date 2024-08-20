import os

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
print(f"Current task ID: {task_id}")
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
from sgvb_psd import SpecVI
from sgvb_psd import lr_tuner
from hyperopt import hp, tpe, fmin
import true_vma
import time

np.random.seed(task_id)
tf.random.set_seed(task_id)

n = 256
sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
vmaCoef = np.array([[[1.0, 0.0], [0.0, 1.0]], [[-0.75, 0.5], [0.5, 0.75]]])

Simulation = true_vma.VarmaSim(n=n)
freq = np.arange(0, int(n / 2)) / n
spec_true = Simulation.calculateSpecMatrix(freq, vmaCoef, sigma)

# load data
data_whole = pd.read_csv(f"vma1_256_data.csv")
data_whole = data_whole.values

iteration_start_time = time.time()

x = data_whole[task_id * n : ((task_id + 1) * n), :]

N_delta = 30
N_theta = 30
nchunks = 1
time_interval = 1
ntrain_map = 10000
fmax_for_analysis = x.shape[0] / 2

psd_estimator = lr_tuner.OptimalSpectralDensityEstimator(
    N_delta=N_delta,
    N_theta=N_theta,
    nchunks=nchunks,
    time_interval=time_interval,
    ntrain_map=ntrain_map,
    x=x,
)
# Run the estimation
spectral_density_q, spectral_density_all = psd_estimator.run()

spec_true, spectral_density_all, spectral_density_q = [
    arr / (np.pi / 0.5)
    for arr in [spec_true, spectral_density_all, spectral_density_q]
]
n_freq = spectral_density_q.shape[1]

from sgvb_psd import psd_analyzer

PSD_analyzer = psd_analyzer.psdanalyzer(
    spec_true, spectral_density_q, n_freq, task_id, psd_estimator
)
PSD_analyzer.run_analysis(iteration_start_time)
