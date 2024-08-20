import os
from sgvb_psd.logging import logger


task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
logger.info(f"Current task ID: {task_id}")
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from sgvb_psd.optimal_psd_estimator import OptimalPSDEstimator
from hyperopt import hp, tpe, fmin
from sgvb_psd.postproc import PSDAnalyzer
import true_var
import time

np.random.seed(task_id)
tf.random.set_seed(task_id)

n = 256
sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
varCoef = np.array([[[0.5, 0.0], [0.0, -0.3]], [[0.0, 0.0], [0.0, -0.5]]])
vmaCoef = np.array([[[1.0, 0.0], [0.0, 1.0]]])

Simulation = true_var.VarmaSim(n=n)
freq = np.arange(0, int(n / 2)) / n
spec_true = Simulation.calculateSpecMatrix(freq, varCoef, vmaCoef, sigma)

# load data
data_whole = pd.read_csv(f"var2_256_data.csv")
data_whole = data_whole.values

iteration_start_time = time.time()

x = data_whole[task_id * n : ((task_id + 1) * n), :]

N_delta = 30
N_theta = 30
nchunks = 1
duration = 1
ntrain_map = 10000
fmax_for_analysis = x.shape[0] / 2

psd_estimator = OptimalPSDEstimator(
    N_delta=N_delta,
    N_theta=N_theta,
    nchunks=nchunks,
    duration=duration,
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



PSD_analyzer = PSDAnalyzer(
    spec_true, spectral_density_q, n_freq, task_id, psd_estimator
)
PSD_analyzer.run_analysis(iteration_start_time)
