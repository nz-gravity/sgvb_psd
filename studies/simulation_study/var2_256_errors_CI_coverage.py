import os
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
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
import true_var
import time


np.random.seed(task_id)
tf.random.set_seed(task_id)

n = 256
sigma = np.array([[1., 0.9], [0.9, 1.]])  
varCoef = np.array([[[0.5, 0.], [0., -0.3]], [[0., 0.], [0., -0.5]]])
vmaCoef = np.array([[[1.,0.],[0.,1.]]])

Simulation = true_var.VarmaSim(n=n)
freq = np.arange(0, int(n/2)) / n
spec_true = Simulation.calculateSpecMatrix(freq, varCoef, vmaCoef, sigma)
spec_true = spec_true/(np.pi/0.5)

# load data
data_whole = pd.read_csv(f'var2_256_data.csv')
data_whole = data_whole.values


iteration_start_time = time.time()

x = data_whole[task_id*n:((task_id+1)*n),:]  

N_delta = 30
N_theta = 30
nchunks = 1
time_interval = 1
ntrain_map = 10000
required_part = x.shape[0]/2

psd_estimator = lr_tuner.OptimalSpectralDensityEstimator(N_delta=N_delta, N_theta=N_theta, nchunks=nchunks,
                                                time_interval=time_interval, ntrain_map=ntrain_map, x=x)
# Run the estimation
spectral_density_q, spectral_density_all = psd_estimator.run()

spectral_density_all = spectral_density_all/(np.pi/0.5)
spec_mat_median = spectral_density_q[1]/(np.pi/0.5)
n_samples, n_freq, p, _ = spectral_density_all.shape

#function transforms the complex matrix to real matrix
def complex_to_real(matrix):
    n = matrix.shape[0]
    real_matrix = np.zeros_like(matrix, dtype=float)
    real_matrix[np.triu_indices(n)] = np.real(matrix[np.triu_indices(n)])
    real_matrix[np.tril_indices(n, -1)] = np.imag(matrix[np.tril_indices(n, -1)])
    
    return real_matrix

#transform the elements of true psd to real values
real_spec_true = np.zeros_like(spec_true, dtype=float)
for j in range(n_freq):
    real_spec_true[j] = complex_to_real(spec_true[j])

#find L2 error given optimal lrs in phase 1 that maximises elbo-------------------------------------------
N2_VI = np.empty(n//2)
for i in range(n//2):
    N2_VI[i] = np.sum(np.diag((spec_mat_median[i,:,:]-spec_true[i,:,:]) @
                                (spec_mat_median[i,:,:]-spec_true[i,:,:])))
L2_VI = np.sqrt(np.mean(N2_VI))

##find length of CI given optimal lrs in phase 1 that maximises elbo-------------------------------------------
##pointwise CI
spec_mat_lower = spectral_density_q[0]/(np.pi/0.5)

spec_mat_upper = spectral_density_q[2]/(np.pi/0.5)

len_point_CI_f11 = (np.median(np.real(spec_mat_upper[:,0,0])) - np.median(np.real(spec_mat_lower[:,0,0])))
len_point_CI_re_f12 = (np.median(np.real(spec_mat_upper[:,0,1])) - np.median(np.real(spec_mat_lower[:,0,1])))
len_point_CI_im_f12 = (np.median(np.imag(spec_mat_upper[:,1,0])) - np.median(np.imag(spec_mat_lower[:,1,0])))
len_point_CI_f22 = (np.median(np.real(spec_mat_upper[:,1,1])) - np.median(np.real(spec_mat_lower[:,1,1])))


##find coverage given optimal lrs in phase 1 that maximises elbo-------------------------------------------
##pointwise CI
spec_mat_lower_real = np.zeros_like(spec_mat_lower, dtype=float)
for j in range(n_freq):
    spec_mat_lower_real[j] = complex_to_real(spec_mat_lower[j])

spec_mat_upper_real = np.zeros_like(spec_mat_upper, dtype=float)
for j in range(n_freq):
    spec_mat_upper_real[j] = complex_to_real(spec_mat_upper[j])

coverage_point_CI = np.mean((spec_mat_lower_real <= real_spec_true) & (real_spec_true <= spec_mat_upper_real))

iteration_end_time = time.time()
total_iteration_time = iteration_end_time - iteration_start_time


results = {
    'task_id': task_id,
    'L2_errors_VI': L2_VI,
    'len_point_CI_f11': len_point_CI_f11,
    'len_point_CI_re_f12': len_point_CI_re_f12,
    'len_point_CI_im_f12': len_point_CI_im_f12,
    'len_point_CI_f22': len_point_CI_f22,
    'coverage_pointwise': coverage_point_CI,
    'optimal_lr': psd_estimator.optimal_lr,
    'hyperopt_time': psd_estimator.hyperopt_time,
    'total_iteration_time': total_iteration_time
}

result_df = pd.DataFrame([results])
csv_file = f'var2_256_errors{task_id}.csv'
result_df.to_csv(csv_file, index=False)
















