import os
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
print(f"Current task ID: {task_id}")
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from sgvb_psd import SpecVI # model class
from hyperopt import hp, tpe, fmin
import time
import true_var

import numpy as np


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

lr_map_values = []
loss_values = []
all_samp = []
model_info = {}      

N_delta = 30
N_theta = 30
nchunks = 1
time_interval = 1
ntrain_map = 10_000
required_part = x.shape[0]/2
# find optimal lrs for highest elbo by Hyperopt
def objective(params):
    lr_map = params['lr_map']
    Spec = SpecVI(x)
    result_list = Spec.runModel(N_delta=N_delta, N_theta=N_theta, lr_map=lr_map, ntrain_map=ntrain_map, sparse_op=False,
                                nchunks=nchunks, time_interval=time_interval, required_part=required_part)
    
    losses = result_list[0]
    samp = result_list[2]
    
    if model_info == {}:
        model_info['Xmat_delta'], model_info['Xmat_theta'] = result_list[1].Xmtrix(N_delta=N_delta, N_theta=N_theta)
        model_info['p_dim'] = result_list[1].p_dim
    
    lr_map_values.append(lr_map)
    loss_values.append(losses[-1].numpy())
    all_samp.append(samp)
    
    
    return losses[-1].numpy()  

space = {
    'lr_map': hp.uniform('lr_map', 0.002, 0.02),
}
algo = tpe.suggest

hyperopt_start_time = time.time()

best = fmin(objective, space, algo=algo, max_evals=1)

hyperopt_end_time = time.time()
hyperopt_time = hyperopt_end_time - hyperopt_start_time

min_loss_index = loss_values.index(min(loss_values))
optimal_lr = lr_map_values[min_loss_index]
best_samp = all_samp[min_loss_index]

#find estimated psd given the max elbo-----------------------------------------------------------------
Xmat_delta = model_info['Xmat_delta']
Xmat_theta = model_info['Xmat_theta']
p_dim = model_info['p_dim']

delta2_all_s = tf.exp(tf.matmul(Xmat_delta, tf.transpose(best_samp[0], [0,2,1]))) #(500, #freq, p)

theta_re_s = tf.matmul(Xmat_theta, tf.transpose(best_samp[2], [0,2,1])) #(500, #freq, p(p-1)/2)
theta_im_s = tf.matmul(Xmat_theta, tf.transpose(best_samp[4], [0,2,1]))

theta_all_s = -(tf.complex(theta_re_s, theta_im_s)) #(500, #freq, p(p-1)/2)
theta_all_np = theta_all_s.numpy() 

D_all = tf.map_fn(lambda x: tf.linalg.diag(x), delta2_all_s).numpy() #(500, #freq, p, p)


num_slices, num_freq, num_elements = theta_all_np.shape
row_indices, col_indices = np.tril_indices(p_dim, k=-1)
diag_matrix = np.eye(p_dim, dtype=np.complex64)
T_all = np.tile(diag_matrix, (num_slices, num_freq, 1, 1))
T_all[:, :, row_indices, col_indices] = theta_all_np.reshape(num_slices, num_freq, -1)

T_all_conj_trans = np.conj(np.transpose(T_all, axes=(0, 1, 3, 2)))

D_all_inv = np.linalg.inv(D_all)

spectral_density_inverse_all = T_all_conj_trans @ D_all_inv @ T_all
spectral_density_all = np.linalg.inv(spectral_density_inverse_all) 

num_freq = spectral_density_all.shape[1]
spectral_density_q = np.zeros((3, num_freq, p_dim, p_dim), dtype=complex)

diag_indices = np.diag_indices(p_dim)
spectral_density_q[:, :, diag_indices[0], diag_indices[1]] = np.quantile(np.real(spectral_density_all[:, :, diag_indices[0], diag_indices[1]]), [0.05, 0.5, 0.95], axis=0)

triu_indices = np.triu_indices(p_dim, k=1)
real_part = (np.real(spectral_density_all[:, :, triu_indices[1], triu_indices[0]]))
imag_part = (np.imag(spectral_density_all[:, :, triu_indices[1], triu_indices[0]]))

spectral_density_q[0, :, triu_indices[1], triu_indices[0]] = (np.quantile(real_part, 0.05, axis=0) + 1j * np.quantile(imag_part, 0.05, axis=0)).T
spectral_density_q[1, :, triu_indices[1], triu_indices[0]] = (np.quantile(real_part, 0.50, axis=0) + 1j * np.quantile(imag_part, 0.50, axis=0)).T
spectral_density_q[2, :, triu_indices[1], triu_indices[0]] = (np.quantile(real_part, 0.95, axis=0) + 1j * np.quantile(imag_part, 0.95, axis=0)).T

spectral_density_q[:, :, triu_indices[0], triu_indices[1]] = np.conj(spectral_density_q[:, :, triu_indices[1], triu_indices[0]])

spectral_density_all = spectral_density_all/(np.pi/0.5)

spec_mat_median = spectral_density_q[1]/(np.pi/0.5)

n_samples, n_freq, p, _ = spectral_density_all.shape

#function transform the complex matrix to real matrix
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
    'optimal_lr_elbo': optimal_lr,
    'hyperopt_time': hyperopt_time,
    'total_iteration_time': total_iteration_time
}

result_df = pd.DataFrame([results])
csv_file = f'var2_256_errors{task_id}.csv'
result_df.to_csv(csv_file, index=False)
















