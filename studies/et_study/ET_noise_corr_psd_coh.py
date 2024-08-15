import h5py
import numpy as np   
from scipy.stats import median_abs_deviation
import time
start_time = time.time()
from sgvb_psd import SpecVI
import tensorflow as tf
import tensorflow_probability as tfp
from sgvb_psd.lr_tuner import lr_tuner




tfd = tfp.distributions
tfb = tfp.bijectors

file_path_x = f'X_ETnoise_GP.hdf5'
with h5py.File(file_path_x, 'r') as f:
    X_ETnoise_GP = f['E1:STRAIN'][:]

file_path_y = f'Y_ETnoise_GP.hdf5'
with h5py.File(file_path_y, 'r') as f:
    Y_ETnoise_GP = f['E2:STRAIN'][:]

file_path_z = f'Z_ETnoise_GP.hdf5'
with h5py.File(file_path_z, 'r') as f:
    Z_ETnoise_GP = f['E3:STRAIN'][:]

channels = np.column_stack((X_ETnoise_GP, Y_ETnoise_GP, Z_ETnoise_GP))
#channels_short = channels
q = 10**22/1.0

time_interval = 2000
nchunks = 125
required_part = 128

from hyperopt import hp, tpe, fmin

lr_map_values = []
loss_values = []
all_samp = []
model_info = {} 
N_delta = 450
N_theta = 450
ntrain_map = 100
max_evals = 1

def objective(params):
    lr_map = params['lr_map']
    Spec_m = SpecVI(channels * q)
    result_list = Spec_m.runModel(N_delta=N_delta, N_theta=N_theta, lr_map=lr_map, ntrain_map=ntrain_map,
                            sparse_op=False, nchunks = nchunks, time_interval = time_interval, required_part = required_part,
                            degree_fluctuate = N_delta )

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
best = fmin(objective, space, algo=algo, max_evals=max_evals)

min_loss_index = loss_values.index(min(loss_values))
optimal_lr = lr_map_values[min_loss_index]
best_samp = all_samp[min_loss_index]

print('The optimised lr is', optimal_lr)

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

spec_mat_median = spectral_density_q[1]
spec_mat_point_low = spectral_density_q[0]
spec_mat_point_upp = spectral_density_q[1]

with h5py.File('ETnoise_correlated_GP_pointwise_spec_matrices_XYZ.hdf5', 'w') as f:
    f.create_dataset('ETnoise_correlated_GP_spec_mat_median_XYZ', data=spec_mat_median)
    f.create_dataset('ETnoise_correlated_GP_spec_mat_lower_XYZ', data=spec_mat_point_low)
    f.create_dataset('ETnoise_correlated_GP_spec_mat_upper_XYZ', data=spec_mat_point_upp)

#find uniform CI given optimal lrs in phase 1 that maximises elbo for psd-------------------------------------------
def complex_to_real(matrix):
    n = matrix.shape[0]
    real_matrix = np.zeros_like(matrix, dtype=float)
    real_matrix[np.triu_indices(n)] = np.real(matrix[np.triu_indices(n)])
    real_matrix[np.tril_indices(n, -1)] = np.imag(matrix[np.tril_indices(n, -1)])
    
    return real_matrix

n_samples, n_freq, p, _ = spectral_density_all.shape

real_spectral_density_all = np.zeros_like(spectral_density_all, dtype=float)
real_spec_mat_median = np.zeros_like(spec_mat_median, dtype=float)

for i in range(n_samples):
    for j in range(n_freq):
        real_spectral_density_all[i, j] = complex_to_real(spectral_density_all[i, j])

for j in range(n_freq):
    real_spec_mat_median[j] = complex_to_real(spec_mat_median[j])


mad = median_abs_deviation(real_spectral_density_all, axis=0, nan_policy='omit')
mad[mad == 0] = 1e-10 

def uniformmax_multi(mSample):
    N_sample, N, d, _ = mSample.shape
    C_help = np.zeros((N_sample, N, d, d))

    for j in range(N):
        for r in range(d):
            for s in range(d):
                C_help[:, j, r, s] = uniformmax_help(mSample[:, j, r, s])

    return np.max(C_help, axis=0)

def uniformmax_help(sample):
    return np.abs(sample - np.median(sample)) / median_abs_deviation(sample)
    
max_std_abs_dev = uniformmax_multi(real_spectral_density_all) 
    
threshold = np.quantile(max_std_abs_dev, 0.9)    
lower_bound = real_spec_mat_median - threshold * mad
upper_bound = real_spec_mat_median + threshold * mad

with h5py.File('ETnoise_correlated_GP_uniform_spec_matrices_XYZ.hdf5', 'w') as f:
    f.create_dataset('ETnoise_correlated_GP_spec_mat_median_XYZ', data=spec_mat_median)
    f.create_dataset('ETnoise_correlated_GP_spec_mat_lower_XYZ', data=lower_bound)
    f.create_dataset('ETnoise_correlated_GP_spec_mat_upper_XYZ', data=upper_bound)

#find estimated squared coherences-----------------------------------------------------------------------
num_samples, num_freq, p_dim, _ = spectral_density_all.shape
num_pairs = p_dim * (p_dim - 1) // 2

squared_coherences = np.zeros((num_samples, num_freq, num_pairs))
for i in range(num_samples):
    for j in range(num_freq):
        matrix = spectral_density_all[i, j]
        squared_matrix = np.abs(matrix)**2
        diag = np.real(np.diag(matrix))
        squared_coherences[i, j] = squared_matrix[np.triu_indices(p_dim, k=1)] / np.outer(diag, diag)[np.triu_indices(p_dim, k=1)]

squared_coherences_stats = np.percentile(squared_coherences, [5, 50, 95], axis=0)
squared_coherences_point_low = squared_coherences_stats[0]
squared_coherences_median = squared_coherences_stats[1]
squared_coherences_point_upp = squared_coherences_stats[2]

with h5py.File('ETnoise_correlated_GP_pointwise_squared_coh_XYZ.hdf5', 'w') as f:
    f.create_dataset('ETnoise_correlated_GP_coh_median_XYZ', data=squared_coherences_median)
    f.create_dataset('ETnoise_correlated_GP_coh_lower_XYZ', data=squared_coherences_point_low)
    f.create_dataset('ETnoise_correlated_GP_coh_upper_XYZ', data=squared_coherences_point_upp)

#find uniform CI for estimated squared coherence---------------------------------------------------------
coh_all = squared_coherences
coh_med = squared_coherences_median

mad = median_abs_deviation(coh_all, axis=0, nan_policy='omit')
mad[mad == 0] = 1e-10 

def uniformmax_multi(coh_whole):
    N_sample, N, d = coh_whole.shape
    C_help = np.zeros((N_sample, N, d))

    for j in range(N):
        for r in range(d):
                C_help[:, j, r] = uniformmax_help(coh_whole[:, j, r])

    return np.max(C_help, axis=0)
    
max_std_abs_dev = uniformmax_multi(coh_all) 

threshold = np.quantile(max_std_abs_dev, 0.9)    
coh_lower = coh_med - threshold * mad
coh_upper = coh_med + threshold * mad

with h5py.File('ETnoise_correlated_GP_uniform_squared_coh_XYZ.hdf5', 'w') as f:
    f.create_dataset('ETnoise_correlated_GP_coh_median_XYZ', data=coh_med)
    f.create_dataset('ETnoise_correlated_GP_coh_lower_XYZ', data=coh_lower)
    f.create_dataset('ETnoise_correlated_GP_coh_upper_XYZ', data=coh_upper)













