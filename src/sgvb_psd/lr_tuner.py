from sgvb_psd import SpecVI
from hyperopt import hp, tpe, fmin
import numpy as np
import tensorflow as tf
import time

class OptimalSpectralDensityEstimator:
    def __init__(self, N_delta=30, N_theta=30, nchunks=1, time_interval=1, ntrain_map=10000, x=None):
        self.N_delta = N_delta
        self.N_theta = N_theta
        self.nchunks = nchunks
        self.time_interval = time_interval
        self.ntrain_map = ntrain_map
        self.required_part = x.shape[0] / 2
        self.x = x

        # Internal variables
        self.lr_map_values = []
        self.loss_values = []
        self.all_samp = []
        self.model_info = {}

    def objective(self, params):
        lr_map = params['lr_map']
        Spec = SpecVI(self.x)
        result_list = Spec.runModel(N_delta=self.N_delta, N_theta=self.N_theta, lr_map=lr_map,
                                    ntrain_map=self.ntrain_map, nchunks=self.nchunks, 
                                    time_interval=self.time_interval, required_part=self.required_part)
        
        losses = result_list[0]
        samp = result_list[2]
        
        if self.model_info == {}:
            self.model_info['Xmat_delta'], self.model_info['Xmat_theta'] = result_list[1].Xmtrix(N_delta=self.N_delta, N_theta=self.N_theta)
            self.model_info['p_dim'] = result_list[1].p_dim
        
        self.lr_map_values.append(lr_map)
        self.loss_values.append(losses[-1].numpy())
        self.all_samp.append(samp)
        
        return losses[-1].numpy()  

    def find_optimal_lr(self):
        space = {'lr_map': hp.uniform('lr_map', 0.002, 0.02)}
        algo = tpe.suggest

        hyperopt_start_time = time.time()
        best = fmin(self.objective, space, algo=algo, max_evals=10)
        hyperopt_end_time = time.time()
        self.hyperopt_time = hyperopt_end_time - hyperopt_start_time

        min_loss_index = self.loss_values.index(min(self.loss_values))
        self.optimal_lr = self.lr_map_values[min_loss_index]
        best_samp = self.all_samp[min_loss_index]

        return best_samp

    def compute_spectral_density(self, best_samp):
        Xmat_delta = self.model_info['Xmat_delta']
        Xmat_theta = self.model_info['Xmat_theta']
        p_dim = self.model_info['p_dim']

        delta2_all_s = tf.exp(tf.matmul(Xmat_delta, tf.transpose(best_samp[0], [0, 2, 1])))  # (500, #freq, p)

        theta_re_s = tf.matmul(Xmat_theta, tf.transpose(best_samp[2], [0, 2, 1]))  # (500, #freq, p(p-1)/2)
        theta_im_s = tf.matmul(Xmat_theta, tf.transpose(best_samp[4], [0, 2, 1]))

        theta_all_s = -(tf.complex(theta_re_s, theta_im_s))  # (500, #freq, p(p-1)/2)
        theta_all_np = theta_all_s.numpy() 

        D_all = tf.map_fn(lambda x: tf.linalg.diag(x), delta2_all_s).numpy()  # (500, #freq, p, p)

        num_slices, num_freq, num_elements = theta_all_np.shape
        row_indices, col_indices = np.tril_indices(p_dim, k=-1)
        diag_matrix = np.eye(p_dim, dtype=np.complex64)
        T_all = np.tile(diag_matrix, (num_slices, num_freq, 1, 1))
        T_all[:, :, row_indices, col_indices] = theta_all_np.reshape(num_slices, num_freq, -1)

        T_all_conj_trans = np.conj(np.transpose(T_all, axes=(0, 1, 3, 2)))
        D_all_inv = np.linalg.inv(D_all)

        spectral_density_inverse_all = T_all_conj_trans @ D_all_inv @ T_all
        spectral_density_all = np.linalg.inv(spectral_density_inverse_all) 

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

        return spectral_density_q, spectral_density_all

    def run(self):
        best_samp = self.find_optimal_lr()
        return self.compute_spectral_density(best_samp)





