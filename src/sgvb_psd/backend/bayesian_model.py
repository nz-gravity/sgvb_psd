import timeit
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import median_abs_deviation
from ..logging import logger
from .analysis_data import AnalysisData

tfd = tfp.distributions
tfb = tfp.bijectors


class BayesianModel(AnalysisData):
    def __init__(
        self, x, hyper, nchunks, fmax_for_analysis, fs, N_theta, N_delta
    ):
        super().__init__(x, nchunks, fmax_for_analysis, fs)
        # x:      N-by-p, multivariate timeseries with N samples and p dimensions
        # hyper:  list of hyperparameters for prior
        # ts:     time series == x
        # y_ft:   fourier transformed time series
        # freq:   frequencies w/ y_ft
        # p_dim:  dimension of ts
        # Xmat:   basis matrix
        # Zar:    arry of design matrix Z_k for every freq k
        self.hyper = hyper
        self.trainable_vars = []  # all trainable variables
        self.log_map_vals = tf.Variable(0.0)
        # comput fft
        self.sc_fft()
        # compute array of design matrix Z, 3d
        self.Zmtrix()
        # compute X matrix related to basis function on ffreq
        self.Xmtrix(N_delta, N_theta)
        # convert all above to tensorflow object
        self.toTensor()
        # create tranable variables
        self.createModelVariables_hs()
        logger.debug(f"Model instantiated: {self}")

    def toTensor(self):
        # convert to tensorflow object
        self.ts = tf.convert_to_tensor(self.ts, dtype=tf.float32)
        self.y_ft = tf.convert_to_tensor(self.y_ft, dtype=tf.complex64)
        self.y_work = tf.convert_to_tensor(self.y_ft, dtype=tf.complex64)
        self.y_re = tf.math.real(self.y_work)  # not y_ft
        self.y_im = tf.math.imag(self.y_work)
        self.freq = tf.convert_to_tensor(self.freq, dtype=tf.float32)
        self.p_dim = tf.convert_to_tensor(self.p_dim, dtype=tf.int32)
        self.N_delta = tf.convert_to_tensor(self.N_delta, dtype=tf.int32)
        self.N_theta = tf.convert_to_tensor(self.N_theta, dtype=tf.int32)
        self.Xmat_delta = tf.convert_to_tensor(
            self.Xmat_delta, dtype=tf.float32
        )
        self.Xmat_theta = tf.convert_to_tensor(
            self.Xmat_theta, dtype=tf.float32
        )

        self.Zar = tf.convert_to_tensor(
            self.Zar, dtype=tf.complex64
        )  # complex array
        self.Z_re = tf.convert_to_tensor(self.Zar_re, dtype=tf.float32)
        self.Z_im = tf.convert_to_tensor(self.Zar_im, dtype=tf.float32)

        self.hyper = [
            tf.convert_to_tensor(self.hyper[i], dtype=tf.float32)
            for i in range(len(self.hyper))
        ]
        if self.p_dim > 1:
            self.n_theta = tf.cast(
                self.p_dim * (self.p_dim - 1) / 2, tf.int32
            )  # number of theta in the model

    def createModelVariables_hs(self, batch_size=1):
        #
        #
        # rule:  self.trainable_vars[0, 2, 4] must be corresponding spline regression parameters for p_dim>1
        # in 1-d case, self.trainable_vars[0] must be ga_delta parameters, no ga_theta included.

        # initial values are quite important for training
        p = int(self.y_ft.shape[2])
        size_delta = int(self.Xmat_delta.shape[1])
        size_theta = int(self.Xmat_theta.shape[1])

        # initializer = tf.initializers.GlorotUniform() # xavier initializer
        # initializer = tf.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        # initializer = tf.initializers.zeros()

        # better to have deterministic inital on reg coef to control
        ga_initializer = tf.initializers.zeros()
        ga_initializer_para = tf.initializers.constant(value=0.0)
        ga_initializer_para2 = tf.initializers.constant(value=0.0)
        if size_delta <= 10:
            cvec_d = 0.0
        else:
            cvec_d = tf.concat(
                [tf.zeros(10 - 2) + 0.0, tf.zeros(size_delta - 10) + 1.0], 0
            )
        if size_theta <= 10:
            cvec_o = 0.5
        else:
            cvec_o = tf.concat(
                [tf.zeros(10) + 0.5, tf.zeros(size_theta - 10) + 1.5], 0
            )

        ga_delta = tf.Variable(
            ga_initializer_para(
                shape=(batch_size, p, size_delta), dtype=tf.float32
            ),
            name="ga_delta",
        )
        lla_delta = tf.Variable(
            ga_initializer(
                shape=(batch_size, p, size_theta - 2), dtype=tf.float32
            )
            - cvec_d,
            name="lla_delta",
        )
        ltau = tf.Variable(
            ga_initializer(shape=(batch_size, p, 1), dtype=tf.float32) - 1,
            name="ltau",
        )
        self.trainable_vars.append(ga_delta)
        self.trainable_vars.append(lla_delta)

        nn = int(self.n_theta)  # number of thetas in the model
        ga_theta_re = tf.Variable(
            ga_initializer_para2(
                shape=(batch_size, nn, size_theta), dtype=tf.float32
            ),
            name="ga_theta_re",
        )
        ga_theta_im = tf.Variable(
            ga_initializer_para2(
                shape=(batch_size, nn, size_theta), dtype=tf.float32
            ),
            name="ga_theta_im",
        )

        lla_theta_re = tf.Variable(
            ga_initializer(
                shape=(batch_size, nn, size_theta), dtype=tf.float32
            )
            - cvec_o,
            name="lla_theta_re",
        )
        lla_theta_im = tf.Variable(
            ga_initializer(
                shape=(batch_size, nn, size_theta), dtype=tf.float32
            )
            - cvec_o,
            name="lla_theta_im",
        )

        ltau_theta = tf.Variable(
            ga_initializer(shape=(batch_size, nn, 1), dtype=tf.float32) - 1.5,
            name="ltau_theta",
        )

        self.trainable_vars.append(ga_theta_re)
        self.trainable_vars.append(lla_theta_re)
        self.trainable_vars.append(ga_theta_im)
        self.trainable_vars.append(lla_theta_im)

        self.trainable_vars.append(ltau)
        self.trainable_vars.append(ltau_theta)

        # params:          self.trainable_vars (ga_delta, lla_delta,
        #                                       ga_theta_re, lla_theta_re,
        #                                       ga_theta_im, lla_theta_im,
        #                                       ltau, ltau_theta)

    def loglik(self, params):  # log-likelihood for mvts p_dim > 1
        # y_re:            self.y_re
        # y_im:            self.y_im
        # Z_:              self.Zar
        # X_:              self.Xmat
        # params:          self.trainable_vars (ga_delta, xxx,
        #                                       ga_theta_re, xxx,
        #                                       ga_theta_im, xxx, ...)
        # each of params is a 3-d tensor with sample_size as the fist dim.
        # self.trainable_vars[:,[0, 2, 4]] must be corresponding spline regression parameters

        xγ = tf.matmul(self.Xmat_delta, tf.transpose(params[0], [0, 2, 1]))
        sum_xγ = -tf.reduce_sum(xγ, [1, 2])
        exp_xγ_inv = tf.exp(-xγ)

        xα = tf.matmul(
            self.Xmat_theta, tf.transpose(params[2], [0, 2, 1])
        )  # no need \ here
        xβ = tf.matmul(self.Xmat_theta, tf.transpose(params[4], [0, 2, 1]))

        # Z = Sum [(xα + i xβ) * y]
        Z_theta_re = tf.linalg.matvec(
            tf.expand_dims(self.Z_re, 0), xα
        ) - tf.linalg.matvec(tf.expand_dims(self.Z_im, 0), xβ)
        Z_theta_im = tf.linalg.matvec(
            tf.expand_dims(self.Z_re, 0), xβ
        ) + tf.linalg.matvec(tf.expand_dims(self.Z_im, 0), xα)

        u_re = self.y_re - Z_theta_re
        u_im = self.y_im - Z_theta_im

        numerator = tf.square(u_re) + tf.square(u_im)
        internal = tf.multiply(numerator, exp_xγ_inv)
        tmp2_ = -tf.reduce_sum(internal, [-2, -1])  # sum over p_dim and freq
        log_lik = tf.reduce_sum(sum_xγ + tmp2_)  # sum over all LnL
        return log_lik

    def logpost_hs(self, params):
        return self.loglik(params) + self.logprior_hs(params)

    #
    # Model training one step
    #
    def map_train_step(self, optimizer):  # one training step to get close to MAP
        with tf.GradientTape() as tape:
            self.log_map_vals = -1* self.logpost_hs(self.trainable_vars)
        grads = tape.gradient(self.log_map_vals, self.trainable_vars)
        optimizer.apply_gradients(zip(grads, self.trainable_vars))
        self.log_map_vals *= -1 # return POSITIVE log posterior
        return self.log_map_vals

    # For new prior strategy, need new createModelVariables() and logprior()

    def logprior_hs(self, params):
        # hyper:           list of hyperparameters (tau0, c2, sig2_alp, degree_fluctuate)
        # params:          self.trainable_vars (ga_delta, lla_delta,
        #                                       ga_theta_re, lla_theta_re,
        #                                       ga_theta_im, lla_theta_im,
        #                                       ltau, ltau_theta)
        # each of params is a 3-d tensor with sample_size as the fist dim.
        # self.trainable_vars[:,[0, 2, 4]] must be corresponding spline regression parameters
        Sigma1 = tf.multiply(
            tf.eye(tf.constant(2), dtype=tf.float32), self.hyper[2]
        )
        priorDist1 = tfd.MultivariateNormalTriL(
            scale_tril=tf.linalg.cholesky(Sigma1)
        )  # can also use tfd.MultivariateNormalDiag

        Sigm = tfb.Sigmoid()
        s_la_alp = Sigm(
            -tf.range(1, params[1].shape[-1] + 1.0, dtype=tf.float32)
            + self.hyper[3]
        )
        priorDist_la_alp = tfd.HalfCauchy(tf.constant(0, tf.float32), s_la_alp)

        s_la_theta = Sigm(
            -tf.range(1, params[3].shape[-1] + 1.0, dtype=tf.float32)
            + self.hyper[3]
        )
        priorDist_la_theta = tfd.HalfCauchy(
            tf.constant(0, tf.float32), s_la_theta
        )

        a2 = tf.square(tf.exp(params[1]))
        Sigma2i_diag = tf.divide(
            tf.multiply(
                tf.multiply(a2, tf.square(tf.exp(params[6]))), self.hyper[1]
            ),
            tf.multiply(a2, tf.square(tf.exp(params[6]))) + self.hyper[1],
        )

        priorDist2 = tfd.MultivariateNormalDiag(scale_diag=Sigma2i_diag)

        lpriorAlp_delt = tf.reduce_sum(
            priorDist1.log_prob(params[0][:, :, 0:2]), [1]
        )  #
        lprior_delt = tf.reduce_sum(
            priorDist2.log_prob(params[0][:, :, 2:]), [1]
        )  # only 2 dim due to log_prob rm the event_shape dim
        lpriorla_delt = tf.reduce_sum(
            priorDist_la_alp.log_prob(tf.exp(params[1])), [1, 2]
        ) + tf.reduce_sum(params[1], [1, 2])
        lpriorDel = lprior_delt + lpriorla_delt + lpriorAlp_delt

        a3 = tf.square(tf.exp(params[3]))
        Sigma3i_diag = tf.divide(
            tf.multiply(
                tf.multiply(a3, tf.square(tf.exp(params[7]))), self.hyper[1]
            ),
            tf.multiply(a3, tf.square(tf.exp(params[7]))) + self.hyper[1],
        )

        priorDist3 = tfd.MultivariateNormalDiag(scale_diag=Sigma3i_diag)

        lprior_thet_re = tf.reduce_sum(priorDist3.log_prob(params[2]), [1])
        lpriorla_thet_re = tf.reduce_sum(
            priorDist_la_theta.log_prob(tf.exp(params[3])), [1, 2]
        ) + tf.reduce_sum(params[3], [1, 2])
        lpriorThe_re = lprior_thet_re + lpriorla_thet_re

        a4 = tf.square(tf.exp(params[5]))
        Sigma4i_diag = tf.divide(
            tf.multiply(
                tf.multiply(a4, tf.square(tf.exp(params[7]))), self.hyper[1]
            ),
            tf.multiply(a4, tf.square(tf.exp(params[7]))) + self.hyper[1],
        )

        priorDist4 = tfd.MultivariateNormalDiag(scale_diag=Sigma4i_diag)

        lprior_thet_im = tf.reduce_sum(priorDist4.log_prob(params[4]), [1])
        lpriorla_thet_im = tf.reduce_sum(
            priorDist_la_theta.log_prob(tf.exp(params[5])), [1, 2]
        ) + tf.reduce_sum(params[5], [1, 2])
        lpriorThe_im = lprior_thet_im + lpriorla_thet_im

        priorDist_tau = tfd.HalfCauchy(
            tf.constant(0, tf.float32), self.hyper[0]
        )
        logPrior = (
            lpriorDel
            + lpriorThe_re
            + lpriorThe_im
            + tf.reduce_sum(
                priorDist_tau.log_prob(tf.exp(params[6])) + params[6], [1, 2]
            )
            + tf.reduce_sum(
                priorDist_tau.log_prob(tf.exp(params[7])) + params[7], [1, 2]
            )
        )
        return logPrior

    def compute_psd(
        self,
        vi_samples: np.ndarray,
        quantiles=[0.05, 0.5, 0.95],
        psd_scaling=1.0,
        fs=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return compute_psd(
            self.Xmat_delta,
            self.Xmat_theta,
            self.p_dim,
            vi_samples,
            quantiles,
            psd_scaling,
            fs,
        )


def compute_psd(
    Xmat_delta,
    Xmat_theta,
    p_dim,
    vi_samples: np.ndarray,
    quantiles=[0.05, 0.5, 0.95],
    psd_scaling=1.0,
    fs=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function is used to compute the spectral density given best surrogate posterior parameters
    :param vi_samples: the surrogate posterior parameters

    Computes:
        1. self.psd_q: the quantiles of the spectral density [n-quantiles, n-freq, dim, dim]
        2. self.psd_all: Nsamp instances of the spectral density [Nsamp, n-freq, dim, dim]

    """

    delta2_all_s = tf.exp(
        tf.matmul(Xmat_delta, tf.transpose(vi_samples[0], [0, 2, 1]))
    )  # (500, #freq, p)

    theta_re_s = tf.matmul(
        Xmat_theta, tf.transpose(vi_samples[2], [0, 2, 1])
    )  # (500, #freq, p(p-1)/2)
    theta_im_s = tf.matmul(Xmat_theta, tf.transpose(vi_samples[4], [0, 2, 1]))

    theta_all_s = -(
        tf.complex(theta_re_s, theta_im_s)
    )  # (500, #freq, p(p-1)/2)
    theta_all_np = theta_all_s.numpy()

    D_all = tf.map_fn(
        lambda x: tf.linalg.diag(x), delta2_all_s
    ).numpy()  # (500, #freq, p, p)

    num_slices, num_freq, num_elements = theta_all_np.shape
    row_indices, col_indices = np.tril_indices(p_dim, k=-1)
    diag_matrix = np.eye(p_dim, dtype=np.complex64)
    T_all = np.tile(diag_matrix, (num_slices, num_freq, 1, 1))
    T_all[:, :, row_indices, col_indices] = theta_all_np.reshape(
        num_slices, num_freq, -1
    )

    T_all_conj_trans = np.conj(np.transpose(T_all, axes=(0, 1, 3, 2)))
    D_all_inv = np.linalg.inv(D_all)

    spectral_density_inverse_all = T_all_conj_trans @ D_all_inv @ T_all
    psd_all = np.linalg.inv(spectral_density_inverse_all)
    psd_q = get_pointwise_ci(psd_all, quantiles)


    # changing freq from [0, 1/2] to [0, samp_freq/2] (and applying scaling)
    if fs:
        true_fmax = fs / 2
        psd_q = psd_q / (true_fmax / 0.5)
        psd_all = psd_all / (true_fmax / 0.5)

    return psd_all * psd_scaling**2, psd_q * psd_scaling**2


def get_pointwise_ci(psd_all, quantiles):
    _, num_freq, p_dim, _ = psd_all.shape
    psd_q = np.zeros((3, num_freq, p_dim, p_dim), dtype=complex)

    diag_indices = np.diag_indices(p_dim)
    psd_q[:, :, diag_indices[0], diag_indices[1]] = np.quantile(
        np.real(psd_all[:, :, diag_indices[0], diag_indices[1]]),
        quantiles,
        axis=0,
    )

    # we dont do lower triangle because it is symmetric
    upper_triangle_idx = np.triu_indices(p_dim, k=1)
    real_part = np.real(psd_all[:, :, upper_triangle_idx[1], upper_triangle_idx[0]])
    imag_part = np.imag(psd_all[:, :, upper_triangle_idx[1], upper_triangle_idx[0]])

    for i, q in enumerate(quantiles):
        psd_q[i, :, upper_triangle_idx[1], upper_triangle_idx[0]] = (
                np.quantile(real_part, q, axis=0)
                + 1j * np.quantile(imag_part, q, axis=0)
        ).T

    psd_q[:, :, upper_triangle_idx[0], upper_triangle_idx[1]] = np.conj(
        psd_q[:, :, upper_triangle_idx[1], upper_triangle_idx[0]]
    )
    return psd_q


# Convert a complex matrix to a real matrix
def complex_to_real(matrix):
    n = matrix.shape[0]
    real_matrix = np.zeros_like(matrix, dtype=float)
    real_matrix[np.triu_indices(n)] = np.real(matrix[np.triu_indices(n)])
    real_matrix[np.tril_indices(n, -1)] = np.imag(matrix[np.tril_indices(n, -1)])
    
    return real_matrix

#Find the normalized median absolute deviation for every element among all smaples
#For all samples of each frequency, each matrix, return their maximum normalized absolute deviation
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

#Convert a real matrix to a complex matrix
def real_to_complex(matrix):
    n = matrix.shape[0]
    complex_matrix = np.zeros((n, n), dtype=complex)
    
    complex_matrix[np.diag_indices(n)] = matrix[np.diag_indices(n)]
    
    complex_matrix[np.triu_indices(n, 1)] = matrix[np.triu_indices(n, 1)] + 1j * matrix[np.tril_indices(n, -1)]
    complex_matrix[np.tril_indices(n, -1)] = matrix[np.triu_indices(n, 1)] - 1j * matrix[np.tril_indices(n, -1)]
    
    return complex_matrix

def get_uniform_ci(psd_all, psd_q):
    psd_median = psd_q[1]
    n_samples, n_freq, p, _ = psd_all.shape
    
    #transform elements of psd_all and psd_median to the real numbers
    real_psd_all = np.zeros_like(psd_all, dtype=float)
    real_psd_median = np.zeros_like(psd_median, dtype=float)

    for i in range(n_samples):
        for j in range(n_freq):
            real_psd_all[i, j] = complex_to_real(psd_all[i, j])

    for j in range(n_freq):
        real_psd_median[j] = complex_to_real(psd_median[j])
        
    #find the maximum normalized absolute deviation for real_psd_all
    max_std_abs_dev = uniformmax_multi(real_psd_all) 
    #find the threshold of the 90% as a screening criterion
    threshold = np.quantile(max_std_abs_dev, 0.9)
    
    #find the uniform CI for real_psd_median
    mad = median_abs_deviation(real_psd_all, axis=0, nan_policy='omit')
    mad[mad == 0] = 1e-10
    lower_bound = real_psd_median - threshold * mad
    upper_bound = real_psd_median + threshold * mad
    
    #Converts lower_bound and upper_bound to the complex matrix
    psd_uni_lower = np.zeros_like(lower_bound, dtype=complex)
    psd_uni_upper = np.zeros_like(upper_bound, dtype=complex)

    for i in range(n_freq):
        psd_uni_lower[i] = real_to_complex(lower_bound[i])
        psd_uni_upper[i] = real_to_complex(upper_bound[i])
        
    psd_uniform = np.stack([psd_uni_lower, psd_median, psd_uni_upper], axis=0)
        
    return psd_uniform
