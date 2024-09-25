"""Module to compute the spectral density given the best surrogate posterior parameters and samples"""
import numpy as np
import tensorflow as tf
from typing import Tuple, List
from scipy.stats import median_abs_deviation

def compute_psd(
        Xmat_delta:tf.Tensor,
        Xmat_theta:tf.Tensor,
        p_dim:int,
        vi_samples: List[tf.Tensor],
        quantiles=[0.05, 0.5, 0.95],
        psd_scaling=1.0,
        fs=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is used to compute the spectral density given the best surrogate posterior parameters
    :param vi_samples: the surrogate posterior parameters

    Computes:
        1  psd_all: Nsamp instances of the spectral density [Nsamp, n-freq, dim, dim]
        2. pointwise_ci: the quantiles of the spectral density [n-quantiles, n-freq, dim, dim]
        3. uniform_ci: the uniform confidence interval of the spectral density [n-quantiles, n-freq, dim, dim]

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

    pointwise_ci = __get_pointwise_ci(psd_all, quantiles)
    uniform_ci = __get_uniform_ci(psd_all, pointwise_ci)

    # changing freq from [0, 1/2] to [0, samp_freq/2] (and applying scaling)
    if fs:
        true_fmax = fs / 2
        psd_all = psd_all / (true_fmax / 0.5)
        pointwise_ci = pointwise_ci / (true_fmax / 0.5)
        uniform_ci = uniform_ci / (true_fmax / 0.5)

    return (
        psd_all * psd_scaling ** 2,
        pointwise_ci * psd_scaling ** 2,
        uniform_ci * psd_scaling ** 2,
    )


def __get_pointwise_ci(psd_all, quantiles):
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
def __complex_to_real(matrix):
    n = matrix.shape[0]
    real_matrix = np.zeros_like(matrix, dtype=float)
    real_matrix[np.triu_indices(n)] = np.real(matrix[np.triu_indices(n)])
    real_matrix[np.tril_indices(n, -1)] = np.imag(matrix[np.tril_indices(n, -1)])

    return real_matrix


# Find the normalized median absolute deviation for every element among all smaples
# For all samples of each frequency, each matrix, return their maximum normalized absolute deviation
def __uniformmax_multi(mSample):
    N_sample, N, d, _ = mSample.shape
    C_help = np.zeros((N_sample, N, d, d))

    for j in range(N):
        for r in range(d):
            for s in range(d):
                C_help[:, j, r, s] = __uniformmax_help(mSample[:, j, r, s])

    return np.max(C_help, axis=0)


def __uniformmax_help(sample):
    return np.abs(sample - np.median(sample)) / median_abs_deviation(sample)


# Convert a real matrix to a complex matrix
def __real_to_complex(matrix):
    n = matrix.shape[0]
    complex_matrix = np.zeros((n, n), dtype=complex)

    complex_matrix[np.diag_indices(n)] = matrix[np.diag_indices(n)]

    complex_matrix[np.triu_indices(n, 1)] = matrix[np.triu_indices(n, 1)] - 1j * matrix[np.tril_indices(n, -1)]
    complex_matrix[np.tril_indices(n, -1)] = matrix[np.triu_indices(n, 1)] + 1j * matrix[np.tril_indices(n, -1)]

    return complex_matrix


def __get_uniform_ci(psd_all, psd_q):
    psd_median = psd_q[1]
    n_samples, n_freq, p, _ = psd_all.shape

    # transform elements of psd_all and psd_median to the real numbers
    real_psd_all = np.zeros_like(psd_all, dtype=float)
    real_psd_median = np.zeros_like(psd_median, dtype=float)

    for i in range(n_samples):
        for j in range(n_freq):
            real_psd_all[i, j] = __complex_to_real(psd_all[i, j])

    for j in range(n_freq):
        real_psd_median[j] = __complex_to_real(psd_median[j])

    # find the maximum normalized absolute deviation for real_psd_all
    max_std_abs_dev = __uniformmax_multi(real_psd_all)
    # find the threshold of the 90% as a screening criterion
    threshold = np.quantile(max_std_abs_dev, 0.9)

    # find the uniform CI for real_psd_median
    mad = median_abs_deviation(real_psd_all, axis=0, nan_policy='omit')
    mad[mad == 0] = 1e-10
    lower_bound = real_psd_median - threshold * mad
    upper_bound = real_psd_median + threshold * mad

    # Converts lower_bound and upper_bound to the complex matrix
    psd_uni_lower = np.zeros_like(lower_bound, dtype=complex)
    psd_uni_upper = np.zeros_like(upper_bound, dtype=complex)

    for i in range(n_freq):
        psd_uni_lower[i] = __real_to_complex(lower_bound[i])
        psd_uni_upper[i] = __real_to_complex(upper_bound[i])

    psd_uniform = np.stack([psd_uni_lower, psd_median, psd_uni_upper], axis=0)

    return psd_uniform
