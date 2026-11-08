"""Module to compute the spectral density given the best surrogate posterior parameters and samples"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List
from .._psd_summary import (
    get_pointwise_ci as __get_pointwise_ci,
    get_uniform_ci as __get_uniform_ci,
)


def compute_psd(
    Xmat_delta: tf.Tensor,
    Xmat_theta: tf.Tensor,
    p_dim: int,
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
        original_fmax = 0.5
        true_fmax = fs / 2
        new_scale = true_fmax / original_fmax
        psd_all = psd_all / new_scale
        pointwise_ci = pointwise_ci / new_scale
        uniform_ci = uniform_ci / new_scale

    return (
        psd_all,
        pointwise_ci * psd_scaling**2,
        uniform_ci * psd_scaling**2,
    )
