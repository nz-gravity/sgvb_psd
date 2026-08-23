from typing import List

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

from .eigenbasis_analysis_data import EigenbasisAnalysisData

tfd = tfp.distributions
tfb = tfp.bijectors


class BlockBayesianModel:
    """Bayesian model for one lower-triangular row block.

    Block j contains delta_j and theta_j0, ..., theta_j,j-1. The block
    likelihood is written directly in terms of u, matching the factorised
    likelihood row by row.
    """

    def __init__(
        self,
        data: EigenbasisAnalysisData,
        block_index: int,
        degree_fluctuate: float = None,
        init_params: List[tf.Variable] = None,
        Nbw: float = 1.0,
    ):
        self.data = data
        self.block_index = int(block_index)
        self.Nbw = tf.convert_to_tensor(Nbw, dtype=tf.float32)

        if self.block_index < 0 or self.block_index >= int(data.p):
            raise ValueError(
                f"block_index must be in [0, {int(data.p) - 1}], "
                f"got {self.block_index}"
            )

        if degree_fluctuate is None:
            degree_fluctuate = data.N_delta / 2

        self.degree_fluctuate = tf.convert_to_tensor(
            degree_fluctuate, dtype=tf.float32
        )
        self.tau0 = tf.convert_to_tensor(0.01, dtype=tf.float32)
        self.c2 = tf.convert_to_tensor(4, dtype=tf.float32)
        self.sig2_alp = tf.convert_to_tensor(10, dtype=tf.float32)
        self.hyper = [self.tau0, self.c2, self.sig2_alp, self.degree_fluctuate]

        self.log_map_vals = tf.Variable(0.0)
        self.trainable_vars = self._get_trainable_vars()
        if init_params is not None:
            for i, p in enumerate(init_params):
                self.trainable_vars[i].assign(p)

    def _get_trainable_vars(self, batch_size: int = 1) -> List[tf.Variable]:
        size_delta = int(self.data.Xmat_delta.shape[1])
        size_theta = int(self.data.Xmat_theta.shape[1])
        n_theta_block = self.block_index

        zero_init = tf.initializers.zeros()
        reg_init = tf.initializers.constant(value=0.0)
        theta_init = tf.initializers.constant(value=0.0)

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
            reg_init(shape=(batch_size, 1, size_delta), dtype=tf.float32),
            name=f"ga_delta_block_{self.block_index}",
            trainable=True,
            dtype=tf.float32,
        )
        lla_delta = tf.Variable(
            zero_init(shape=(batch_size, 1, size_delta - 2), dtype=tf.float32)
            - cvec_d,
            name=f"lla_delta_block_{self.block_index}",
            trainable=True,
            dtype=tf.float32,
        )
        ltau = tf.Variable(
            zero_init(shape=(batch_size, 1, 1), dtype=tf.float32) - 1,
            name=f"ltau_block_{self.block_index}",
            trainable=True,
            dtype=tf.float32,
        )

        if n_theta_block == 0:
            return [ga_delta, lla_delta, ltau]

        ga_theta_re = tf.Variable(
            theta_init(
                shape=(batch_size, n_theta_block, size_theta), dtype=tf.float32
            ),
            name=f"ga_theta_re_block_{self.block_index}",
            trainable=True,
            dtype=tf.float32,
        )
        lla_theta_re = tf.Variable(
            zero_init(
                shape=(batch_size, n_theta_block, size_theta), dtype=tf.float32
            )
            - cvec_o,
            name=f"lla_theta_re_block_{self.block_index}",
            trainable=True,
            dtype=tf.float32,
        )
        ga_theta_im = tf.Variable(
            theta_init(
                shape=(batch_size, n_theta_block, size_theta), dtype=tf.float32
            ),
            name=f"ga_theta_im_block_{self.block_index}",
            trainable=True,
            dtype=tf.float32,
        )
        lla_theta_im = tf.Variable(
            zero_init(
                shape=(batch_size, n_theta_block, size_theta), dtype=tf.float32
            )
            - cvec_o,
            name=f"lla_theta_im_block_{self.block_index}",
            trainable=True,
            dtype=tf.float32,
        )
        ltau_theta = tf.Variable(
            zero_init(shape=(batch_size, n_theta_block, 1), dtype=tf.float32)
            - 1.5,
            name=f"ltau_theta_block_{self.block_index}",
            trainable=True,
            dtype=tf.float32,
        )

        return [
            ga_delta,
            lla_delta,
            ga_theta_re,
            lla_theta_re,
            ga_theta_im,
            lla_theta_im,
            ltau,
            ltau_theta,
        ]

    def loglik(self, params: List[tf.Variable]) -> tf.float32:
        x_gamma = tf.matmul(
            self.data.Xmat_delta, tf.transpose(params[0], [0, 2, 1])
        )[:, :, 0]
        sum_x_gamma = -tf.reduce_sum(x_gamma, axis=1) * self.data.nchunks
        exp_x_gamma_inv = tf.exp(-x_gamma)

        block = self.block_index
        u = tf.transpose(
            tf.complex(self.data.u_re, self.data.u_im), perm=[2, 0, 1]
        )
        resid = u[:, :, block][None, ...]

        if block > 0:
            theta_re = tf.matmul(
                self.data.Xmat_theta, tf.transpose(params[2], [0, 2, 1])
            )
            theta_im = tf.matmul(
                self.data.Xmat_theta, tf.transpose(params[4], [0, 2, 1])
            )
            theta = tf.complex(theta_re, theta_im)
            u_previous = u[:, :, :block]
            fitted = tf.reduce_sum(
                theta[:, None, :, :] * u_previous[None, :, :, :], axis=-1
            )
            resid = resid - fitted

        numerator = tf.reduce_sum(tf.square(tf.abs(resid)), axis=1)
        tmp = -tf.reduce_sum(numerator * exp_x_gamma_inv, axis=1)
        log_lik = tf.reduce_sum(sum_x_gamma + tmp)
        return log_lik / self.Nbw

    def logpost(self, params: List[tf.Variable]) -> tf.float32:
        return self.loglik(params) + self.logprior(params)

    def map_train_step(self, optimizer: Adam) -> tf.float32:
        with tf.GradientTape() as tape:
            self.log_map_vals = -1 * self.logpost(self.trainable_vars)

        grads = tape.gradient(self.log_map_vals, self.trainable_vars)
        grads_and_vars = [
            (g, v) for g, v in zip(grads, self.trainable_vars) if g is not None
        ]
        if grads_and_vars:
            optimizer.apply_gradients(grads_and_vars)

        self.log_map_vals *= -1
        return self.log_map_vals

    def logprior(self, params: List[tf.Variable]) -> tf.Tensor:
        Sigma1 = tf.multiply(
            tf.eye(tf.constant(2), dtype=tf.float32), self.hyper[2]
        )
        priorDist1 = tfd.MultivariateNormalTriL(
            scale_tril=tf.linalg.cholesky(Sigma1)
        )

        Sigm = tfb.Sigmoid()
        s_la_alp = Sigm(
            -tf.range(1, params[1].shape[-1] + 1.0, dtype=tf.float32)
            + self.hyper[3]
        )
        priorDist_la_alp = tfd.HalfCauchy(tf.constant(0, tf.float32), s_la_alp)

        ltau = params[2] if self.block_index == 0 else params[6]
        a2 = tf.square(tf.exp(params[1]))
        Sigma2i_diag = tf.divide(
            tf.multiply(
                tf.multiply(a2, tf.square(tf.exp(ltau))), self.hyper[1]
            ),
            tf.multiply(a2, tf.square(tf.exp(ltau))) + self.hyper[1],
        )
        priorDist2 = tfd.MultivariateNormalDiag(scale_diag=Sigma2i_diag)

        lprior_alp_delta = tf.reduce_sum(
            priorDist1.log_prob(params[0][:, :, 0:2]), [1]
        )
        lprior_delta = tf.reduce_sum(
            priorDist2.log_prob(params[0][:, :, 2:]), [1]
        )
        lpriorla_delta = tf.reduce_sum(
            priorDist_la_alp.log_prob(tf.exp(params[1])), [1, 2]
        ) + tf.reduce_sum(params[1], [1, 2])

        priorDist_tau = tfd.HalfCauchy(
            tf.constant(0, tf.float32), self.hyper[0]
        )
        log_prior = (
            lprior_delta
            + lpriorla_delta
            + lprior_alp_delta
            + tf.reduce_sum(
                priorDist_tau.log_prob(tf.exp(ltau)) + ltau, [1, 2]
            )
        )

        if self.block_index == 0:
            return log_prior

        s_la_theta = Sigm(
            -tf.range(1, params[3].shape[-1] + 1.0, dtype=tf.float32)
            + self.hyper[3]
        )
        priorDist_la_theta = tfd.HalfCauchy(
            tf.constant(0, tf.float32), s_la_theta
        )

        a3 = tf.square(tf.exp(params[3]))
        Sigma3i_diag = tf.divide(
            tf.multiply(
                tf.multiply(a3, tf.square(tf.exp(params[7]))), self.hyper[1]
            ),
            tf.multiply(a3, tf.square(tf.exp(params[7]))) + self.hyper[1],
        )
        priorDist3 = tfd.MultivariateNormalDiag(scale_diag=Sigma3i_diag)

        lprior_theta_re = tf.reduce_sum(priorDist3.log_prob(params[2]), [1])
        lpriorla_theta_re = tf.reduce_sum(
            priorDist_la_theta.log_prob(tf.exp(params[3])), [1, 2]
        ) + tf.reduce_sum(params[3], [1, 2])

        a4 = tf.square(tf.exp(params[5]))
        Sigma4i_diag = tf.divide(
            tf.multiply(
                tf.multiply(a4, tf.square(tf.exp(params[7]))), self.hyper[1]
            ),
            tf.multiply(a4, tf.square(tf.exp(params[7]))) + self.hyper[1],
        )
        priorDist4 = tfd.MultivariateNormalDiag(scale_diag=Sigma4i_diag)

        lprior_theta_im = tf.reduce_sum(priorDist4.log_prob(params[4]), [1])
        lpriorla_theta_im = tf.reduce_sum(
            priorDist_la_theta.log_prob(tf.exp(params[5])), [1, 2]
        ) + tf.reduce_sum(params[5], [1, 2])

        log_prior = (
            log_prior
            + lprior_theta_re
            + lpriorla_theta_re
            + lprior_theta_im
            + lpriorla_theta_im
            + tf.reduce_sum(
                priorDist_tau.log_prob(tf.exp(params[7])) + params[7], [1, 2]
            )
        )
        return log_prior
