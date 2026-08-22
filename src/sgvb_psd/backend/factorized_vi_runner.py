import timeit
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

from ..logging import logger
from .block_bayesian_model import BlockBayesianModel
from .compute_psd import compute_psd
from .eigenbasis_analysis_data import EigenbasisAnalysisData

tfd = tfp.distributions
tfb = tfp.bijectors


class BlockViRunner:
    def __init__(
        self,
        model: BlockBayesianModel,
        variation_factor: float = 0.0,
        surrogate_posterior: tfd.JointDistributionSequential = None,
    ):
        self.model = model
        self.variation_factor = variation_factor
        self.surrogate_posterior = surrogate_posterior

    def run(
        self,
        lr_map: float = 5e-4,
        ntrain_map: int = 5000,
        inference_size: int = 500,
        n_elbo_maximisation_steps: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray, BlockBayesianModel, List[tf.Tensor]]:
        self.run_phase1(lr_map, ntrain_map)
        self.run_phase2(n_elbo_maximisation_steps)
        samp = self.surrogate_posterior.sample(inference_size)
        return self.kdl_losses, self.lp, self.model, samp

    def run_phase1(self, lr_map: float = 5e-4, ntrain_map: int = 5000):
        optimizer_hs = Adam(lr_map)
        start_map = timeit.default_timer()
        ntrain_map = tf.constant(ntrain_map, dtype=tf.int32)
        logger.debug(
            f"Start block {self.model.block_index} Phase 1: MAP search "
        )

        @tf.function(reduce_retracing=True)
        def tune_model_to_map(
            model: BlockBayesianModel, optimizer: Adam, n_train: int
        ) -> Tuple[List[tf.Variable], tf.Tensor]:
            n_samp = model.trainable_vars[0].shape[0]
            lpost = tf.constant(0.0, tf.float32, [n_samp])
            lp = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            for i in tf.range(n_train):
                lpost = model.map_train_step(optimizer)
                if optimizer.iterations % 5000 == 0:
                    tf.print(
                        "Block",
                        model.block_index,
                        "step",
                        optimizer.iterations,
                        "/",
                        n_train,
                        ": log posterior",
                        lpost,
                    )
                lp = lp.write(tf.cast(i, tf.int32), lpost)
            return model.trainable_vars, lp.stack()

        _, self.lp = tune_model_to_map(self.model, optimizer_hs, ntrain_map)
        self.map_time = timeit.default_timer() - start_map
        logger.debug(
            f"Block {self.model.block_index} MAP Training Time: {self.map_time:.2f}s"
        )

    def run_phase2(self, n_elbo_maximisation_steps: int = 1000):
        optimizer_vi = Adam(5e-2)
        self.init_surrogate_posterior(params=self.model.trainable_vars)

        def conditioned_log_prob(*z):
            return self.model.loglik(z) + self.model.logprior(z)

        logger.debug(
            f"Start block {self.model.block_index} Phase 2: ELBO maximisation "
        )
        start_vi = timeit.default_timer()
        self.kdl_losses = tf.function(
            lambda log_prob_fn: tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=log_prob_fn,
                surrogate_posterior=self.surrogate_posterior,
                optimizer=optimizer_vi,
                num_steps=n_elbo_maximisation_steps,
            )
        )(conditioned_log_prob)
        self.vi_time = timeit.default_timer() - start_vi
        self.total_time = self.map_time + self.vi_time
        self.posteriorPointEst = self.surrogate_posterior.mean()
        self.posteriorPointEstStd = self.surrogate_posterior.stddev()
        self.variationalDistribution = self.surrogate_posterior
        logger.debug(
            f"Block {self.model.block_index} VI Time: {self.vi_time:.2f}s"
        )

    def init_surrogate_posterior(self, params: List[tf.Variable]) -> None:
        if self.variation_factor <= 0:
            self.surrogate_posterior = tfd.JointDistributionSequential(
                [
                    tfd.Independent(
                        tfd.MultivariateNormalDiag(
                            loc=params[i][0],
                            scale_diag=tfp.util.TransformedVariable(
                                tf.constant(
                                    1e-4, tf.float32, params[i][0].shape
                                ),
                                tfb.Softplus(),
                                name=f"q_z_scale_block_{self.model.block_index}_{i}",
                            ),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                    for i in range(len(params))
                ]
            )
        else:
            self.surrogate_posterior = tfd.JointDistributionSequential(
                [
                    tfd.Independent(
                        tfd.MultivariateNormalDiagPlusLowRank(
                            loc=params[i][0],
                            scale_diag=tfp.util.TransformedVariable(
                                tf.constant(
                                    1e-4, tf.float32, params[i][0].shape
                                ),
                                tfb.Softplus(),
                            ),
                            scale_perturb_factor=tfp.util.TransformedVariable(
                                tf.random_uniform_initializer()(
                                    params[i][0].shape + self.variation_factor
                                ),
                                tfb.Identity(),
                            ),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                    for i in range(len(params))
                ]
            )


class FactorizedViRunner:
    def __init__(
        self,
        x: np.ndarray,
        N_theta: int = 30,
        nchunks: int = 400,
        variation_factor: float = 0.0,
        fmax_for_analysis: float = None,
        fs: float = 2048,
        degree_fluctuate: float = None,
        init_params: List[tf.Tensor] = None,
        surrogate_posterior: tfd.JointDistributionSequential = None,
        fmin_for_analysis: float = None,
        fmin_idx_extension: int = 0,
        fmax_idx_extension: int = 32,
        Nbw: float = 1.0,
    ):
        if surrogate_posterior is not None:
            raise ValueError(
                "surrogate_posterior is not supported for factorized VI"
            )

        self.data = EigenbasisAnalysisData(
            x=x,
            nchunks=nchunks,
            fmax_for_analysis=fmax_for_analysis,
            fs=fs,
            N_theta=N_theta,
            N_delta=N_theta,
            fmin_for_analysis=fmin_for_analysis,
            fmin_idx_extension=fmin_idx_extension,
            fmax_idx_extension=fmax_idx_extension,
        )
        self.variation_factor = variation_factor
        self.degree_fluctuate = degree_fluctuate
        self.Nbw = Nbw
        self.init_params = init_params
        self.block_runners: List[BlockViRunner] = []
        self.block_samples = None
        self.model = self
        self.surrogate_posterior = None

    def run(
        self,
        lr_map: float = 5e-4,
        ntrain_map: int = 5000,
        inference_size: int = 500,
        n_elbo_maximisation_steps: int = 1000,
    ):
        self.block_runners = []
        self.block_samples = []
        self.block_lp = []
        self.block_kdl_losses = []
        self.map_time = 0.0
        self.vi_time = 0.0

        for block_index in range(int(self.data.p)):
            logger.info(f"Running factorized posterior block {block_index}")
            model = BlockBayesianModel(
                self.data,
                block_index=block_index,
                degree_fluctuate=self.degree_fluctuate,
                init_params=self._slice_init_params(block_index),
                Nbw=self.Nbw,
            )
            runner = BlockViRunner(
                model, variation_factor=self.variation_factor
            )
            kdl_losses, lp, _, samp = runner.run(
                lr_map=lr_map,
                ntrain_map=ntrain_map,
                inference_size=inference_size,
                n_elbo_maximisation_steps=n_elbo_maximisation_steps,
            )
            self.block_runners.append(runner)
            self.block_samples.append(samp)
            self.block_lp.append(lp)
            self.block_kdl_losses.append(kdl_losses)
            self.map_time += runner.map_time
            self.vi_time += runner.vi_time

        self.total_time = self.map_time + self.vi_time
        self.lp = tf.add_n(
            [tf.reshape(losses, [-1]) for losses in self.block_lp]
        )
        self.kdl_losses = tf.add_n(
            [tf.reshape(losses, [-1]) for losses in self.block_kdl_losses]
        )
        self.samps = self._stitch_block_samples(self.block_samples)
        return self.kdl_losses, self.lp, self, self.samps

    def sample_posterior(self, n_samples: int):
        if not self.block_runners:
            raise RuntimeError(
                "The factorized runner must be trained before sampling"
            )
        block_samples = [
            runner.surrogate_posterior.sample(n_samples)
            for runner in self.block_runners
        ]
        return self._stitch_block_samples(block_samples)

    def compute_psd(
        self,
        vi_samples: List[tf.Tensor],
        quantiles=[0.05, 0.5, 0.95],
        psd_scaling=1.0,
        fs=None,
    ):
        return compute_psd(
            self.data.Xmat_delta,
            self.data.Xmat_theta,
            self.data.p,
            vi_samples,
            quantiles,
            psd_scaling,
            fs,
        )

    def _slice_init_params(self, block_index: int):
        if self.init_params is None:
            return None

        delta_params = [
            self.init_params[0][:, block_index : block_index + 1, :],
            self.init_params[1][:, block_index : block_index + 1, :],
        ]
        if block_index == 0:
            return delta_params + [
                self.init_params[6][:, block_index : block_index + 1, :]
            ]

        theta_start = block_index * (block_index - 1) // 2
        theta_end = block_index * (block_index + 1) // 2
        return delta_params + [
            self.init_params[2][:, theta_start:theta_end, :],
            self.init_params[3][:, theta_start:theta_end, :],
            self.init_params[4][:, theta_start:theta_end, :],
            self.init_params[5][:, theta_start:theta_end, :],
            self.init_params[6][:, block_index : block_index + 1, :],
            self.init_params[7][:, theta_start:theta_end, :],
        ]

    def _stitch_block_samples(self, block_samples: List[List[tf.Tensor]]):
        ga_delta_parts = []
        lla_delta_parts = []
        ltau_parts = []
        ga_theta_re_parts = []
        lla_theta_re_parts = []
        ga_theta_im_parts = []
        lla_theta_im_parts = []
        ltau_theta_parts = []

        for block_index, samp in enumerate(block_samples):
            ga_delta_parts.append(samp[0])
            lla_delta_parts.append(samp[1])
            if block_index == 0:
                ltau_parts.append(samp[2])
                continue

            ga_theta_re_parts.append(samp[2])
            lla_theta_re_parts.append(samp[3])
            ga_theta_im_parts.append(samp[4])
            lla_theta_im_parts.append(samp[5])
            ltau_parts.append(samp[6])
            ltau_theta_parts.append(samp[7])

        return [
            tf.concat(ga_delta_parts, axis=1),
            tf.concat(lla_delta_parts, axis=1),
            tf.concat(ga_theta_re_parts, axis=1),
            tf.concat(lla_theta_re_parts, axis=1),
            tf.concat(ga_theta_im_parts, axis=1),
            tf.concat(lla_theta_im_parts, axis=1),
            tf.concat(ltau_parts, axis=1),
            tf.concat(ltau_theta_parts, axis=1),
        ]
