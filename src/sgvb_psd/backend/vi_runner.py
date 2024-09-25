import timeit

import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp
from typing import List, Tuple

from ..logging import logger
from .bayesian_model import BayesianModel
from .analysis_data import AnalysisData

tfd = tfp.distributions
tfb = tfp.bijectors


class ViRunner:
    def __init__(
            self,
            x:np.ndarray,
            N_theta:int=30,
            nchunks:int=400,
            variation_factor:float=0.,
            fmax_for_analysis:float=None,
            fs:float=2048,
            degree_fluctuate:float=None,
    ):
        self.data = AnalysisData(
            x=x,
            nchunks=nchunks,
            fmax_for_analysis=fmax_for_analysis,
            fs=fs,
            N_theta=N_theta,
            N_delta=N_theta, # N_theta == N_delta in all cases
        )

        ## Define Model
        self.model = BayesianModel(
            self.data,
            degree_fluctuate=degree_fluctuate,
        )
        self.variation_factor = variation_factor
        self.surrogate_posterior:tfd.JointDistributionSequential = None

    def run(
            self,
            lr_map:float=5e-4,
            ntrain_map:int=5000,
            inference_size:int=500,
            n_elbo_maximisation_steps:int=1000,
    )->Tuple[
            np.ndarray, np.ndarray, BayesianModel, List[tf.Tensor]
        ]:

        logger.debug("Starting Model Inference Training..")

        lr = lr_map
        n_train = ntrain_map

        """
        # Phase1 obtain MAP
        """
        optimizer_hs = Adam(lr)

        start_map = timeit.default_timer()

        # train
        @tf.function
        def tune_model_to_map(model:BayesianModel, optimizer:Adam, n_train:int)->Tuple[List[tf.Variable], tf.Tensor]:
            n_samp = model.trainable_vars[0].shape[0]
            lpost = tf.constant(0.0, tf.float32, [n_samp])
            lp = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            for i in tf.range(n_train):
                lpost = model.map_train_step(optimizer)

                if optimizer.iterations % 5000 == 0:
                    tf.print(
                        "Step", optimizer.iterations, ": log posterior", lpost
                    )
                lp = lp.write(tf.cast(i, tf.int32), lpost)
            return model.trainable_vars, lp.stack()

        logger.debug(f"Start MAP search ({n_train} steps)... ")
        opt_vars_hs, self.lp = tune_model_to_map(self.model, optimizer_hs, n_train)
        self.map_time = timeit.default_timer() - start_map
        logger.debug(f"MAP Training Time: {self.map_time:.2f}s")

        """
        Phase 2 UQ
        """
        optimizer_vi = Adam(5e-2)
        self.init_surrogate_posterior(params=opt_vars_hs)

        def conditioned_log_prob(*z):
            return self.model.loglik(z) + self.model.logprior(z)

        logger.debug(f"Start ELBO maximisation ({n_elbo_maximisation_steps} steps)... ")
        start_vi = timeit.default_timer()
        # For more on TF's fit_surrogate_posterior, see
        # https://www.tensorflow.org/probability/api_docs/python/tfp/vi/fit_surrogate_posterior
        self.kdl_losses = tf.function(
            lambda l: tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=l,
                surrogate_posterior=self.surrogate_posterior,
                optimizer=optimizer_vi,
                num_steps=n_elbo_maximisation_steps,
                # jit_compile=True, ## this is running slower than without it
            )
        )(
            conditioned_log_prob
        )  #
        self.vi_time = timeit.default_timer() - start_vi
        logger.debug(f"VI Time: {self.vi_time:.2f}s")
        self.total_time = self.map_time + self.vi_time
        logger.debug(
            f"Total Inference Training Time: {self.total_time:.2f}s"
        )

        self.posteriorPointEst = self.surrogate_posterior.mean()
        self.posteriorPointEstStd = self.surrogate_posterior.stddev()
        self.variationalDistribution = self.surrogate_posterior

        # once model is trained -- we should be able to sample from it
        samp = self.surrogate_posterior.sample(inference_size)
        return self.kdl_losses, self.lp, self.model, samp

    def init_surrogate_posterior(self, params:List[tf.Variable])->None:
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
                                name="q_z_scale",
                            ),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                    for i in tf.range(len(params))
                ]
            )
        else:  # variation_factor > 0
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
                                    params[i][0].shape
                                    + self.variation_factor
                                ),
                                tfb.Identity(),
                            ),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                    for i in tf.range(len(params))
                ]
            )



