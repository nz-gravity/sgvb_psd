import timeit

import tensorflow as tf
import tensorflow_probability as tfp

from ..logging import logger
from .spec_model import SpecModel

tfd = tfp.distributions
tfb = tfp.bijectors


class SpecVI:
    def __init__(self, x):
        self.data = x

    def runModel(
        self,
        N_theta=30,
        lr_map=5e-4,
        ntrain_map=5e3,
        inference_size=500,
        nchunks=400,
        variation_factor=0,
        fmax_for_analysis=None,
        fs=2048,
        degree_fluctuate=None,
    ):

        x = self.data
        logger.debug(f"Inputted data shape: {x.shape}")

        ## Hyperparameter
        ## N_delta + N_theta are always set the same
        N_delta = N_theta
        if degree_fluctuate is None:
            degree_fluctuate = N_delta / 2
        hyper_hs = []
        tau0 = 0.01
        c2 = 4
        sig2_alp = 10
        hyper_hs.extend([tau0, c2, sig2_alp, degree_fluctuate])

        ## Define Model
        ##
        Spec_hs = SpecModel(
            x,
            hyper_hs,
            nchunks=nchunks,
            fmax_for_analysis=fmax_for_analysis,
            fs=fs,
        )
        self.model = Spec_hs  # save model object
        # comput fft
        Spec_hs.sc_fft()
        # compute array of design matrix Z, 3d

        Spec_hs.Zmtrix()

        # compute X matrix related to basis function on ffreq
        Spec_hs.Xmtrix(N_delta, N_theta)
        # convert all above to tensorflow object
        Spec_hs.toTensor()
        # create tranable variables
        Spec_hs.createModelVariables_hs()
        logger.debug(f"Model instantiated: {Spec_hs}")
        logger.debug("Starting Model Inference Training..")

        """
        # Phase1 obtain MAP
        """
        lr = lr_map
        n_train = ntrain_map  #
        optimizer_hs = tf.keras.optimizers.Adam(lr)

        start_total = timeit.default_timer()
        start_map = timeit.default_timer()

        # train
        @tf.function
        def train_hs(model, optimizer, n_train):
            # model:    model object
            # optimizer
            # n_train:  times of training
            n_samp = model.trainable_vars[0].shape[0]
            lpost = tf.constant(0.0, tf.float32, [n_samp])
            lp = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

            for i in tf.range(n_train):
                lpost = model.train_one_step(
                    optimizer, model.loglik, model.logprior_hs
                )

                if optimizer.iterations % 5000 == 0:
                    tf.print(
                        "Step", optimizer.iterations, ": log posterior", lpost
                    )
                lp = lp.write(tf.cast(i, tf.int32), lpost)
            return model.trainable_vars, lp.stack()

        logger.debug("Start Point Estimating... ")
        opt_vars_hs, lp_hs = train_hs(Spec_hs, optimizer_hs, n_train)
        # opt_vars_hs:         self.trainable_vars(ga_delta, lla_delta,
        #                                       ga_theta_re, lla_theta_re,
        #                                       ga_theta_im, lla_theta_im,
        #                                       ltau)
        # Variational inference for regression parameters
        end_map = timeit.default_timer()
        logger.debug(f"MAP Training Time: {end_map-start_map:.2f}s")
        self.lp = lp_hs

        """
        Phase 2 UQ
        """
        optimizer_vi = tf.optimizers.Adam(5e-2)
        if variation_factor <= 0:
            trainable_Mvnormal = tfd.JointDistributionSequential(
                [
                    tfd.Independent(
                        tfd.MultivariateNormalDiag(
                            loc=opt_vars_hs[i][0],
                            scale_diag=tfp.util.TransformedVariable(
                                tf.constant(
                                    1e-4, tf.float32, opt_vars_hs[i][0].shape
                                ),
                                tfb.Softplus(),
                                name="q_z_scale",
                            ),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                    for i in tf.range(len(opt_vars_hs))
                ]
            )
        else:  # variation_factor > 0
            trainable_Mvnormal = tfd.JointDistributionSequential(
                [
                    tfd.Independent(
                        tfd.MultivariateNormalDiagPlusLowRank(
                            loc=opt_vars_hs[i][0],
                            scale_diag=tfp.util.TransformedVariable(
                                tf.constant(
                                    1e-4, tf.float32, opt_vars_hs[i][0].shape
                                ),
                                tfb.Softplus(),
                            ),
                            scale_perturb_factor=tfp.util.TransformedVariable(
                                tf.random_uniform_initializer()(
                                    opt_vars_hs[i][0].shape + variation_factor
                                ),
                                tfb.Identity(),
                            ),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                    for i in tf.range(len(opt_vars_hs))
                ]
            )

        def conditioned_log_prob(*z):
            return Spec_hs.loglik(z) + Spec_hs.logprior_hs(z)

        logger.debug("Start ELBO maximisation... ")
        start = timeit.default_timer()
        # For more on TF's fit_surrogate_posterior, see
        # https://www.tensorflow.org/probability/api_docs/python/tfp/vi/fit_surrogate_posterior
        losses = tf.function(
            lambda l: tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=l,
                surrogate_posterior=trainable_Mvnormal,
                optimizer=optimizer_vi,
                num_steps=600,
            )
        )(
            conditioned_log_prob
        )  #
        stop = timeit.default_timer()
        logger.debug(f"VI Time: {stop-start:.2f}s")
        stop_total = timeit.default_timer()
        self.kld = losses
        # plt.plot(losses)

        logger.debug(
            f"Total Inference Training Time: {stop_total-start_total:.2f}s"
        )

        self.posteriorPointEst = trainable_Mvnormal.mean()
        self.posteriorPointEstStd = trainable_Mvnormal.stddev()
        self.variationalDistribution = trainable_Mvnormal

        samp = trainable_Mvnormal.sample(inference_size)
        Spec_hs.freq = Spec_hs.sc_fft()["fq_y"]
        Xmat_delta, Xmat_theta = Spec_hs.Xmtrix(
            N_delta=N_delta, N_theta=N_theta
        )
        Spec_hs.toTensor()

        return losses, Spec_hs, samp

    # TODO: _compute_psd()
    # TODO: _compute_coherence()
    # TODO: _compute_psd_quantiles()
    # TODO: _plot_basis
    # TODO: _plot_loss




