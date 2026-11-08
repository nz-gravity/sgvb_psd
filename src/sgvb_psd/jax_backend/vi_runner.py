"""MAP followed by mean-field SGVB using NumPyro's reparameterized ELBO."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

from .model import PARAMETER_NAMES, JaxAnalysisData, JaxBayesianModel


class JaxViRunner:
    def __init__(
        self,
        x,
        N_theta=30,
        nchunks=1,
        fmax_for_analysis=None,
        fs=1.0,
        degree_fluctuate=None,
        init_params=None,
        fmin_for_analysis=None,
        Nbw=1.0,
        seed=None,
        elbo_samples=1,
    ):
        if not isinstance(elbo_samples, int) or elbo_samples < 1:
            raise ValueError("elbo_samples must be a positive integer")
        self.data = JaxAnalysisData(
            x,
            nchunks,
            fmax_for_analysis,
            fs,
            N_theta,
            N_theta,
            fmin_for_analysis,
        )
        self.model = JaxBayesianModel(
            self.data,
            degree_fluctuate,
            init_params,
            Nbw,
        )
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        self.rng_key = jax.random.PRNGKey(seed)
        self.elbo_samples = elbo_samples
        self.svi_params = None

    def _numpyro_model(self):
        params = []
        for name, shape in zip(PARAMETER_NAMES, self.model.parameter_shapes):
            if np.prod(shape):
                value = numpyro.sample(
                    name, dist.ImproperUniform(constraints.real, (), shape)
                )
            else:
                value = jnp.zeros(shape, dtype=jnp.float32)
            params.append(value[None, ...])
        # The prior already includes the log-coordinate Jacobians. The
        # improper base measure avoids adding another prior or Jacobian.
        numpyro.factor("log_posterior", self.model.logpost(tuple(params))[0])

    def _guide(self):
        for name, initial in zip(PARAMETER_NAMES, self.model.trainable_vars):
            shape = initial.shape[1:]
            if np.prod(shape):
                loc = numpyro.param(name + "_loc", initial[0])
                scale = numpyro.param(
                    name + "_scale",
                    jnp.full(shape, 1e-4),
                    constraint=constraints.softplus_positive,
                )
                numpyro.sample(name, dist.Normal(loc, scale).to_event(2))

    def run_phase1(self, lr_map=5e-4, ntrain_map=5000):
        if not np.isfinite(lr_map) or lr_map <= 0:
            raise ValueError("lr_map must be finite and positive")
        if not isinstance(ntrain_map, int) or ntrain_map < 0:
            raise ValueError("ntrain_map must be a nonnegative integer")
        optimizer = Adam(lr_map)
        state = optimizer.init(self.model.trainable_vars)

        def step(state, _):
            params = optimizer.get_params(state)
            loss, grads = jax.value_and_grad(
                lambda z: -jnp.sum(self.model.logpost(z))
            )(params)
            return optimizer.update(grads, state), -loss

        start = time.perf_counter()
        state, self.lp = jax.jit(
            lambda s: jax.lax.scan(step, s, None, length=ntrain_map)
        )(state)
        jax.block_until_ready((state, self.lp))
        self.map_time = time.perf_counter() - start
        self.model.trainable_vars = optimizer.get_params(state)

    def run_phase2(self, n_elbo_maximisation_steps=1000):
        if (
            not isinstance(n_elbo_maximisation_steps, int)
            or n_elbo_maximisation_steps < 1
        ):
            raise ValueError(
                "n_elbo_maximisation_steps must be a positive integer"
            )
        self.rng_key, key = jax.random.split(self.rng_key)
        self.svi = SVI(
            self._numpyro_model,
            self._guide,
            Adam(5e-2),
            Trace_ELBO(num_particles=self.elbo_samples),
        )
        start = time.perf_counter()
        result = self.svi.run(
            key,
            n_elbo_maximisation_steps,
            progress_bar=False,
        )
        jax.block_until_ready(result)
        self.vi_time = time.perf_counter() - start
        self.total_time = self.map_time + self.vi_time
        self.kdl_losses = result.losses
        self.svi_params = result.params

    def sample_posterior(self, n_samples=1000):
        if self.svi_params is None:
            raise RuntimeError("run inference before sampling the posterior")
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        self.rng_key, key = jax.random.split(self.rng_key)
        keys = jax.random.split(key, len(PARAMETER_NAMES))
        samples = []
        for name, shape, key in zip(
            PARAMETER_NAMES,
            self.model.parameter_shapes,
            keys,
        ):
            if np.prod(shape):
                samples.append(
                    self.svi_params[name + "_loc"]
                    + self.svi_params[name + "_scale"]
                    * jax.random.normal(
                        key, (n_samples,) + shape, dtype=jnp.float32
                    )
                )
            else:
                samples.append(
                    jnp.zeros((n_samples,) + shape, dtype=jnp.float32)
                )
        return tuple(samples)

    def run(
        self,
        lr_map=5e-4,
        ntrain_map=5000,
        inference_size=500,
        n_elbo_maximisation_steps=1000,
    ):
        self.run_phase1(lr_map, ntrain_map)
        self.run_phase2(n_elbo_maximisation_steps)
        samples = self.sample_posterior(inference_size)
        return self.kdl_losses, self.lp, self.model, samples
