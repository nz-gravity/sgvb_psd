"""Independent Cholesky-row SGVB fits batched with JAX vmap."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.flatten_util import ravel_pytree
from numpyro import distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

from .model import PARAMETER_NAMES
from .vi_runner import JaxViRunner


def _map_row(target, initial, rate, steps, row, indices, mask):
    optimizer = Adam(rate)

    def update(state, _):
        value, gradient = jax.value_and_grad(
            lambda z: -target(z, row, indices, mask)
        )(optimizer.get_params(state))
        return optimizer.update(gradient, state), -value

    state, trace = jax.lax.scan(
        update, optimizer.init(initial), None, length=steps
    )
    return optimizer.get_params(state), trace


class JaxRowViRunner(JaxViRunner):
    """Same row-factorized target and guide family, independent row RNGs.

    Rows have different sizes. Padding is excluded from MAP and uses fixed
    unit-Normal auxiliary coordinates in both VI model and guide, cancelling
    exactly from the ELBO. All row fits are compiled/vmapped together; this
    does not promise a particular CPU thread count or multi-GPU distribution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        leaves = tuple(a[0] for a in self.model.trainable_vars)
        self._base, self._unravel = ravel_pytree(leaves)
        groups = [[] for _ in range(self.data.p)]
        offset = 0
        for group, leaf in enumerate(leaves):
            indices = np.arange(offset, offset + leaf.size).reshape(leaf.shape)
            for row in range(self.data.p):
                selected = (
                    indices[row]
                    if group in (0, 1, 6)
                    else indices[self.model.row == row]
                )
                groups[row].extend(selected.ravel().tolist())
            offset += leaf.size
        assert sorted(i for g in groups for i in g) == list(range(offset))
        self.row_dimensions = tuple(map(len, groups))
        width = max(self.row_dimensions)
        indices = np.zeros((self.data.p, width), int)
        mask = np.zeros_like(indices, dtype=bool)
        for row, group in enumerate(groups):
            indices[row, : len(group)] = group
            mask[row, : len(group)] = True
        self._indices, self._mask = jnp.asarray(indices), jnp.asarray(mask)
        self._rows = jnp.arange(self.data.p)
        self._row_map_fit = jax.jit(
            lambda initial, rate, steps: jax.vmap(
                lambda z, row, idx, m: _map_row(
                    self._row_target, z, rate, steps, row, idx, m
                )
            )(initial, self._rows, self._indices, self._mask),
            static_argnames=("steps",),
        )
        self._row_svi_fits = {}

    def _pack(self, params):
        flat, _ = ravel_pytree(tuple(a[0] for a in params))
        return jnp.where(self._mask, flat[self._indices], 0.0)

    def _merge(self, rows):
        return (
            jnp.zeros_like(self._base)
            .at[self._indices]
            .add(jnp.where(self._mask, rows, 0.0))
        )

    def _row_target(self, z, row, indices, mask):
        full = self._base.at[indices].add(
            jnp.where(mask, z - self._base[indices], 0.0)
        )
        params = tuple(a[None] for a in self._unravel(full))
        return self.model.logpost_rows(params)[0, row]

    def _row_model(self, row, locations, indices, mask):
        # The active base measure is flat; inactive auxiliaries have proper
        # unit-Normal model/guide densities, so padding adds no approximation.
        z = numpyro.sample(
            "row_parameters",
            dist.Normal(jnp.zeros_like(locations), 1.0)
            .mask(~mask)
            .to_event(1),
        )
        numpyro.factor(
            "row_log_posterior", self._row_target(z, row, indices, mask)
        )

    def _row_guide(self, row, locations, indices, mask):
        loc = (
            numpyro.param("loc", locations)
            if self.optimize_mean
            else locations
        )
        scale = numpyro.param(
            "scale",
            jnp.full_like(locations, 1e-4),
            constraint=constraints.softplus_positive,
        )
        numpyro.sample(
            "row_parameters",
            dist.Normal(
                jnp.where(mask, loc, 0.0), jnp.where(mask, scale, 1.0)
            ).to_event(1),
        )

    def run_phase1(self, lr_map=5e-4, ntrain_map=5000):
        if not np.isfinite(lr_map) or lr_map <= 0:
            raise ValueError("lr_map must be finite and positive")
        if not isinstance(ntrain_map, int) or ntrain_map < 0:
            raise ValueError("ntrain_map must be a nonnegative integer")
        start = time.perf_counter()
        values, self.row_lp = self._row_map_fit(
            self._pack(self.model.trainable_vars),
            jnp.asarray(lr_map, jnp.float32),
            ntrain_map,
        )
        jax.block_until_ready((values, self.row_lp))
        self.map_time = time.perf_counter() - start
        self.lp = self.row_lp.sum(0)
        self.model.trainable_vars = tuple(
            a[None] for a in self._unravel(self._merge(values))
        )

    def run_phase2(self, n_elbo_maximisation_steps=1000):
        if (
            not isinstance(n_elbo_maximisation_steps, int)
            or n_elbo_maximisation_steps < 1
        ):
            raise ValueError(
                "n_elbo_maximisation_steps must be a positive integer"
            )
        if self.svi is None:
            self.svi = SVI(
                self._row_model,
                self._row_guide,
                Adam(0.05),
                Trace_ELBO(num_particles=self.elbo_samples),
            )
        steps = n_elbo_maximisation_steps
        if steps not in self._row_svi_fits:
            self._row_svi_fits[steps] = jax.jit(
                jax.vmap(
                    lambda key, row, locations, indices, mask: self.svi.run(
                        key,
                        steps,
                        row,
                        locations,
                        indices,
                        mask,
                        progress_bar=False,
                    )
                )
            )
        self.rng_key, key = jax.random.split(self.rng_key)
        start = time.perf_counter()
        result = self._row_svi_fits[steps](
            jax.random.split(key, self.data.p),
            self._rows,
            self._pack(self.model.trainable_vars),
            self._indices,
            self._mask,
        )
        jax.block_until_ready(result)
        self.vi_time = time.perf_counter() - start
        self.total_time = self.map_time + self.vi_time
        self.row_kdl_losses = result.losses
        self.kdl_losses = result.losses.sum(0)
        scales = self._unravel(self._merge(result.params["scale"]))
        self.svi_params = {
            name + "_scale": scale
            for name, scale in zip(PARAMETER_NAMES, scales)
            if scale.size
        }
        if self.optimize_mean:
            locations = self._unravel(self._merge(result.params["loc"]))
            self.svi_params.update(
                {
                    name + "_loc": loc
                    for name, loc in zip(PARAMETER_NAMES, locations)
                    if loc.size
                }
            )
