"""Explicit guide building blocks; none silently changes the default guide."""

import jax.numpy as jnp
from jax import random
from numpyro import distributions as dist
from numpyro.distributions import constraints


def diagonal_plus_lowrank_scale(diagonal, factor):
    """TFP-compatible scale S = diag(diagonal) + factor @ factor.T.

    The diagonal contains standard-deviation scale entries. Covariance is
    S @ S.T, unlike NumPyro LowRankMultivariateNormal's covariance update.
    Leading dimensions are independent batches, as in the original guide.
    """
    eye = jnp.eye(diagonal.shape[-1], dtype=diagonal.dtype)
    return eye * diagonal[..., None, :] + factor @ factor.swapaxes(-1, -2)


class LowRankScaleNormal(dist.Distribution):
    """Gaussian evaluated through its scale, avoiding a squared condition number."""

    support = constraints.real_vector
    has_rsample = True
    pytree_data_fields = ("loc", "scale")

    def __init__(self, loc, diagonal, factor, validate_args=None):
        batch_shape = jnp.broadcast_shapes(
            loc.shape[:-1], diagonal.shape[:-1], factor.shape[:-2]
        )
        self.loc = jnp.broadcast_to(loc, batch_shape + loc.shape[-1:])
        self.scale = jnp.broadcast_to(
            diagonal_plus_lowrank_scale(diagonal, factor),
            batch_shape + (loc.shape[-1], loc.shape[-1]),
        )
        super().__init__(
            batch_shape, loc.shape[-1:], validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        noise = random.normal(
            key,
            sample_shape + self.batch_shape + self.event_shape,
            dtype=self.loc.dtype,
        )
        return self.loc + jnp.einsum("...ij,...j->...i", self.scale, noise)

    def log_prob(self, value):
        whitened = jnp.linalg.solve(self.scale, (value - self.loc)[..., None])[
            ..., 0
        ]
        return (
            -0.5
            * (
                jnp.sum(whitened**2, axis=-1)
                + self.event_shape[0] * jnp.log(2 * jnp.pi)
            )
            - jnp.linalg.slogdet(self.scale)[1]
        )

    @property
    def mean(self):
        return self.loc

    @property
    def covariance_matrix(self):
        return self.scale @ self.scale.swapaxes(-1, -2)


def legacy_lowrank_normal(loc, diagonal, factor):
    """Port the optional TFP scale-perturbation family exactly."""
    return LowRankScaleNormal(loc, diagonal, factor)
