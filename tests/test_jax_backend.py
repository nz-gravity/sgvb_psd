"""Independent NumPy/SciPy scientific controls for the JAX backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import halfcauchy, multivariate_normal, norm

from sgvb_psd.jax_backend import JaxAnalysisData, JaxBayesianModel, JaxViRunner
from sgvb_psd.jax_backend.model import PARAMETER_NAMES


def make_model(p, nchunks):
    x = np.random.default_rng(41).normal(size=(192, max(2, p)))
    x /= np.std(x)
    data = JaxAnalysisData(
        x[:, :p],
        nchunks=nchunks,
        fs=16,
        fmin_for_analysis=1,
        fmax_for_analysis=7,
        N_delta=4,
        N_theta=4,
    )
    return JaxBayesianModel(data, Nbw=1.5)


def varied_params(model, draws):
    rng = np.random.default_rng(66)
    return tuple(
        np.asarray(a).repeat(draws, axis=0)
        + rng.normal(0, 0.025, (draws,) + a.shape[1:]).astype(np.float32)
        for a in model.trainable_vars
    )


def numpy_target(model, params):
    """Direct Fourier residuals, not the JAX summed-periodogram contraction."""
    ga, local, real, lr, imag, li, tau, tt = [
        np.asarray(a, float) for a in params
    ]
    xd = np.asarray(model.data.Xmat_delta, float)
    xt = np.asarray(model.data.Xmat_theta, float)
    y = np.asarray(model.data.y_ft, complex)
    gamma = np.einsum("fk,spk->sfp", xd, ga)
    theta = np.einsum("fk,sek->sfe", xt, real + 1j * imag)
    residual = np.broadcast_to(y, (len(ga),) + y.shape).copy()
    for channel in range(1, model.data.p):
        start = channel * (channel - 1) // 2
        residual[..., channel] -= np.einsum(
            "sfi,bfi->sbf",
            theta[..., start : start + channel],
            y[..., :channel],
        )
    likelihood = (
        -model.data.nchunks * gamma.sum(axis=(1, 2))
        - np.sum(abs(residual) ** 2 * np.exp(-gamma[:, None]), axis=(1, 2, 3))
    ) / model.Nbw
    prior = norm.logpdf(ga[..., :2], scale=np.sqrt(10)).sum(axis=(1, 2))
    for coef, shrink, global_scale in (
        (ga[..., 2:], local, tau),
        (real, lr, tt),
        (imag, li, tt),
    ):
        bounded = np.exp(2 * (shrink + global_scale))
        scale = 4 * bounded / (4 + bounded)
        prior += norm.logpdf(coef, scale=scale).sum(axis=(1, 2))
        hc = expit(
            -np.arange(1, shrink.shape[-1] + 1) + model.degree_fluctuate
        )
        prior += (halfcauchy.logpdf(np.exp(shrink), scale=hc) + shrink).sum(
            axis=(1, 2)
        )
    for shrink in (tau, tt):
        prior += (halfcauchy.logpdf(np.exp(shrink), scale=0.01) + shrink).sum(
            axis=(1, 2)
        )
    return likelihood, prior


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("nchunks", [1, 3])
@pytest.mark.parametrize("draws", [1, 4])
def test_target_and_directional_gradients_against_numpy(p, nchunks, draws):
    model = make_model(p, nchunks)
    params = varied_params(model, draws)
    ll, prior = numpy_target(model, params)
    np.testing.assert_allclose(model.loglik(params), ll, rtol=4e-6, atol=2e-4)
    np.testing.assert_allclose(
        model.logprior(params), prior, rtol=4e-6, atol=2e-4
    )
    np.testing.assert_allclose(
        jax.jit(model.logpost)(params), ll + prior, rtol=4e-6, atol=4e-4
    )
    gradient = jax.grad(lambda z: jnp.sum(model.logpost(z)))(params)
    rng = np.random.default_rng(912)
    for index, leaf in enumerate(params):
        if not leaf.size:
            continue
        direction = rng.normal(size=leaf.shape)
        direction /= np.linalg.norm(direction)
        plus = [np.asarray(a, float) for a in params]
        minus = [a.copy() for a in plus]
        step = 1e-4
        plus[index] = plus[index] + step * direction
        minus[index] = minus[index] - step * direction
        lp = numpy_target(model, plus)
        lm = numpy_target(model, minus)
        finite_difference = (
            sum(a.sum() for a in lp) - sum(a.sum() for a in lm)
        ) / (2 * step)
        actual = np.sum(np.asarray(gradient[index]) * direction)
        np.testing.assert_allclose(
            actual, finite_difference, rtol=6e-4, atol=3e-3
        )


@pytest.mark.parametrize("p", [1, 2, 3])
def test_psd_matches_inverse_precision_and_unit_scaling(p):
    model = make_model(p, 3)
    params = varied_params(model, 7)
    gamma = np.einsum(
        "fk,spk->sfp", np.asarray(model.data.Xmat_delta), params[0]
    )
    theta = np.einsum(
        "fk,sek->sfe",
        np.asarray(model.data.Xmat_theta),
        params[2] + 1j * params[4],
    )
    t = np.broadcast_to(
        np.eye(p, dtype=complex), gamma.shape[:2] + (p, p)
    ).copy()
    row, col = np.tril_indices(p, -1)
    t[..., row, col] = -theta
    precision = t.conj().swapaxes(-1, -2) @ (np.exp(-gamma)[..., :, None] * t)
    expected = np.linalg.inv(precision) / 16
    actual = np.asarray(model.psd_samples(params, fs=16))
    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=2e-7)
    assert np.all(np.linalg.eigvalsh(actual) > 0)
    raw, ci, _ = model.compute_psd(params, fs=16, psd_scaling=2.5)
    np.testing.assert_allclose(raw, expected, rtol=5e-6, atol=2e-7)
    diag = np.diagonal(ci, axis1=-2, axis2=-1).real
    np.testing.assert_allclose(
        diag,
        np.quantile(
            np.diagonal(expected, axis1=-2, axis2=-1).real,
            [0.05, 0.5, 0.95],
            axis=0,
        )
        * 2.5**2,
        rtol=5e-6,
    )


@pytest.mark.parametrize("rank", [1, 2])
def test_lowrank_scale_family_against_scipy(rank):
    from sgvb_psd.jax_backend.guides import legacy_lowrank_normal

    rng = np.random.default_rng(123)
    loc = rng.normal(size=(2, 4)).astype(np.float32)
    diagonal = np.full((2, 4), 0.3, np.float32)
    factor = rng.normal(0, 0.15, (2, 4, rank)).astype(np.float32)
    scale = np.eye(4) * diagonal[..., None, :] + factor @ factor.swapaxes(
        -1, -2
    )
    covariance = scale @ scale.swapaxes(-1, -2)
    port = legacy_lowrank_normal(
        jnp.asarray(loc), jnp.asarray(diagonal), jnp.asarray(factor)
    )
    values = rng.normal(size=(5, 2, 4)).astype(np.float32)
    expected = np.column_stack(
        [
            multivariate_normal(loc[i], covariance[i]).logpdf(values[:, i])
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(port.covariance_matrix, covariance, rtol=3e-6)
    np.testing.assert_allclose(
        port.log_prob(values), expected, rtol=3e-6, atol=2e-5
    )


@pytest.mark.parametrize("p", [1, 2])
def test_default_guide_locations_stay_at_map(p):
    from numpyro.handlers import seed, substitute, trace

    runner = JaxViRunner(
        np.random.default_rng(0).normal(size=(96, p)), N_theta=3
    )
    runner.run(ntrain_map=1, n_elbo_maximisation_steps=3, inference_size=3)
    assert all(n.endswith("_scale") for n in runner.svi_params)
    sites = trace(
        seed(substitute(runner._guide, data=runner.svi_params), rng_seed=0)
    ).get_trace()
    for name, a in zip(PARAMETER_NAMES, runner.model.trainable_vars):
        if a.size:
            np.testing.assert_array_equal(
                sites[name]["fn"].base_dist.loc, a[0]
            )
