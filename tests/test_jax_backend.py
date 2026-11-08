"""Deterministic backend parity plus bounded inference controls."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip(
    "tensorflow", reason="isolated legacy parity reference"
)

jax = pytest.importorskip("jax")
pytest.importorskip("numpyro")
import jax.numpy as jnp

from sgvb_psd._spectral import compute_chunked_fft
from sgvb_psd.backend import AnalysisData, BayesianModel
from sgvb_psd.jax_backend import JaxAnalysisData, JaxBayesianModel, JaxViRunner
from sgvb_psd.psd_estimator import PSDEstimator


@pytest.mark.parametrize("rank", [1, 2])
def test_optional_lowrank_scale_port_matches_tfp(rank):
    import tensorflow_probability as tfp
    from sgvb_psd.jax_backend.guides import legacy_lowrank_normal

    rng = np.random.default_rng(123)
    loc = rng.normal(size=(2, 4)).astype(np.float32)
    diagonal = np.full((2, 4), 0.3, np.float32)
    factor = rng.normal(0, 0.15, size=(2, 4, rank)).astype(np.float32)
    reference = tfp.distributions.MultivariateNormalDiagPlusLowRank(
        loc=loc, scale_diag=diagonal, scale_perturb_factor=factor
    )
    port = legacy_lowrank_normal(
        jnp.asarray(loc), jnp.asarray(diagonal), jnp.asarray(factor)
    )
    np.testing.assert_allclose(
        port.covariance_matrix, reference.covariance(), rtol=2e-6
    )
    value = rng.normal(size=(5, 2, 4)).astype(np.float32)
    np.testing.assert_allclose(
        port.log_prob(value), reference.log_prob(value), rtol=3e-6, atol=2e-5
    )


def paired_models(p, nchunks, n_basis=4):
    x = np.random.default_rng(41).normal(size=(192, max(2, p)))
    x /= np.std(x)
    options = dict(
        nchunks=nchunks,
        fs=16.0,
        fmin_for_analysis=1.0,
        fmax_for_analysis=7.0,
        N_delta=n_basis,
        N_theta=n_basis,
    )
    tf_data = AnalysisData(x=x, **options)
    if p == 1:
        # AnalysisData rejects p=1, but the MIT BayesianModel supports the
        # empty theta blocks. Restrict its data to make a 1D TF oracle.
        tf_data = SimpleNamespace(**vars(tf_data))
        tf_data.p = 1
        tf_data.y_ft = tf_data.y_ft[..., :1]
        tf_data.summed_periodogram = tf_data.summed_periodogram[:, :1, :1]
    ja_data = JaxAnalysisData(x=x[:, :p], **options)
    return BayesianModel(tf_data, Nbw=1.5), JaxBayesianModel(ja_data, Nbw=1.5)


def varied_params(model, draws=4):
    rng = np.random.default_rng(66)
    return tuple(
        np.asarray(a).repeat(draws, axis=0)
        + rng.normal(0, 0.025, size=(draws,) + a.shape[1:]).astype(np.float32)
        for a in model.trainable_vars
    )


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("nchunks", [1, 3])
@pytest.mark.parametrize("draws", [1, 4])
def test_log_density_and_gradients_match_tensorflow(p, nchunks, draws):
    reference, model = paired_models(p, nchunks)
    params = varied_params(model, draws)
    tf_params = tuple(tf.Variable(a) for a in params)
    actual = model.loglik(params)
    assert actual.shape == (draws,)
    # Never use legacy logpost for several draws: its scalar likelihood
    # broadcasts across the vector prior. Compare each target explicitly.
    expected = np.array(
        [
            reference.loglik(tuple(a[i : i + 1] for a in tf_params)).numpy()
            for i in range(draws)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-4)
    np.testing.assert_allclose(
        model.logprior(params),
        reference.logprior(tf_params),
        rtol=3e-6,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        jnp.sum(actual), reference.loglik(tf_params), rtol=3e-6, atol=1e-4
    )

    with tf.GradientTape() as tape:
        target = reference.loglik(tf_params) + tf.reduce_sum(
            reference.logprior(tf_params)
        )
    expected_grads = tape.gradient(target, tf_params)
    actual_grads = jax.grad(lambda z: jnp.sum(model.logpost(z)))(params)
    for a, b in zip(actual_grads, expected_grads):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=5e-4)
    np.testing.assert_allclose(
        jax.jit(model.logpost)(params),
        model.logpost(params),
        rtol=3e-6,
        atol=2e-4,
    )


@pytest.mark.parametrize("p", [1, 2, 3])
def test_psd_reconstruction_and_scaling_match_tensorflow(p):
    reference, model = paired_models(p, 3)
    params = varied_params(model, 7)
    expected = reference.compute_psd(
        [tf.constant(a) for a in params], fs=16, psd_scaling=2.5
    )
    actual = model.compute_psd(params, fs=16, psd_scaling=2.5)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=5e-5, atol=2e-6)
    samples = np.asarray(model.psd_samples(params, fs=16))
    np.testing.assert_allclose(
        samples, samples.conj().swapaxes(-1, -2), atol=1e-7
    )
    assert np.all(np.linalg.eigvalsh(samples) > 0)


def test_prior_jacobians_against_independent_scalar_formula():
    from scipy.special import expit
    from scipy.stats import halfcauchy, norm

    _, model = paired_models(1, 3)
    params = varied_params(model, 1)
    ga, local, _, _, _, _, tau, _ = params
    scale = np.exp(2 * local) * np.exp(2 * tau)
    scale = scale * 4 / (scale + 4)
    expected = np.sum(norm.logpdf(ga[..., :2], scale=np.sqrt(10)))
    expected += np.sum(norm.logpdf(ga[..., 2:], scale=scale))
    hc = expit(-np.arange(1, local.shape[-1] + 1) + model.degree_fluctuate)
    expected += np.sum(halfcauchy.logpdf(np.exp(local), scale=hc) + local)
    expected += np.sum(halfcauchy.logpdf(np.exp(tau), scale=0.01) + tau)
    np.testing.assert_allclose(model.logprior(params), expected, rtol=2e-6)


@pytest.mark.parametrize("p", [1, 2])
def test_reference_and_default_jax_guides_keep_map_locations(p):
    from numpyro.handlers import seed, substitute, trace
    from sgvb_psd.backend import ViRunner
    from sgvb_psd.jax_backend.model import PARAMETER_NAMES

    reference, model = paired_models(p, 3)
    runner = ViRunner.__new__(ViRunner)
    runner.variation_factor = 0.0
    runner.init_surrogate_posterior(reference.trainable_vars)
    # Regression for the 30% discrepancy: TF only exposes scale variables.
    assert all(
        "q_z_scale" in v.name
        for v in runner.surrogate_posterior.trainable_variables
    )
    assert not any(
        isinstance(d.distribution.loc, tf.Variable)
        for d in runner.surrogate_posterior.model
    )

    ja = JaxViRunner(np.random.default_rng(0).normal(size=(96, p)), N_theta=3)
    ja.run(ntrain_map=1, n_elbo_maximisation_steps=3, inference_size=3)
    assert all(name.endswith("_scale") for name in ja.svi_params)
    tr = trace(
        seed(substitute(ja._guide, data=ja.svi_params), rng_seed=0)
    ).get_trace()
    for name, a in zip(PARAMETER_NAMES, ja.model.trainable_vars):
        if a.size:
            np.testing.assert_array_equal(tr[name]["fn"].base_dist.loc, a[0])
