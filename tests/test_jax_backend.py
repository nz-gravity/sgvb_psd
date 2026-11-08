"""Deterministic backend parity plus bounded inference controls."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

jax = pytest.importorskip("jax")
pytest.importorskip("numpyro")
import jax.numpy as jnp

from sgvb_psd._spectral import compute_chunked_fft
from sgvb_psd.backend import AnalysisData, BayesianModel
from sgvb_psd.jax_backend import JaxAnalysisData, JaxBayesianModel, JaxViRunner
from sgvb_psd.psd_estimator import PSDEstimator


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


@pytest.mark.parametrize("length", [64, 65])
def test_fft_convention_is_unchanged(length):
    x = np.random.default_rng(10).normal(size=(3 * length + 2, 1))
    y, f = compute_chunked_fft(x, 3, 6.0, 16.0, 1.0)
    # Freeze existing odd-length behavior too: the last positive bin is
    # currently excluded. Correcting it belongs in a separate change.
    idx = length // 2
    expected = np.fft.fft(x[: 3 * length].reshape(3, length, 1), axis=1)[
        :, 1:idx
    ] / np.sqrt(length)
    fq = np.fft.fftfreq(length)[1:idx]
    keep = (fq * 16 >= 1) & (fq * 16 < 6)
    np.testing.assert_array_equal(f, fq[keep])
    np.testing.assert_allclose(y, expected[:, keep])


def test_white_noise_normalization_and_draw_independence():
    data = JaxAnalysisData(
        np.random.default_rng(40).normal(size=128),
        nchunks=2,
        fs=16,
        N_delta=2,
        N_theta=2,
    )
    model = JaxBayesianModel(data)
    params = [
        np.asarray(a).repeat(3, axis=0).copy() for a in model.trainable_vars
    ]
    variance = np.array([0.7, 1.1, 2.0], dtype=np.float32)
    params[0][:, 0, 0] = np.log(variance)
    power = np.sum(np.abs(data.y_ft) ** 2)
    expected = (
        -data.nchunks * len(data.freq) * np.log(variance) - power / variance
    )
    np.testing.assert_allclose(model.loglik(params), expected, rtol=2e-6)
    psd = np.asarray(model.psd_samples(params, fs=16))
    np.testing.assert_allclose(
        psd[:, :, 0, 0],
        np.broadcast_to(variance[:, None] / 16, psd.shape[:2]),
        rtol=2e-6,
    )
    altered = [a.copy() for a in params]
    altered[0][1, 0, 0] += 0.2
    np.testing.assert_array_equal(
        np.asarray(model.logpost(params))[[0, 2]],
        np.asarray(model.logpost(altered))[[0, 2]],
    )


def test_distinct_delta_and_theta_basis_sizes():
    model = JaxBayesianModel(
        JaxAnalysisData(
            np.random.default_rng(0).normal(size=(64, 2)), N_delta=3, N_theta=5
        )
    )
    assert model.trainable_vars[1].shape[-1] == 3
    assert np.all(np.isfinite(model.logpost(model.trainable_vars)))


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


def test_numpyro_target_has_no_duplicate_prior_or_jacobian():
    from numpyro.infer.util import log_density

    runner = JaxViRunner(np.random.default_rng(0).normal(size=64), N_theta=3)
    from sgvb_psd.jax_backend.model import PARAMETER_NAMES

    params = varied_params(runner.model, 1)
    sites = {name: a[0] for name, a in zip(PARAMETER_NAMES, params) if a.size}
    density, _ = log_density(runner._numpyro_model, (), {}, sites)
    np.testing.assert_allclose(
        density, runner.model.logpost(params)[0], rtol=2e-6
    )


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("particles", [1, 4])
def test_estimator_inference_and_rng(p, particles):
    x = np.random.default_rng(12).normal(size=(128, p))
    if p == 1:
        x = x[:, 0]
    options = dict(
        x=x,
        backend="jax",
        N_theta=3,
        nchunks=2,
        fs=16,
        N_samples=5,
        ntrain_map=3,
        n_elbo_maximisation_steps=3,
        seed=19,
        elbo_samples=particles,
    )
    first = PSDEstimator(**options)
    second = PSDEstimator(**options)
    a = first.run(lr=1e-3)
    b = second.run(lr=1e-3)
    for arr, other in zip(a, b):
        assert arr.shape == ((5 if arr is a[0] else 3), len(first.freq), p, p)
        assert np.all(np.isfinite(arr))
        np.testing.assert_array_equal(arr, other)
    assert np.all(np.isfinite(first.inference_runner.kdl_losses))
    # Continuation uses fresh keys, and the public sampling route is usable.
    params, psd = first.sample_posterior(5)
    assert not np.array_equal(params[0], first.samps[0])
    assert psd[0].shape == a[0].shape
    if p == 1 and particles == 1:
        import matplotlib.pyplot as plt

        axes = first.plot()
        assert axes.shape == (1, 1)
        plt.close(axes[0, 0].figure)


def test_jax_learning_rate_search_smoke():
    estimator = PSDEstimator(
        x=np.random.default_rng(2).normal(size=64),
        backend="jax",
        N_theta=2,
        ntrain_map=1,
        n_elbo_maximisation_steps=1,
        N_samples=3,
        max_hyperparm_eval=1,
        seed=0,
    )
    psd, _, _ = estimator.run()
    assert psd.shape == (3, len(estimator.freq), 1, 1)
    assert np.all(np.isfinite(psd))


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


def test_trainable_guide_means_are_an_explicit_jax_option():
    ja = JaxViRunner(
        np.random.default_rng(0).normal(size=64), N_theta=3, optimize_mean=True
    )
    ja.run(ntrain_map=1, n_elbo_maximisation_steps=3, inference_size=3)
    assert any(name.endswith("_loc") for name in ja.svi_params)
    assert not np.array_equal(
        ja.svi_params["ga_delta_loc"], ja.model.trainable_vars[0][0]
    )
    with pytest.raises(ValueError, match="requires backend='jax'"):
        PSDEstimator(np.ones((64, 2)), optimize_mean=True)


def test_conditional_calibration_target_and_fft_generator_match_model():
    # The calibration screen fixes curvature and its nuisance parameters.
    # Its two Normal-prior baseline coefficients must retain the exact
    # likelihood, FFT scaling and prior from the full SGVB target.
    from numpyro import distributions as dist

    rng = np.random.default_rng(44)
    n, blocks = 64, 2
    frequencies = np.fft.fftfreq(n)[1 : n // 2]
    design = np.column_stack([np.ones_like(frequencies), frequencies])
    beta = np.array([0.3, -0.7], np.float32)
    spectrum = np.exp(design @ beta)
    y = np.sqrt(spectrum / 2) * (
        rng.normal(size=(blocks, len(frequencies)))
        + 1j * rng.normal(size=(blocks, len(frequencies)))
    )
    full = np.zeros((blocks, n), dtype=complex)
    full[:, 1 : n // 2] = y
    full[:, -np.arange(1, n // 2)] = y.conj()
    x = np.fft.ifft(full * np.sqrt(n), axis=1).real.reshape(-1, 1)
    data = JaxAnalysisData(x, nchunks=blocks, N_delta=1, N_theta=1)
    np.testing.assert_allclose(data.y_ft[..., 0], y, atol=1e-12)
    power = np.sum(np.abs(y) ** 2, axis=0)
    model = JaxBayesianModel(data)

    def conditional_target(b):
        gamma = jnp.asarray(design) @ b
        return (
            -blocks * jnp.sum(gamma)
            - jnp.sum(power * jnp.exp(-gamma))
            + jnp.sum(dist.Normal(0, jnp.sqrt(10.0)).log_prob(b))
        )

    zero = model.trainable_vars
    params = list(zero)
    params[0] = params[0].at[0, 0, :2].set(beta)
    np.testing.assert_allclose(
        model.logpost(params) - model.logpost(zero),
        conditional_target(beta) - conditional_target(jnp.zeros(2)),
        rtol=2e-5,
        atol=2e-5,
    )


def test_jax_import_does_not_load_tensorflow():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import sgvb_psd.jax_backend; assert 'tensorflow' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "unknown"},
        {"backend": "jax", "use_eigenbasis": True},
        {"backend": "jax", "posterior_mode": "factorized"},
    ],
)
def test_rejects_unsupported_backends(options):
    with pytest.raises(ValueError, match="backend"):
        PSDEstimator(x=np.ones((32, 2)), **options)


@pytest.mark.parametrize(
    "x", [np.zeros(32), np.full(32, np.nan), np.ones((32, 0))]
)
def test_rejects_invalid_input(x):
    with pytest.raises(ValueError):
        PSDEstimator(x=x, backend="jax")
