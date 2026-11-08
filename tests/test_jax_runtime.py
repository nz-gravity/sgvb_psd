"""JAX-only runtime and scientific controls; no TensorFlow dependency."""

import subprocess
import sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from sgvb_psd._spectral import compute_chunked_fft
from sgvb_psd.jax_backend import JaxAnalysisData, JaxBayesianModel, JaxViRunner
from sgvb_psd.psd_estimator import PSDEstimator


def test_lowrank_scale_and_covariance_updates_are_distinct():
    from sgvb_psd.jax_backend.guides import (
        diagonal_plus_lowrank_scale,
        legacy_lowrank_normal,
    )
    from numpyro.distributions import LowRankMultivariateNormal

    diagonal = jnp.array([0.3, 0.6])
    factor = jnp.array([[0.2], [-0.4]])
    scale = diagonal_plus_lowrank_scale(diagonal, factor)
    faithful = legacy_lowrank_normal(jnp.zeros(2), diagonal, factor)
    covariance_form = LowRankMultivariateNormal(
        jnp.zeros(2), factor, diagonal**2
    )
    np.testing.assert_allclose(
        faithful.covariance_matrix, scale @ scale.T, rtol=1e-6
    )
    assert not np.allclose(
        faithful.covariance_matrix, covariance_form.covariance_matrix
    )


def test_public_default_runs_with_tensorflow_imports_forbidden():
    code = """
import sys, importlib.abc
class ForbidLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tensorflow','tensorflow_probability','tf_keras'}:
            raise AssertionError('legacy import attempted: '+fullname)
sys.meta_path.insert(0, ForbidLegacy())
import numpy as np
from sgvb_psd.psd_estimator import PSDEstimator
e=PSDEstimator(np.random.default_rng(0).normal(size=64),N_theta=2,
              ntrain_map=1,n_elbo_maximisation_steps=1,N_samples=3,seed=0)
assert e.backend=='jax'
assert np.all(np.isfinite(e.run(lr=0.001)[0]))
assert not any(s.startswith(('tensorflow','tf_keras')) for s in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_default_package_requirements_exclude_tensorflow():
    from importlib.metadata import requires
    from packaging.requirements import Requirement

    active = [
        Requirement(r)
        for r in requires("sgvb_psd")
        if not Requirement(r).marker
        or Requirement(r).marker.evaluate({"extra": ""})
    ]
    assert not any(
        r.name.lower() in {"tensorflow", "tensorflow-probability", "tf-keras"}
        for r in active
    )


def test_removed_tensorflow_runtime_has_explicit_error():
    with pytest.raises(ValueError, match="archived validation reference"):
        PSDEstimator(
            np.random.default_rng(0).normal(size=(64, 2)), backend="tensorflow"
        )


def test_cached_vi_uses_updated_map_locations():
    x = np.random.default_rng(7).normal(size=64)
    runner = JaxViRunner(x, N_theta=2)
    runner.map_time = 0.0
    runner.rng_key = jax.random.PRNGKey(7)
    runner.run_phase2(1)
    first = np.asarray(runner.kdl_losses)
    changed = list(runner.model.trainable_vars)
    changed[0] = changed[0].at[0, 0, 0].add(0.7)
    runner.model.trainable_vars = tuple(changed)
    runner.rng_key = jax.random.PRNGKey(7)
    runner.run_phase2(1)
    fresh = JaxViRunner(x, N_theta=2, init_params=changed)
    fresh.map_time = 0.0
    fresh.rng_key = jax.random.PRNGKey(7)
    fresh.run_phase2(1)
    np.testing.assert_allclose(runner.kdl_losses, fresh.kdl_losses, rtol=1e-6)
    assert not np.allclose(first, runner.kdl_losses)


def test_cached_map_respects_changed_learning_rate():
    x = np.random.default_rng(8).normal(size=64)
    runner = JaxViRunner(x, N_theta=2)
    initial = runner.model.trainable_vars
    runner.run_phase1(lr_map=0.001, ntrain_map=2)
    slow = runner.model.trainable_vars
    runner.model.trainable_vars = initial
    runner.run_phase1(lr_map=0.02, ntrain_map=2)
    fresh = JaxViRunner(x, N_theta=2)
    fresh.run_phase1(lr_map=0.02, ntrain_map=2)
    for a, b in zip(runner.model.trainable_vars, fresh.model.trainable_vars):
        np.testing.assert_allclose(a, b, rtol=1e-6)
    assert not np.allclose(slow[0], runner.model.trainable_vars[0])


def varied_params(model, draws=4):
    rng = np.random.default_rng(66)
    return tuple(
        np.asarray(a).repeat(draws, axis=0)
        + rng.normal(0, 0.025, size=(draws,) + a.shape[1:]).astype(np.float32)
        for a in model.trainable_vars
    )


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
        PSDEstimator(
            np.ones((64, 2)), backend="tensorflow", optimize_mean=True
        )


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


def test_lowrank_scale_handles_legacy_small_diagonal_without_squaring_condition():
    from sgvb_psd.jax_backend.guides import legacy_lowrank_normal

    q = legacy_lowrank_normal(
        jnp.zeros(4),
        jnp.full(4, 1e-4),
        jnp.array([[0.2], [0.7], [-0.3], [0.4]]),
    )
    draws = q.sample(jax.random.PRNGKey(0), (16,))
    assert draws.shape == (16, 4)
    assert np.all(np.isfinite(jax.jit(q.log_prob)(draws)))


def test_ported_lowrank_scale_guide_runs_in_numpyro():
    import numpyro
    from numpyro import distributions as dist
    from numpyro.distributions import constraints
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.optim import Adam
    from sgvb_psd.jax_backend.guides import legacy_lowrank_normal

    def model():
        numpyro.sample("z", dist.Normal(jnp.zeros(3), 1).to_event(1))

    def guide():
        diagonal = numpyro.param(
            "scale", jnp.full(3, 0.3), constraint=constraints.softplus_positive
        )
        factor = numpyro.param("factor", jnp.full((3, 1), 0.1))
        numpyro.sample(
            "z", legacy_lowrank_normal(jnp.zeros(3), diagonal, factor)
        )

    fit = SVI(model, guide, Adam(0.001), Trace_ELBO()).run(
        jax.random.PRNGKey(0), 3, progress_bar=False
    )
    assert np.all(np.isfinite(fit.losses))
    assert not np.array_equal(fit.params["scale"], jnp.full(3, 0.3))
