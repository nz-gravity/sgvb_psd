"""Row factorization preserves the joint model and batches independent fits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import distributions as dist
from numpyro.infer.util import log_density

from sgvb_psd.jax_backend import JaxAnalysisData, JaxBayesianModel, JaxViRunner
from sgvb_psd.jax_backend.row_vi_runner import JaxRowViRunner
from sgvb_psd.psd_estimator import PSDEstimator


def varied(model):
    rng = np.random.default_rng(15)
    return tuple(
        np.asarray(a).repeat(3, axis=0)
        + rng.normal(0, 0.01, (3,) + a.shape[1:]).astype(np.float32)
        for a in model.trainable_vars
    )


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("reference", [False, True])
def test_row_targets_and_gradients_sum_to_joint(p, reference):
    def ref(f):
        l = np.eye(p) + 0.1 * np.tril(np.ones((p, p)), -1) * 1j
        return np.broadcast_to(l @ l.conj().T / 8, (len(f), p, p))

    data = JaxAnalysisData(
        np.random.default_rng(4).normal(size=(96, p)),
        nchunks=2,
        fs=8,
        N_theta=3,
        N_delta=3,
        reference_psd=ref if reference else None,
    )
    model = JaxBayesianModel(data, Nbw=1.4)
    params = varied(model)
    rows = model.logpost_rows(params)
    assert rows.shape == (3, p)
    np.testing.assert_allclose(
        rows.sum(-1), model.logpost(params), rtol=3e-6, atol=3e-4
    )
    a = jax.grad(lambda z: model.logpost_rows(z).sum())(params)
    b = jax.grad(lambda z: model.logpost(z).sum())(params)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=3e-5, atol=3e-4)


def test_row_coordinate_partition_padding_and_density():
    runner = JaxRowViRunner(
        np.random.default_rng(0).normal(size=(96, 3)), N_theta=3
    )
    assert runner.row_dimensions == (9, 30, 51)
    packed = runner._pack(runner.model.trainable_vars)
    original = runner.model.logpost_rows(runner.model.trainable_vars)[0]
    for i in range(3):
        row, indices, mask = (
            runner._rows[i],
            runner._indices[i],
            runner._mask[i],
        )
        actual = runner._row_target(packed[i], row, indices, mask)
        np.testing.assert_allclose(actual, original[i], rtol=1e-6)
        changed = packed[i] + jnp.where(mask, 0.0, 31.0)
        np.testing.assert_array_equal(
            runner._row_target(changed, row, indices, mask), actual
        )
        density, _ = log_density(
            runner._row_model,
            (row, packed[i], indices, mask),
            {},
            {"row_parameters": packed[i]},
        )
        aux = dist.Normal(0, 1).log_prob(packed[i])
        np.testing.assert_allclose(
            density - jnp.sum(jnp.where(mask, 0, aux)),
            actual,
            rtol=1e-6,
            atol=1e-5,
        )
    # A row update cannot alter another row's posterior contribution.
    updated = runner._merge(packed.at[1, 0].add(0.3))
    params = tuple(a[None] for a in runner._unravel(updated))
    np.testing.assert_array_equal(
        runner.model.logpost_rows(params)[0, jnp.array([0, 2])],
        original[jnp.array([0, 2])],
    )


def test_parallel_map_matches_joint_parameter_updates():
    x = np.random.default_rng(2).normal(size=(96, 2))
    joint = JaxViRunner(x, N_theta=3, seed=0)
    rows = JaxRowViRunner(x, N_theta=3, seed=0)
    for runner in (joint, rows):
        runner.run_phase1(lr_map=0.001, ntrain_map=3)
    for a, b in zip(joint.model.trainable_vars, rows.model.trainable_vars):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("optimize_mean", [False, True])
@pytest.mark.parametrize("particles", [1, 4])
def test_parallel_public_runner_reproducible_and_finite(
    optimize_mean, particles
):
    options = dict(
        x=np.random.default_rng(2).normal(size=(96, 3)),
        N_theta=3,
        nchunks=2,
        ntrain_map=2,
        n_elbo_maximisation_steps=3,
        N_samples=5,
        seed=31,
        parallel_rows=True,
        optimize_mean=optimize_mean,
        elbo_samples=particles,
        reference_psd=lambda f: np.broadcast_to(np.eye(3) / 8, (len(f), 3, 3)),
        fs=8,
    )
    first = PSDEstimator(**options)
    second = PSDEstimator(**options)
    a = first.run(lr=0.001)
    b = second.run(lr=0.001)
    assert isinstance(first.inference_runner, JaxRowViRunner)
    assert first.inference_runner.row_kdl_losses.shape == (3, 3)
    for x, y in zip(a, b):
        assert np.all(np.isfinite(x))
        np.testing.assert_array_equal(x, y)
    assert (
        any(n.endswith("_loc") for n in first.inference_runner.svi_params)
        == optimize_mean
    )
    assert np.all(np.linalg.eigvalsh(a[0]) > 0)
    with pytest.raises(ValueError, match="boolean"):
        PSDEstimator(options["x"], parallel_rows="yes")


def test_cached_row_vi_uses_current_map_locations():
    x = np.random.default_rng(21).normal(size=(64, 2))
    runner = JaxRowViRunner(x, N_theta=2, seed=0)
    runner.map_time = 0.0
    runner.run_phase2(1)
    initial = np.asarray(runner.kdl_losses)
    changed = list(runner.model.trainable_vars)
    changed[0] = changed[0].at[0, 0, 0].add(0.6)
    runner.model.trainable_vars = tuple(changed)
    fresh = JaxRowViRunner(x, N_theta=2, seed=0, init_params=changed)
    fresh.map_time = 0.0
    for current in (runner, fresh):
        current.rng_key = jax.random.PRNGKey(1)
        current.run_phase2(1)
    np.testing.assert_allclose(runner.kdl_losses, fresh.kdl_losses, rtol=2e-6)
    assert not np.allclose(initial, runner.kdl_losses)
