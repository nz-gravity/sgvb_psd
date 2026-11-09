"""Full public JAX inference on the existing bivariate VAR(2) model."""

import numpy as np
import pytest

from sgvb_psd.psd_estimator import PSDEstimator
from sgvb_psd.utils.sim_varma import SimVARMA


@pytest.mark.parametrize("nchunks", [1, 2])
def test_var2_public_inference(nchunks):
    sim = SimVARMA(
        n_samples=512,
        var_coeffs=np.array([[[0.5, 0], [0, -0.3]], [[0, 0], [0, -0.5]]]),
        vma_coeffs=np.eye(2)[None],
        sigma=np.array([[1, 0.9], [0.9, 1]]),
        seed=4,
    )
    estimator = PSDEstimator(
        sim.data,
        N_theta=6,
        nchunks=nchunks,
        fs=sim.fs,
        ntrain_map=40,
        n_elbo_maximisation_steps=50,
        N_samples=32,
        seed=7,
    )
    samples, pointwise, uniform = estimator.run(lr=0.003)
    assert samples.shape == (32, 512 // nchunks // 2 - 1, 2, 2)
    assert pointwise.shape == uniform.shape == (3, samples.shape[1], 2, 2)
    assert all(np.all(np.isfinite(x)) for x in (samples, pointwise, uniform))
    np.testing.assert_allclose(
        samples, samples.conj().swapaxes(-1, -2), atol=2e-7
    )
    assert np.all(np.linalg.eigvalsh(samples) > 0)
    coherence = abs(samples[..., 0, 1]) ** 2 / (
        samples[..., 0, 0].real * samples[..., 1, 1].real
    )
    assert np.all((coherence >= 0) & (coherence <= 1 + 1e-6))
    np.testing.assert_allclose(
        estimator.freq,
        np.arange(1, 512 // nchunks // 2) * sim.fs / (512 // nchunks),
    )
    params, _ = estimator.sample_posterior(1)
    restored = PSDEstimator(
        sim.data, N_theta=6, nchunks=nchunks, fs=sim.fs, init_params=params
    )
    for a, b in zip(restored.inference_runner.model.trainable_vars, params):
        np.testing.assert_array_equal(a, b)
