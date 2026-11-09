"""Frequency compression preserves complex statistics, counts and references."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgvb_psd.inference import AnalysisData, BayesianModel, CoarseGrainConfig
from sgvb_psd.inference.coarse_grain import (
    FrequencyBins,
    make_frequency_bins,
    pool_spectral_matrices,
)
from sgvb_psd.psd_estimator import PSDEstimator


@pytest.mark.parametrize(
    "options",
    [
        {"method": "other"},
        {"tolerance": 0},
        {"tolerance": np.nan},
        {"tolerance": True},
        {"max_bin_width": 0},
        {"max_bin_width": 1.2},
        {"max_bin_width": True},
        {"pilot_window": 0},
        {"pilot_window": 2},
        {"break_frequencies": (2, 1)},
        {"break_frequencies": (1, 1)},
        {"break_frequencies": (np.nan,)},
        {"break_frequencies": (-1,)},
    ],
)
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        CoarseGrainConfig(**options)


def test_fixed_partition_retains_remainder_and_is_frozen():
    config = CoarseGrainConfig(method="fixed", max_bin_width=4)
    bins = make_frequency_bins(np.arange(11.0), config=config)
    np.testing.assert_array_equal(bins.counts, [4, 4, 3])
    np.testing.assert_array_equal(bins.representatives, [1, 5, 9])
    with pytest.raises(FrozenInstanceError):
        config.max_bin_width = 2
    with pytest.raises(ValueError, match="read-only"):
        bins.starts[0] = 1


@pytest.mark.parametrize("method", ["fixed", "adaptive"])
def test_frequency_gaps_and_physical_breaks_split_smoothing_and_bins(method):
    frequency = np.r_[np.arange(1.0, 12), np.arange(19.0, 30)]
    matrix = np.broadcast_to(np.eye(2), (len(frequency), 2, 2))
    config = CoarseGrainConfig(
        method=method, max_bin_width=32, break_frequencies=(5.5, 25.0)
    )
    bins = make_frequency_bins(frequency, matrix, config)
    np.testing.assert_array_equal(bins.stops, [5, 11, 17, 22])
    assert bins.counts.sum() == len(frequency)


@pytest.mark.parametrize("eigenvalue", [0.0, 1e-10, 1.0])
def test_constant_singular_or_regular_pilot_uses_full_width(eigenvalue):
    matrix = np.broadcast_to(np.diag([1.0, eigenvalue]), (24, 2, 2))
    bins = make_frequency_bins(
        np.arange(24.0), matrix, CoarseGrainConfig(max_bin_width=8)
    )
    np.testing.assert_array_equal(bins.counts, [8, 8, 8])


def test_adaptive_bins_protect_cross_spectral_phase_changes():
    n = 48
    phase = np.where(np.arange(n) < n // 2, 0.0, np.pi)
    matrix = np.broadcast_to(
        np.diag([1.0, 3.0]).astype(complex), (n, 2, 2)
    ).copy()
    matrix[:, 0, 1] = 0.7 * np.exp(1j * phase)
    matrix[:, 1, 0] = matrix[:, 0, 1].conj()
    bins = make_frequency_bins(
        np.arange(float(n)),
        matrix,
        CoarseGrainConfig(max_bin_width=12, pilot_window=5),
    )
    assert bins.counts.max() == 12
    assert bins.counts.min() < 12
    assert not any(a < n // 2 < b for a, b in zip(bins.starts, bins.stops))


def test_pooling_retains_unsmoothed_complex_sums_and_hermiticity():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(3, 11, 2)) + 1j * rng.normal(size=(3, 11, 2))
    matrix = np.einsum("bfi,bfj->fij", y, y.conj())
    bins = make_frequency_bins(
        np.arange(11.0),
        config=CoarseGrainConfig(method="fixed", max_bin_width=4),
    )
    pooled = pool_spectral_matrices(matrix, bins)
    expected = np.array(
        [matrix[a:b].sum(0) for a, b in zip(bins.starts, bins.stops)]
    )
    np.testing.assert_allclose(pooled, expected)
    np.testing.assert_allclose(pooled.sum(0), matrix.sum(0))
    np.testing.assert_allclose(pooled, pooled.conj().swapaxes(-1, -2))
    assert np.any(abs(pooled[:, 0, 1].imag) > 0)


@pytest.mark.parametrize("frequency", [[], [1, 1], [2, 1], [1, np.nan]])
def test_invalid_frequency_grids_are_rejected(frequency):
    with pytest.raises(ValueError):
        make_frequency_bins(frequency)


def test_invalid_partitions_and_config_types_are_rejected():
    with pytest.raises(ValueError, match="cover"):
        FrequencyBins(np.array([0, 3]), np.array([2, 6]), np.array([0, 4]), 6)
    with pytest.raises(TypeError, match="CoarseGrainConfig"):
        make_frequency_bins([1, 2], config=True)
    with pytest.raises(ValueError, match="residual matrix"):
        make_frequency_bins(
            [1, 2], np.full((2, 1, 1), np.nan), CoarseGrainConfig()
        )


def varied(model, draws=4):
    rng = np.random.default_rng(12)
    return tuple(
        np.asarray(a).repeat(draws, axis=0)
        + rng.normal(0, 0.02, (draws,) + a.shape[1:]).astype(np.float32)
        for a in model.trainable_vars
    )


def reference(f):
    factor = np.zeros((len(f), 2, 2), complex)
    factor[:, 0, 0] = 0.8 + 0.15 * np.cos(2 * np.pi * f / 16)
    factor[:, 1, 0] = 0.3 * np.exp(3j * f)
    factor[:, 1, 1] = 1.1
    return (factor @ factor.conj().swapaxes(-1, -2)) / 16


@pytest.mark.parametrize("with_reference", [False, True])
def test_width_one_is_exact_native_target_gradient_and_prediction(
    with_reference,
):
    options = dict(
        x=np.random.default_rng(5).normal(size=(128, 2)),
        nchunks=2,
        fs=16,
        N_delta=3,
        N_theta=3,
        reference_psd=reference if with_reference else None,
    )
    native = BayesianModel(AnalysisData(**options))
    identity = BayesianModel(
        AnalysisData(
            **options, coarse_grain=CoarseGrainConfig(max_bin_width=1)
        )
    )
    params = varied(native)
    np.testing.assert_array_equal(
        native.logpost(params), identity.logpost(params)
    )
    np.testing.assert_array_equal(
        native.psd_samples(params), identity.psd_samples(params)
    )
    for a, b in zip(
        jax.grad(lambda z: native.logpost(z).sum())(params),
        jax.grad(lambda z: identity.logpost(z).sum())(params),
    ):
        np.testing.assert_array_equal(a, b)


def numpy_binned_raw_likelihood(model, params):
    data = model.data
    ga, _, real, _, imag, _, _, _ = params
    gamma = np.einsum(
        "fk,spk->sfp", np.asarray(data.likelihood_Xmat_delta, float), ga
    )
    theta = np.einsum(
        "fk,sek->sfe",
        np.asarray(data.likelihood_Xmat_theta, float),
        real + 1j * imag,
    )
    transform = (
        np.broadcast_to(np.eye(2), gamma.shape[:2] + (2, 2))
        .astype(complex)
        .copy()
    )
    transform[..., 1, 0] = -theta[..., 0]
    inverse = np.linalg.inv(transform)
    residual = (
        inverse * np.exp(gamma)[..., None, :]
    ) @ inverse.conj().swapaxes(-1, -2)
    # Expand the piecewise-constant residual to each original Fourier bin.
    residual = np.repeat(residual, data.freq_bin_counts, axis=1)
    if data.reference_factor is not None:
        factor = (
            np.linalg.cholesky(reference(data.freq * data.fs))
            * np.sqrt(data.fs)
            / 1.7
        )
        residual = factor @ residual @ factor.conj().swapaxes(-1, -2)
    raw = np.asarray(data.y_ft)
    power = np.einsum(
        "bfi,sfij,bfj->s", raw.conj(), np.linalg.inv(residual), raw
    ).real
    determinant = np.linalg.slogdet(residual)[1].sum(-1)
    return (-data.nchunks * determinant - power) / model.Nbw


@pytest.mark.parametrize("with_reference", [False, True])
def test_counts_and_raw_gaussian_likelihood_gradients(with_reference):
    data = AnalysisData(
        np.random.default_rng(8).normal(size=(192, 2)) / 1.7,
        nchunks=3,
        fs=16,
        N_delta=3,
        N_theta=4,
        data_scale=1.7,
        reference_psd=reference if with_reference else None,
        coarse_grain=CoarseGrainConfig(method="fixed", max_bin_width=6),
    )
    model = BayesianModel(data, Nbw=1.4)
    params = varied(model)
    assert data.freq_bin_counts[-1] == 1  # Keep the partial last bin.
    np.testing.assert_array_equal(data.counts, 3 * data.freq_bin_counts)
    np.testing.assert_allclose(
        model.loglik(params),
        numpy_binned_raw_likelihood(model, params),
        rtol=5e-6,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        jax.jit(model.loglik)(params),
        numpy_binned_raw_likelihood(model, params),
        rtol=5e-6,
        atol=5e-4,
    )
    direction = np.random.default_rng(9).normal(size=params[0].shape)
    direction /= np.linalg.norm(direction)
    step = 1e-4
    plus = (params[0].astype(float) + step * direction,) + params[1:]
    minus = (params[0].astype(float) - step * direction,) + params[1:]
    finite_difference = (
        numpy_binned_raw_likelihood(model, plus).sum()
        - numpy_binned_raw_likelihood(model, minus).sum()
    ) / (2 * step)
    gradient = jax.grad(lambda ga: model.loglik((ga,) + params[1:]).sum())(
        params[0]
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(gradient) * direction),
        finite_difference,
        rtol=4e-4,
        atol=2e-3,
    )
    changed = [a.copy() for a in params]
    changed[0][1, 0, 0] += 0.2
    np.testing.assert_array_equal(
        np.asarray(model.loglik(params))[[0, 2, 3]],
        np.asarray(model.loglik(changed))[[0, 2, 3]],
    )


def test_piecewise_constant_residual_target_is_exact_and_predictions_stay_native():
    x = np.random.default_rng(10).normal(size=(256, 2))
    options = dict(
        x=x, nchunks=2, fs=16, N_delta=3, N_theta=3, reference_psd=reference
    )
    native = BayesianModel(AnalysisData(**options))
    coarse = BayesianModel(
        AnalysisData(
            **options,
            coarse_grain=CoarseGrainConfig(method="fixed", max_bin_width=5),
        )
    )
    params = varied(native)
    # Prediction still evaluates the original cosine basis, without remapping.
    np.testing.assert_array_equal(
        coarse.psd_samples(params), native.psd_samples(params)
    )
    np.testing.assert_array_equal(
        coarse.data.likelihood_Xmat_delta,
        native.data.Xmat_delta[coarse.data.frequency_bins.representatives],
    )
    # Make the independent fine-grid control constant within the same bins.
    native.data.likelihood_Xmat_delta = jnp.repeat(
        coarse.data.likelihood_Xmat_delta, coarse.data.freq_bin_counts, axis=0
    )
    native.data.likelihood_Xmat_theta = jnp.repeat(
        coarse.data.likelihood_Xmat_theta, coarse.data.freq_bin_counts, axis=0
    )
    np.testing.assert_allclose(
        coarse.logpost(params), native.logpost(params), rtol=3e-6, atol=3e-4
    )
    for a, b in zip(
        jax.grad(lambda z: coarse.logpost(z).sum())(params),
        jax.grad(lambda z: native.logpost(z).sum())(params),
    ):
        np.testing.assert_allclose(a, b, rtol=5e-5, atol=3e-4)


def test_narrow_reference_notch_and_phase_remain_on_native_grid():
    x = np.random.default_rng(11).normal(size=(512, 2))
    n = 255
    lower = np.broadcast_to(np.eye(2), (n, 2, 2)).astype(complex).copy()
    lower[:, 0, 0] = 1.0
    lower[64, 0, 0] = 0.01
    lower[:, 1, 0] = 0.3 * np.exp(1j * np.arange(n))
    supplied = (lower @ lower.conj().swapaxes(-1, -2)) * 1e-40 / 16
    model = BayesianModel(
        AnalysisData(
            x,
            fs=16,
            N_delta=3,
            N_theta=3,
            reference_psd=supplied,
            data_scale=1e-20,
            coarse_grain=CoarseGrainConfig(method="fixed", max_bin_width=32),
        )
    )
    assert len(model.data.likelihood_freq) == 8
    actual = (
        np.asarray(model.psd_samples(model.trainable_vars, fs=16), complex)[0]
        * 1e-40
    )
    np.testing.assert_allclose(actual, supplied, rtol=4e-6, atol=1e-48)
    assert actual[64, 0, 0].real < actual[63, 0, 0].real * 0.001


@pytest.mark.parametrize(
    "p,parallel,method",
    [(1, False, "adaptive"), (2, True, "fixed"), (3, True, "adaptive")],
)
def test_public_fit_returns_native_grid_and_reproducible_samples(
    p, parallel, method
):
    x = np.random.default_rng(12).normal(size=(192, p))
    options = dict(
        x=x,
        N_theta=3,
        nchunks=3,
        fs=16,
        seed=7,
        N_samples=5,
        ntrain_map=2,
        n_elbo_maximisation_steps=2,
        parallel_rows=parallel,
        fmin_for_analysis=1.0,
        fmax_for_analysis=6.0,
        coarse_grain=CoarseGrainConfig(
            method=method, max_bin_width=8, tolerance=0.3
        ),
        reference_psd=1 / 16,
    )
    first, second = PSDEstimator(**options), PSDEstimator(**options)
    result = first.run(lr=0.003)
    other = second.run(lr=0.003)
    data = first.inference_runner.data
    assert len(data.likelihood_freq) < len(data.freq)
    assert int(data.freq_bin_counts.sum()) == len(data.freq)
    np.testing.assert_array_equal(first.freq, data.freq * 16)
    assert result[0].shape == (5, len(first.freq), p, p)
    for a, b in zip(result, other):
        assert np.isfinite(a).all()
        np.testing.assert_array_equal(a, b)
    assert np.all(np.linalg.eigvalsh(result[0]) > 0)
