"""Reference spectra are part of inference and physical reconstruction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgvb_psd.jax_backend import JaxAnalysisData, JaxBayesianModel
from sgvb_psd.psd_estimator import PSDEstimator


def matrix_reference(frequencies, fs=16.0, amplitude=1.0):
    f = np.asarray(frequencies) / fs
    lower = np.zeros((len(f), 2, 2), complex)
    lower[:, 0, 0] = 0.8 + 0.2 * np.cos(2 * np.pi * f)
    lower[:, 1, 0] = 0.25 * np.exp(2j * np.pi * f)
    lower[:, 1, 1] = 1.1 + 0.1 * np.sin(2 * np.pi * f)
    return amplitude * (lower @ lower.conj().swapaxes(-1, -2)) / fs


def params_for(model, draws=4):
    rng = np.random.default_rng(9)
    return tuple(
        np.asarray(a).repeat(draws, axis=0)
        + rng.normal(0, 0.02, (draws,) + a.shape[1:]).astype(np.float32)
        for a in model.trainable_vars
    )


def numpy_spectrum(model, params, reference, data_scale):
    ga, _, real, _, imag, _, _, _ = params
    gamma = np.einsum(
        "fk,spk->sfp", np.asarray(model.data.Xmat_delta, dtype=float), ga
    )
    theta = np.einsum(
        "fk,sek->sfe",
        np.asarray(model.data.Xmat_theta, dtype=float),
        real + 1j * imag,
    )
    t = np.broadcast_to(
        np.eye(2, dtype=complex), gamma.shape[:2] + (2, 2)
    ).copy()
    t[..., 1, 0] = -theta[..., 0]
    precision = t.conj().swapaxes(-1, -2) @ (np.exp(-gamma)[..., :, None] * t)
    residual = np.linalg.inv(precision)
    factor = (
        np.linalg.cholesky(reference) * np.sqrt(model.data.fs) / data_scale
    )
    return factor @ residual @ factor.conj().swapaxes(-1, -2)


def direct_raw_likelihood(model, spectrum):
    # Evaluate the raw complex Gaussian density, independently of whitening.
    raw = np.asarray(model.data.y_ft)
    quadratic = np.einsum(
        "bfi,sfij,bfj->s", raw.conj(), np.linalg.inv(spectrum), raw
    ).real
    logdet = np.linalg.slogdet(spectrum)[1].sum(-1)
    return (-model.data.nchunks * logdet - quadratic) / model.Nbw


def test_scalar_reference_is_exact_log_additive_model():
    fs = 16.0
    scale = 2.3

    def reference(f):
        return 0.4 / (fs * abs(1 - 0.7 * np.exp(-2j * np.pi * f / fs)) ** 2)

    data = JaxAnalysisData(
        np.random.default_rng(0).normal(size=128) / scale,
        nchunks=2,
        fs=fs,
        N_theta=3,
        N_delta=3,
        reference_psd=reference,
        data_scale=scale,
    )
    model = JaxBayesianModel(data)
    params = params_for(model)
    gamma = np.einsum("fk,spk->sfp", np.asarray(data.Xmat_delta), params[0])[
        :, :, 0
    ]
    actual = (
        np.asarray(model.psd_samples(params, fs=fs), dtype=complex)[
            :, :, 0, 0
        ].real
        * scale**2
    )
    expected = reference(data.freq * fs)[None] * np.exp(gamma)
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    # Omitting the reference log-determinant would fail this density check.
    full_variance = expected * fs / scale**2
    power = np.sum(abs(data.y_ft[..., 0]) ** 2, axis=0)
    ll = -np.sum(
        data.nchunks * np.log(full_variance) + power / full_variance, axis=1
    )
    np.testing.assert_allclose(model.loglik(params), ll, rtol=4e-6, atol=1e-4)


@pytest.mark.parametrize("amplitude", [1.0, 1e-40])
def test_zero_residual_recovers_full_reference_in_physical_units(amplitude):
    fs = 16.0
    scale = 2.5 * np.sqrt(amplitude)
    data = JaxAnalysisData(
        np.random.default_rng(1).normal(size=(128, 2)) / 2.5,
        nchunks=2,
        fs=fs,
        N_theta=3,
        N_delta=3,
        reference_psd=lambda f: matrix_reference(f, fs, amplitude),
        data_scale=scale,
    )
    model = JaxBayesianModel(data)
    raw = (
        np.asarray(
            model.psd_samples(model.trainable_vars, fs=fs), dtype=complex
        )
        * scale**2
    )
    np.testing.assert_allclose(
        raw[0],
        matrix_reference(data.freq * fs, fs, amplitude),
        rtol=3e-6,
        atol=amplitude * 1e-9,
    )
    # The reference is fixed; priors remain priors on residual coefficients.
    plain = JaxBayesianModel(
        JaxAnalysisData(data.x, nchunks=2, fs=fs, N_theta=3, N_delta=3)
    )
    np.testing.assert_array_equal(
        model.logprior(model.trainable_vars),
        plain.logprior(plain.trainable_vars),
    )


def test_full_matrix_target_gradients_and_reconstruction_match_raw_density():
    scale = 1.7
    data = JaxAnalysisData(
        np.random.default_rng(3).normal(size=(192, 2)) / scale,
        nchunks=3,
        fs=16,
        N_theta=3,
        N_delta=3,
        reference_psd=matrix_reference,
        data_scale=scale,
    )
    model = JaxBayesianModel(data, Nbw=1.5)
    params = params_for(model)
    ref = matrix_reference(data.freq * data.fs)
    expected_spectrum = numpy_spectrum(model, params, ref, scale)
    np.testing.assert_allclose(
        model.psd_samples(params), expected_spectrum, rtol=5e-6, atol=1e-7
    )
    expected = direct_raw_likelihood(model, expected_spectrum)
    np.testing.assert_allclose(
        model.loglik(params), expected, rtol=4e-6, atol=3e-4
    )
    np.testing.assert_allclose(
        jax.jit(model.loglik)(params), expected, rtol=4e-6, atol=3e-4
    )
    gradient = jax.grad(lambda a: jnp.sum(model.loglik((a,) + params[1:])))(
        params[0]
    )
    direction = np.random.default_rng(4).normal(size=params[0].shape)
    direction /= np.linalg.norm(direction)
    step = 1e-4
    plus = (params[0].astype(float) + step * direction,) + params[1:]
    minus = (params[0].astype(float) - step * direction,) + params[1:]
    fd = (
        direct_raw_likelihood(
            model, numpy_spectrum(model, plus, ref, scale)
        ).sum()
        - direct_raw_likelihood(
            model, numpy_spectrum(model, minus, ref, scale)
        ).sum()
    ) / (2 * step)
    np.testing.assert_allclose(
        np.sum(np.asarray(gradient) * direction), fd, rtol=5e-4, atol=2e-3
    )
    altered = [a.copy() for a in params]
    altered[0][1, 0, 0] += 0.2
    np.testing.assert_array_equal(
        np.asarray(model.loglik(params))[[0, 2, 3]],
        np.asarray(model.loglik(altered))[[0, 2, 3]],
    )


def test_no_reference_and_identity_reference_retain_original_model():
    x = np.random.default_rng(5).normal(size=(128, 2))
    options = dict(nchunks=2, fs=16, N_theta=3, N_delta=3)
    plain = JaxBayesianModel(JaxAnalysisData(x, **options))
    explicit = JaxBayesianModel(
        JaxAnalysisData(x, reference_psd=None, **options)
    )
    identity = JaxBayesianModel(
        JaxAnalysisData(x, reference_psd=1 / 16, **options)
    )
    params = params_for(plain)
    for other in (explicit, identity):
        np.testing.assert_array_equal(
            plain.logpost(params), other.logpost(params)
        )
        np.testing.assert_array_equal(
            plain.psd_samples(params), other.psd_samples(params)
        )


def test_reference_table_selects_analysis_band_without_interpolation():
    fs = 16.0
    full_frequency = np.fft.rfftfreq(64, 1 / fs)
    values = matrix_reference(full_frequency)
    data = JaxAnalysisData(
        np.ones((128, 2)),
        nchunks=2,
        fs=fs,
        N_theta=3,
        N_delta=3,
        fmin_for_analysis=2,
        fmax_for_analysis=6,
        reference_psd=(values, full_frequency),
    )
    np.testing.assert_allclose(
        data.reference_psd, matrix_reference(data.freq * fs)
    )
    with pytest.raises(ValueError, match="analysis frequencies"):
        JaxAnalysisData(
            np.ones((128, 2)),
            nchunks=2,
            fs=fs,
            reference_psd=(values, full_frequency + 0.05),
        )


@pytest.mark.parametrize(
    "reference,match",
    [
        (0.0, "positive definite"),
        (-1.0, "positive definite"),
        (np.nan, "finite"),
        (np.ones(4), "shape|scalar"),
        (1 + 1j, "Hermitian"),
        (
            lambda f: np.broadcast_to(
                np.array([[1, 1j], [1j, 1]]), (len(f), 2, 2)
            ),
            "Hermitian",
        ),
        (
            lambda f: np.broadcast_to(
                np.array([[1, 2], [2, 1]]), (len(f), 2, 2)
            ),
            "positive definite",
        ),
        ((np.ones((2, 2)), np.array([1.0, 0.5])), "strictly increasing"),
    ],
)
def test_invalid_references_are_rejected(reference, match):
    with pytest.raises(ValueError, match=match):
        JaxAnalysisData(
            np.ones((128, 2)), nchunks=2, fs=16, reference_psd=reference
        )


def test_public_estimator_uses_supplied_reference_during_fit():
    seen = []

    def reference(f):
        seen.append(f.copy())
        return matrix_reference(f, fs=8)

    estimator = PSDEstimator(
        np.random.default_rng(8).normal(size=(128, 2)),
        fs=8,
        nchunks=2,
        N_theta=3,
        ntrain_map=3,
        n_elbo_maximisation_steps=3,
        N_samples=8,
        seed=0,
        reference_psd=reference,
    )
    raw, pointwise, _ = estimator.run(lr=0.001)
    assert len(seen) == 1
    np.testing.assert_allclose(seen[0], estimator.freq)
    np.testing.assert_allclose(
        estimator.model.data.reference_psd,
        matrix_reference(estimator.freq, fs=8),
    )
    assert raw.shape == (8, len(estimator.freq), 2, 2)
    assert np.all(np.isfinite(pointwise))
    assert np.all(np.linalg.eigvalsh(raw) > 0)


def test_three_channel_reference_preserves_a_narrow_dip_without_basis_fitting():
    fs = 8.0

    def reference(f):
        # Positive, very narrow dip exactly on an analysis bin; full complex
        # mixing makes this a matrix test rather than three independent curves.
        notch = 0.00001 + np.sin(np.pi * (f - 1.5) / fs) ** 2
        scales = np.column_stack(
            [np.sqrt(notch), np.ones(len(f)), 1.2 * np.ones(len(f))]
        )
        mixing = np.array([[1, 0, 0], [0.2j, 1, 0], [0.1, 0.15j, 1]], complex)
        factor = scales[..., :, None] * mixing
        return factor @ factor.conj().swapaxes(-1, -2)

    data = JaxAnalysisData(
        np.random.default_rng(33).normal(size=(128, 3)),
        nchunks=2,
        fs=fs,
        N_delta=1,
        N_theta=1,
        reference_psd=reference,
    )
    model = JaxBayesianModel(data)
    expected = reference(data.freq * fs)
    actual = np.asarray(model.psd_samples(model.trainable_vars, fs=fs))[0]
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-8)
    at_dip = np.argmin(abs(data.freq * fs - 1.5))
    assert actual[at_dip, 0, 0].real < 0.1 * actual[at_dip + 1, 0, 0].real
