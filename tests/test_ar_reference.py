import numpy as np
import pytest
from scipy import linalg, signal

from sgvb_psd.ar_reference import ARReference, fit_ar_reference
from sgvb_psd.inference import AnalysisData, BayesianModel


def test_scalar_fit_matches_independent_yule_walker():
    rng = np.random.default_rng(19)
    x = signal.lfilter([1], [1, -0.7, 0.15], rng.normal(size=8192))
    reference = fit_ar_reference(x, 128, order=8, ridge=0)
    centered = x - x.mean()
    acov = np.array(
        [
            np.dot(centered[k:], centered[: len(x) - k]) / len(x)
            for k in range(9)
        ]
    )
    coefficients = linalg.solve_toeplitz(acov[:-1], acov[1:])
    variance = acov[0] - coefficients @ acov[1:]
    np.testing.assert_allclose(
        reference.coefficients[:, 0, 0], coefficients, atol=1e-12
    )
    np.testing.assert_allclose(
        reference.innovation_covariance[0, 0]
        * reference.channel_scale[0] ** 2,
        variance,
        rtol=1e-12,
    )


def test_two_sided_units_and_tiny_strain_scales():
    fs, phi, variance, scale = 128.0, 0.8, 2.0, 1e-23
    reference = ARReference(
        fs, np.array([[[phi]]]), np.array([[variance]]), np.array([scale])
    )
    frequencies = np.linspace(0, fs / 2, 16385)
    expected = (
        variance
        * scale**2
        / (fs * abs(1 - phi * np.exp(-2j * np.pi * frequencies / fs)) ** 2)
    )
    actual = reference(frequencies)[:, 0, 0].real
    np.testing.assert_allclose(actual, expected, rtol=2e-14)
    np.testing.assert_allclose(
        2 * np.trapezoid(actual, frequencies),
        variance * scale**2 / (1 - phi**2),
        rtol=1e-12,
    )
    assert reference(frequencies[:0]).shape == (0, 1, 1)
    with pytest.raises(ValueError):
        reference.coefficients[0, 0, 0] = 0


def test_fixed_order_reference_captures_a_narrow_line():
    rng = np.random.default_rng(42)
    fs, frequency, radius = 1024.0, 60.0, 0.995
    x = signal.lfilter(
        [1],
        [1, -2 * radius * np.cos(2 * np.pi * frequency / fs), radius**2],
        rng.normal(size=65536),
    )
    reference = fit_ar_reference(x * 1e-23, fs, order=512)
    f = np.linspace(50, 70, 2001)
    spectrum = reference(f)[:, 0, 0].real
    # Finite-sample peak location must lie within half the resonance half-width.
    half_width = fs * (1 - radius) / (2 * np.pi)
    assert abs(f[np.argmax(spectrum)] - frequency) < half_width / 2
    assert spectrum.max() > 50 * min(spectrum[0], spectrum[-1])
    assert reference.order == 512


def test_full_var_block_orientation_and_covariance():
    rng = np.random.default_rng(71)
    n, order, p = 4096, 3, 2
    x = rng.normal(size=(n, p))
    x[1:, 1] += 0.7 * x[:-1, 0]
    reference = fit_ar_reference(x, 32, order, full_matrix=True, ridge=0)
    z = (x - x.mean(0)) / x.std(0)
    # Direct lagged outer products independently check the FFT and block signs.
    acov = np.array([z[k:].T @ z[: n - k] / n for k in range(order + 1)])
    system = np.block(
        [
            [acov[m - l] if m >= l else acov[l - m].T for m in range(order)]
            for l in range(order)
        ]
    )
    cross = np.vstack([acov[k].T for k in range(1, order + 1)])
    solution = np.linalg.solve(system, cross)
    expected = solution.reshape(order, p, p).swapaxes(-1, -2)
    np.testing.assert_allclose(reference.coefficients, expected, atol=1e-13)
    np.testing.assert_allclose(
        reference.innovation_covariance,
        acov[0] - cross.T @ solution,
        atol=1e-13,
    )
    f = np.linspace(0, 16, 33)
    density = reference(f)
    assert np.max(abs(density[:, 0, 1].imag)) > 0.005
    # x1[t] includes delayed x0: S01 has positive phase at a quarter sample rate.
    assert density[len(f) // 2, 0, 1].imag > 0.005
    np.testing.assert_allclose(density, density.conj().swapaxes(-1, -2))
    assert np.all(np.linalg.eigvalsh(density) > 0)
    diagonal = fit_ar_reference(x, 32, order=3)(f)
    np.testing.assert_array_equal(diagonal[:, 0, 1], 0)


def test_correlated_var_simulation_recovers_known_spectrum():
    rng = np.random.default_rng(5)
    n, fs = 65536, 16.0
    phi = np.array([0.8, -0.4])
    mix = np.array([[1.0, 0.3], [0.7, 1.0]])
    independent = np.column_stack(
        [signal.lfilter([1], [1, -value], rng.normal(size=n)) for value in phi]
    )
    x = independent @ mix.T * 1e-23
    reference = fit_ar_reference(x, fs, order=1, full_matrix=True)
    f = np.linspace(0.1, 7.9, 100)
    powers = 1 / (
        fs * abs(1 - phi[None, :] * np.exp(-2j * np.pi * f[:, None] / fs)) ** 2
    )
    truth = np.einsum("ij,fj,kj->fik", mix, powers, mix) * 1e-46
    np.testing.assert_allclose(
        reference(f) / 1e-46, truth / 1e-46, rtol=0.08, atol=0.003
    )
    # Zero residual coefficients must reconstruct the reference once, in input units.
    data = AnalysisData(
        x[:1024] / x.std(),
        nchunks=2,
        fs=fs,
        N_theta=2,
        N_delta=2,
        reference_psd=reference,
        data_scale=float(x.std()),
    )
    model = BayesianModel(data)
    physical = (
        np.asarray(model.psd_samples(model.trainable_vars, fs=fs))[0]
        * x.std() ** 2
    )
    np.testing.assert_allclose(
        physical / 1e-46,
        reference(data.freq * fs) / 1e-46,
        rtol=5e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": 0},
        {"order": True},
        {"order": 32},
        {"order": 1.5},
        {"fs": 0},
        {"fs": np.inf},
        {"fs": True},
        {"ridge": -1},
        {"ridge": np.nan},
        {"ridge": True},
        {"full_matrix": "yes"},
    ],
)
def test_invalid_fit_options(kwargs):
    options = dict(fs=16, order=2)
    options.update(kwargs)
    with pytest.raises(ValueError):
        fit_ar_reference(np.arange(32.0), **options)


@pytest.mark.parametrize(
    "x",
    [
        np.ones(32),
        np.empty((32, 0)),
        np.zeros((2, 2, 2)),
        np.array([1.0, np.nan, 3.0]),
        np.arange(8) + 1j,
    ],
)
def test_invalid_training_data(x):
    with pytest.raises(ValueError):
        fit_ar_reference(x, fs=16, order=1)


@pytest.mark.parametrize("f", [[-1], [9], [np.inf], [np.nan], [[1]], [1 + 1j]])
def test_invalid_reference_frequencies(f):
    ref = fit_ar_reference(np.arange(32.0), 16, order=2)
    with pytest.raises(ValueError):
        ref(f)
