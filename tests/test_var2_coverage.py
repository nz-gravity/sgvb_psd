"""Independent mathematical checks for the fixed-truth coverage experiment."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "var2_coverage",
    Path(__file__).resolve().parents[1] / "validation/var2_coverage.py",
)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_var2_spectrum_units_phase_and_integrated_covariance():
    freq = np.array([0.0, 0.125, 0.25, 0.375, 0.5])
    z = np.exp(-2j * np.pi * freq)
    h0 = 1 / (1 - 0.5 * z)
    h1 = 1 / (1 + 0.3 * z + 0.5 * z * z)
    spectral = campaign.analytic_spectrum(freq, 2 * np.pi)
    np.testing.assert_allclose(spectral[:, 0, 0], abs(h0) ** 2 / (2 * np.pi))
    np.testing.assert_allclose(spectral[:, 1, 1], abs(h1) ** 2 / (2 * np.pi))
    np.testing.assert_allclose(
        spectral[:, 0, 1], 0.9 * h0 * h1.conj() / (2 * np.pi)
    )
    # For diagonal AR filters, population coherence is constant rho^2.
    np.testing.assert_allclose(campaign.functionals(spectral)[:, -1], 0.81)
    lags, radius = campaign.stationary_covariances(16)
    assert radius < 1
    grid = np.arange(4096) / 4096
    integrated = (
        campaign.analytic_spectrum(grid, 2 * np.pi).mean(0) * 2 * np.pi
    )
    np.testing.assert_allclose(integrated, lags[0], atol=1e-13)


def test_expected_window_matches_direct_toeplitz_dft():
    n = 16
    freq = np.fft.fftfreq(n)[1 : n // 2]
    lags, _ = campaign.stationary_covariances(n)
    direct = []
    for f in freq:
        total = np.zeros((2, 2), complex)
        for t in range(n):
            for s in range(n):
                cov = lags[t - s] if t >= s else lags[s - t].T
                total += cov * np.exp(-2j * np.pi * f * (t - s)) / n
        direct.append(total / (2 * np.pi))
    np.testing.assert_allclose(
        campaign.expected_window_spectrum(freq, n, 2 * np.pi),
        direct,
        atol=1e-14,
    )


def test_binomial_intervals_preserve_realization_denominator():
    bounds = campaign.wilson(np.array([0, 90, 100]), 100)
    np.testing.assert_allclose(bounds[1], [0.82563434, 0.94477086], atol=1e-8)
    assert abs(bounds[0, 0]) < 1e-14
    assert abs(bounds[2, 1] - 1) < 1e-14
