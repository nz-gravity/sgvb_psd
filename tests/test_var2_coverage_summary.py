"""Keep coverage denominators and uncertainty at the realization level."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from analyze_var2_coverage import summarize


def inputs(n):
    covered = np.ones((n, 16, 5), bool)
    widths = np.ones_like(covered, float)
    errors = np.zeros_like(widths)
    return covered, widths, errors


def test_correlated_frequency_points_do_not_inflate_sample_size():
    covered, widths, errors = inputs(100)
    covered[90:] = (
        False  # perfectly correlated frequencies in each realization
    )
    summaries, arrays = summarize(
        covered, widths, errors, widths, errors, np.arange(16), 100, 100, 2
    )
    np.testing.assert_allclose(arrays["coverage_completed"], 0.9)
    np.testing.assert_allclose(arrays["coverage_binomial_se"], 0.03)
    for row in summaries:
        assert 0.029 < row["frequency_average_coverage_realization_se"] < 0.032
        assert abs(row["frequency_average_coverage"] - 0.9) < 1e-14


def test_missing_fits_remain_in_all_attempted_bounds():
    covered, widths, errors = inputs(90)
    summaries, arrays = summarize(
        covered, widths, errors, widths, errors, np.arange(16), 100, 100, 2
    )
    np.testing.assert_allclose(arrays["coverage_completed"], 1)
    np.testing.assert_allclose(arrays["coverage_attempted_lower"], 0.9)
    np.testing.assert_allclose(arrays["coverage_attempted_upper"], 1)
    np.testing.assert_allclose(arrays["coverage_all_planned_lower"], 0.9)
