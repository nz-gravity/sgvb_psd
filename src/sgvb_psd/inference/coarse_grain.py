"""Optional, frozen frequency partitions for the residual Whittle likelihood.

The adaptive criterion follows the pilot-guided matrix approach used in
LogPSplinePSD: smoothing selects boundaries; unsmoothed complex spectral
matrices and their observation counts enter inference. This is an
approximation, with no posterior-error guarantee from the pilot tolerance.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np


def _positive_integer(name, value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class CoarseGrainConfig:
    """Opt in to fixed-width or adaptive residual-frequency aggregation.

    ``max_bin_width`` is a number of original Fourier bins. Adaptive bins
    bound the pilot's relative matrix deviation from their representative
    bin by ``tolerance``. ``pilot_window`` must be a positive odd integer.
    ``break_frequencies`` are physical frequencies which no bin may cross;
    place breaks on either side of a narrow feature to protect that region.
    Pass None, rather than a config, to disable aggregation.
    """

    method: str = "adaptive"
    tolerance: float = 0.15
    max_bin_width: int = 32
    pilot_window: int = 31
    break_frequencies: tuple[float, ...] = ()

    def __post_init__(self):
        if self.method not in ("fixed", "adaptive"):
            raise ValueError("method must be 'fixed' or 'adaptive'")
        if (
            isinstance(self.tolerance, (bool, np.bool_))
            or not isinstance(self.tolerance, Real)
            or not np.isfinite(self.tolerance)
            or self.tolerance <= 0
        ):
            raise ValueError("tolerance must be finite and positive")
        for name in ("max_bin_width", "pilot_window"):
            object.__setattr__(
                self, name, _positive_integer(name, getattr(self, name))
            )
        if self.pilot_window % 2 != 1:
            raise ValueError("pilot_window must be odd")
        breaks = np.asarray(self.break_frequencies, dtype=float)
        if (
            breaks.ndim != 1
            or not np.all(np.isfinite(breaks))
            or np.any(breaks < 0)
            or np.any(np.diff(breaks) <= 0)
        ):
            raise ValueError(
                "break_frequencies must be finite, nonnegative and increasing"
            )
        object.__setattr__(self, "break_frequencies", tuple(breaks.tolist()))


@dataclass(frozen=True, slots=True)
class FrequencyBins:
    """A complete contiguous partition of the retained native grid.

    Stops are exclusive. Read-only index arrays keep a partition fixed
    throughout optimization; representatives are original Fourier bins.
    """

    starts: np.ndarray
    stops: np.ndarray
    representatives: np.ndarray
    n_frequencies: int

    def __post_init__(self):
        _positive_integer("n_frequencies", self.n_frequencies)
        for name in ("starts", "stops", "representatives"):
            values = np.asarray(getattr(self, name))
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError(
                    "frequency-bin indices must be integer vectors"
                )
            values = values.astype(np.int64, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if (
            self.starts.size == 0
            or self.starts.shape != self.stops.shape
            or self.starts.shape != self.representatives.shape
            or self.starts[0] != 0
            or self.stops[-1] != self.n_frequencies
            or np.any(self.starts[1:] != self.stops[:-1])
            or np.any(self.stops <= self.starts)
            or np.any(self.representatives < self.starts)
            or np.any(self.representatives >= self.stops)
        ):
            raise ValueError(
                "frequency bins must cover the native grid exactly once"
            )

    @property
    def counts(self):
        """Number of original Fourier bins in each aggregate."""
        return self.stops - self.starts


def _smooth_matrix(matrix, window):
    size = len(matrix)
    indices = np.arange(size)
    left = np.maximum(0, indices - window // 2)
    right = np.minimum(size, indices + window // 2 + 1)
    cumulative = np.concatenate(
        [np.zeros_like(matrix[:1]), matrix.cumsum(axis=0)]
    )
    smoothed = (cumulative[right] - cumulative[left]) / (right - left)[
        :, None, None
    ]
    return (smoothed + smoothed.conj().swapaxes(-1, -2)) / 2


def _relative_matrix_change(pilot, start, stop):
    midpoint = start + (stop - start - 1) // 2
    eigenvalues, vectors = np.linalg.eigh(pilot[midpoint])
    # Only the bin-selection metric is regularized. This never modifies
    # reference spectra, likelihood statistics or priors.
    floor = max(np.max(np.abs(eigenvalues)) * 1e-6, np.finfo(float).tiny)
    inverse_root = (
        vectors / np.sqrt(np.maximum(eigenvalues, floor))
    ) @ vectors.conj().T
    difference = pilot[start:stop] - pilot[midpoint]
    relative = inverse_root @ difference @ inverse_root
    return np.max(np.abs(np.linalg.eigvalsh(relative)))


def make_frequency_bins(frequencies, spectral_matrix=None, config=None):
    """Choose boundaries once, on an already selected physical-frequency grid.

    ``spectral_matrix`` is the unsmoothed empirical *residual* matrix after
    any reference whitening. Missing frequency ranges and supplied breaks
    separate both pilot smoothing and frequency aggregation.
    """
    frequency = np.asarray(frequencies, dtype=float)
    if (
        frequency.ndim != 1
        or frequency.size == 0
        or not np.isfinite(frequency).all()
        or not np.all(np.diff(frequency) > 0)
    ):
        raise ValueError(
            "frequencies must be a finite, nonempty, increasing vector"
        )
    size = frequency.size
    if config is not None and not isinstance(config, CoarseGrainConfig):
        raise TypeError("coarse_grain must be a CoarseGrainConfig or None")
    if config is None or config.max_bin_width == 1:
        indices = np.arange(size)
        return FrequencyBins(indices, indices + 1, indices, size)

    spacing = np.diff(frequency)
    gaps = (
        np.flatnonzero(spacing > 1.5 * np.min(spacing)) + 1
        if spacing.size
        else []
    )
    breaks = np.searchsorted(frequency, config.break_frequencies)
    boundaries = np.unique(np.r_[0, gaps, breaks, size]).astype(int)
    if config.method == "adaptive":
        matrix = np.asarray(spectral_matrix, dtype=np.complex128)
        if (
            matrix.ndim != 3
            or matrix.shape[0] != size
            or matrix.shape[1] < 1
            or matrix.shape[1] != matrix.shape[2]
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(
                "adaptive binning needs a finite (nfreq,p,p) residual matrix"
            )

    starts, stops = [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        pilot = (
            _smooth_matrix(matrix[left:right], config.pilot_window)
            if config.method == "adaptive"
            else None
        )
        position = left
        while position < right:
            limit = min(right, position + config.max_bin_width)
            stop = limit
            if pilot is not None:
                stop = position + 1
                while stop < limit:
                    change = _relative_matrix_change(
                        pilot, position - left, stop + 1 - left
                    )
                    if change > config.tolerance:
                        break
                    stop += 1
            starts.append(position)
            stops.append(stop)
            position = stop
    starts, stops = np.asarray(starts), np.asarray(stops)
    middle = starts + (stops - starts - 1) // 2
    return FrequencyBins(starts, stops, middle, size)


def pool_spectral_matrices(matrix, bins):
    """Sum unsmoothed complex statistics; do not average or reweight them."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 3 or matrix.shape[0] != bins.n_frequencies:
        raise ValueError(
            "spectral matrices do not match the frequency partition"
        )
    return np.add.reduceat(matrix, bins.starts, axis=0)
