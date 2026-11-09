"""Fast, fixed-order autoregressive baselines for residual SGVB fits.

These are plug-in references, not Bayesian estimates of the AR parameters.
No order search, TensorFlow, or time-domain whitening is performed here.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy import fft, linalg, signal


def _sample_rate(fs):
    if isinstance(fs, (bool, np.bool_)) or not isinstance(fs, Real):
        raise ValueError("fs must be finite and positive")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    return float(fs)


@dataclass(frozen=True, slots=True)
class ARReference:
    """Callable full spectral matrix in physical two-sided density units.

    Usually constructed by :func:`fit_ar_reference`. Coefficients have shape
    ``(order, p, p)`` and describe standardized channels:
    ``z[t] = sum(A[k-1] @ z[t-k]) + innovation[t]``.
    ``channel_scale`` restores the input units. Arrays are copied/read-only.
    """

    fs: float
    coefficients: np.ndarray
    innovation_covariance: np.ndarray
    channel_scale: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "fs", _sample_rate(self.fs))
        for name in ("coefficients", "innovation_covariance", "channel_scale"):
            value = np.asarray(getattr(self, name))
            if not np.isrealobj(value) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite and real")
            value = value.astype(float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        a, covariance, scale = (
            self.coefficients,
            self.innovation_covariance,
            self.channel_scale,
        )
        if (
            scale.ndim != 1
            or not len(scale)
            or np.any(scale <= 0)
            or a.ndim != 3
            or a.shape[0] < 1
            or a.shape[1:] != (len(scale), len(scale))
            or covariance.shape != (len(scale), len(scale))
        ):
            raise ValueError("AR arrays have inconsistent shapes or scales")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0):
            raise ValueError("innovation covariance must be symmetric")
        try:
            linalg.cholesky(covariance)
        except linalg.LinAlgError as error:
            raise ValueError(
                "innovation covariance must be positive definite"
            ) from error

    @property
    def order(self):
        return self.coefficients.shape[0]

    @property
    def n_channels(self):
        return len(self.channel_scale)

    def __call__(self, frequencies):
        """Evaluate S(f) on physical frequencies in [0, fs/2], without doubling."""
        frequencies = np.asarray(frequencies)
        if (
            frequencies.ndim != 1
            or not np.isrealobj(frequencies)
            or not np.all(np.isfinite(frequencies))
            or np.any(frequencies < 0)
            or np.any(frequencies > self.fs / 2)
        ):
            raise ValueError(
                "frequencies must be a finite vector in [0, fs/2]"
            )
        phase = np.exp(
            -2j
            * np.pi
            * frequencies[:, None]
            * np.arange(1, self.order + 1)
            / self.fs
        )
        polynomial = np.eye(self.n_channels) - np.einsum(
            "fl,lij->fij", phase, self.coefficients
        )
        transfer = np.linalg.inv(polynomial)
        spectrum = (
            transfer
            @ self.innovation_covariance
            @ transfer.conj().swapaxes(-1, -2)
        ) / self.fs
        spectrum *= (
            self.channel_scale[None, :, None]
            * self.channel_scale[None, None, :]
        )
        if not np.all(np.isfinite(spectrum)) or np.any(
            np.diagonal(spectrum, axis1=-2, axis2=-1).real <= 0
        ):
            raise ValueError("AR spectrum exceeds the supported numeric range")
        return (spectrum + spectrum.conj().swapaxes(-1, -2)) / 2


def _scalar_yule_walker(autocovariance):
    """Levinson-Durbin, checking reflection coefficients without root finding."""
    correlation = autocovariance / autocovariance[0]
    coefficients = np.empty(0)
    error = 1.0
    for lag in range(1, len(correlation)):
        reflection = (
            correlation[lag] - np.dot(coefficients, correlation[1:lag][::-1])
        ) / error
        if not np.isfinite(reflection) or abs(reflection) >= 1:
            raise ValueError(
                "unstable AR recursion; reduce order or increase ridge"
            )
        coefficients = np.r_[
            coefficients - reflection * coefficients[::-1], reflection
        ]
        error *= 1 - reflection**2
    return coefficients, error * autocovariance[0]


def fit_ar_reference(x, fs, order=256, *, full_matrix=False, ridge=1e-8):
    """Fit one fixed-order AR/VAR baseline; return a callable S(f).

    ``x`` is real training data with shape ``(n,)`` or ``(n, p)``. Centering
    and internal scaling avoid strain-unit underflow. Biased (1/n)
    autocovariances and a ``ridge`` addition to standardized lag-zero
    variances stabilize the Yule-Walker equations. No order optimization or
    polynomial root search is needed. Independent channels use O(order²)
    Levinson-Durbin fits. ``full_matrix=True`` instead solves a dense
    (p*order)-dimensional block system and retains cross-spectra; use modest
    orders for a few channels.

    Taper/filter the training data explicitly, consistently with the data
    supplied to SGVB. This helper does not apply a window or a one-sided
    factor of two. Pass the result directly as ``reference_psd``; SGVB handles
    whitening and reconstruction once. Training on a disjoint segment avoids
    fitting the reference twice to the same Fourier data. Reference-parameter
    uncertainty remains fixed, including for the full matrix.
    """
    fs = _sample_rate(fs)
    x = np.asarray(x)
    if not np.isrealobj(x):
        raise ValueError("x must be finite real training data")
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if (
        x.ndim != 2
        or x.shape[1] < 1
        or len(x) < 2
        or not np.all(np.isfinite(x))
    ):
        raise ValueError("x must be finite real training data of shape (n, p)")
    if (
        isinstance(order, (bool, np.bool_))
        or not isinstance(order, (int, np.integer))
        or not 1 <= order < len(x)
    ):
        raise ValueError("order must be an integer between 1 and n-1")
    if not isinstance(full_matrix, (bool, np.bool_)):
        raise ValueError("full_matrix must be a boolean")
    if (
        isinstance(ridge, (bool, np.bool_))
        or not isinstance(ridge, Real)
        or not np.isfinite(ridge)
        or ridge < 0
    ):
        raise ValueError("ridge must be finite and nonnegative")
    scale = np.std(x, axis=0)
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError(
            "every training channel must have positive finite variance"
        )
    z = (x - x.mean(axis=0)) / scale
    p = z.shape[1]
    coefficients = np.zeros((order, p, p))
    covariance = np.zeros((p, p))
    if not full_matrix or p == 1:
        for channel in range(p):
            acov = signal.correlate(
                z[:, channel], z[:, channel], mode="full", method="fft"
            )[len(z) - 1 : len(z) + order] / len(z)
            acov[0] += ridge
            coefficients[:, channel, channel], covariance[channel, channel] = (
                _scalar_yule_walker(acov)
            )
    else:
        nfft = fft.next_fast_len(2 * len(z) - 1)
        transformed = fft.rfft(z, n=nfft, axis=0)
        acov = fft.irfft(
            transformed[:, :, None] * transformed[:, None, :].conj(),
            n=nfft,
            axis=0,
        )[: order + 1] / len(z)
        acov[0] += ridge * np.eye(p)
        lag = np.arange(order)[None, :] - np.arange(order)[:, None]
        blocks = np.where(
            (lag >= 0)[..., None, None],
            acov[abs(lag)],
            acov[abs(lag)].swapaxes(-1, -2),
        )
        system = blocks.transpose(0, 2, 1, 3).reshape(order * p, order * p)
        cross = acov[1:].swapaxes(-1, -2).reshape(order * p, p)
        try:
            solution = linalg.solve(system, cross, assume_a="pos")
        except linalg.LinAlgError as error:
            raise ValueError(
                "singular VAR fit; reduce order or increase ridge"
            ) from error
        coefficients = solution.reshape(order, p, p).swapaxes(-1, -2)
        covariance = acov[0] - cross.T @ solution
        covariance = (covariance + covariance.T) / 2
    return ARReference(fs, coefficients, covariance, scale)
