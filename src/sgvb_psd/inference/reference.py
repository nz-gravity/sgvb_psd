"""Fixed reference spectra evaluated on the analysis Fourier grid."""

import numpy as np


def reference_factor(reference_psd, frequencies, channels, fs, data_scale=1.0):
    """Return a physical reference and its normalized Fourier covariance factor.

    Arrays use physical two-sided density units. A callable receives physical
    frequencies. A (values, frequencies) table must contain the exact analysis
    bins: select matching bins, without interpolation, extrapolation or flooring.
    A shared scalar/vector or per-channel array specifies a diagonal reference.
    Full Hermitian matrices specify a correlated reference.
    """
    if not np.isfinite(data_scale) or data_scale <= 0:
        raise ValueError("data_scale must be finite and positive")
    frequencies = np.asarray(frequencies, dtype=float)
    if callable(reference_psd):
        values = reference_psd(frequencies.copy())
    elif isinstance(reference_psd, tuple) and len(reference_psd) == 2:
        values, grid = reference_psd
        grid = np.asarray(grid)
        if (
            grid.ndim != 1
            or not len(grid)
            or not np.isrealobj(grid)
            or not np.all(np.isfinite(grid))
            or np.any(np.diff(grid) <= 0)
        ):
            raise ValueError(
                "reference frequencies must be finite and strictly increasing"
            )
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != len(grid):
            raise ValueError(
                "reference values and frequencies must have matching lengths"
            )
        right = np.clip(np.searchsorted(grid, frequencies), 0, len(grid) - 1)
        left = np.maximum(right - 1, 0)
        indices = np.where(
            abs(grid[left] - frequencies) < abs(grid[right] - frequencies),
            left,
            right,
        )
        if not np.allclose(
            grid[indices], frequencies, rtol=1e-10, atol=1e-12 * fs
        ):
            raise ValueError(
                "reference table must contain the analysis frequencies; use a callable to evaluate another grid"
            )
        values = values[indices]
    else:
        values = reference_psd
    try:
        values = np.asarray(values, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "reference_psd must contain numeric spectral densities"
        ) from error
    n = len(frequencies)
    if values.ndim == 0:
        values = np.full((n, channels), values)
    elif values.shape == (n,):
        values = np.broadcast_to(values[:, None], (n, channels))
    if values.shape == (n, channels):
        matrix = np.zeros((n, channels, channels), dtype=complex)
        index = np.arange(channels)
        matrix[:, index, index] = values
        values = matrix
    if values.shape != (n, channels, channels):
        raise ValueError(
            "reference_psd must be scalar, (nfreq,), (nfreq, p), or (nfreq, p, p)"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("reference_psd must be finite")
    magnitude = np.max(abs(values), axis=(-2, -1))
    tolerance = 1e-10 * magnitude + np.finfo(float).tiny
    if np.any(
        np.max(abs(values - values.conj().swapaxes(-1, -2)), axis=(-2, -1))
        > tolerance
    ):
        raise ValueError(
            "reference_psd must be Hermitian with real diagonal PSDs"
        )
    # Cholesky is computed in NumPy float64 before normalization/casting, so
    # physical densities such as 1e-40 need not underflow in the JAX runtime.
    try:
        physical_factor = np.linalg.cholesky(values)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "reference_psd must be strictly positive definite; apply any physical noise floor explicitly"
        ) from error
    normalized_factor = physical_factor * (np.sqrt(fs) / data_scale)
    factor32 = normalized_factor.astype(np.complex64)
    if not np.all(np.isfinite(factor32)) or np.any(
        np.diagonal(factor32, axis1=-2, axis2=-1).real <= 0
    ):
        raise ValueError(
            "normalized reference factor is outside float32 range"
        )
    logdet = 2 * np.log(
        np.diagonal(normalized_factor, axis1=-2, axis2=-1).real
    ).sum(-1)
    return values.copy(), factor32, logdet.astype(np.float32)
