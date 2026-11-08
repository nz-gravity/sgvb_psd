"""Shared deterministic AR(1) and existing VAR2D simulation controls."""

import numpy as np
from sgvb_psd.utils.sim_varma import SimVARMA


def cases():
    simulation = SimVARMA(
        n_samples=512,
        var_coeffs=np.array([[[0.5, 0], [0, -0.3]], [[0, 0], [0, -0.5]]]),
        vma_coeffs=np.eye(2)[None],
        sigma=np.array([[1, 0.9], [0.9, 1]]),
        seed=0,
    )
    eps = np.random.default_rng(0).normal(size=768)
    ar = np.zeros_like(eps)
    for i in range(1, len(ar)):
        ar[i] = 0.6 * ar[i - 1] + eps[i]
    return [
        ("ar1", ar[256:, None], 1.0),
        ("var2", simulation.data, simulation.fs),
    ]
