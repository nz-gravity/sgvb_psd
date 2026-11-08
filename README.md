[![Coverage Status](https://coveralls.io/repos/github/nz-gravity/sgvb_psd/badge.svg?branch=main)](https://coveralls.io/github/nz-gravity/sgvb_psd?branch=main)
![PyPI version](https://img.shields.io/pypi/v/sgvb-psd.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2409.13224-b31b1b.svg)](https://arxiv.org/abs/2409.13224)


# SGVB PSD Estimator

This repository contains the code for the papers

- ["Variational inference for correlated gravitational wave detector network noise" by Jianan Liu et al. 2024](https://arxiv.org/abs/2409.13224)
- "Variational Bayesian Inference for the Spectral Structure of LISA Noise" (arXiv link to be added)

Documentation is available at https://nz-gravity.github.io/sgvb_psd/

## Installation

Install the latest release from PyPI:

```bash
pip install sgvb_psd
```

On this migration branch, install the new JAX-only runtime from the checkout
with `pip install -e .` (Python 3.10+; validation uses 3.12). TensorFlow and
TensorFlow Probability are no longer default dependencies. Existing PyPI
releases may still use the previous runtime.

## Inference modes

The original non-eigenbasis joint SGVB model now runs in JAX by default.
Eigenbasis modes are retained in the optional legacy comparison path;
install `pip install -e '.[legacy]'` and explicitly select TensorFlow until
their JAX ports are validated.

```python
from sgvb_psd.psd_estimator import PSDEstimator

# Original joint SGVB with the blocked Whittle likelihood (default)
joint = PSDEstimator(x=data)

# Joint SGVB with the blocked eigenbasis likelihood
eigen_joint = PSDEstimator(
    x=data,
    backend="tensorflow",
    use_eigenbasis=True,
    posterior_mode="joint",
)

# Factorized (componentwise) SGVB with the blocked eigenbasis likelihood
eigen_factorized = PSDEstimator(
    x=data,
    backend="tensorflow",
    use_eigenbasis=True,
    posterior_mode="factorized",
)
```

## JAX runtime and experimental migration

Install with `pip install -e '.[dev]'` from this checkout (`[jax]` remains a
compatibility alias). The original joint model runs in JAX/NumPyro for
univariate or multivariate input:

```python
estimator = PSDEstimator(
    x=data,                  # (n,) or (n, p)
    backend="jax",
    fs=sample_rate,
    nchunks=4,
    N_theta=15,
    ntrain_map=1000,
    n_elbo_maximisation_steps=500,
    seed=0,
    elbo_samples=4,
)
psd_samples, pointwise_ci, uniform_ci = estimator.run(lr=0.003)
```

The default runtime imports no TensorFlow; the optional legacy path retains
the eigenbasis modes. The JAX
guide keeps its means at MAP by default, matching the current TensorFlow
implementation. Set `optimize_mean=True` to train JAX guide means too;
this changes the variational approximation while keeping the target model.
The diagonal guide has known calibration limitations in a small conditional
prior-predictive control; full-model uncertainty calibration remains open.
The JAX backend preserves the current model and PSD conventions; it does not yet
include AR prewhitening or eigenbasis inference. See the
[migration and parity report](validation/JAX_MIGRATION.md) for measured
checks, conventions, known issues, and the remaining consolidation work.
The [guide comparison](validation/GUIDE_COMPARISON.md) distinguishes the
custom guide family from its inference engine and compares richer options.

## Development

Install in editable mode with dev dependencies
```
pip install -e ".[dev]"
pre-commit install
```

Ensure unit tests are passing locally and on the CI!
```
pytest tests/
```
