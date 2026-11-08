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

## Inference modes

Version 2.0.0 keeps the original non-eigenbasis joint SGVB method as the
default and adds joint and factorized eigenbasis methods.

```python
from sgvb_psd.psd_estimator import PSDEstimator

# Original joint SGVB with the blocked Whittle likelihood (default)
legacy = PSDEstimator(x=data)

# Joint SGVB with the blocked eigenbasis likelihood
eigen_joint = PSDEstimator(
    x=data,
    use_eigenbasis=True,
    posterior_mode="joint",
)

# Factorized (componentwise) SGVB with the blocked eigenbasis likelihood
eigen_factorized = PSDEstimator(
    x=data,
    use_eigenbasis=True,
    posterior_mode="factorized",
)
```

## Experimental JAX backend

Install with `pip install 'sgvb_psd[jax]'` (or `pip install -e '.[dev,jax]'`
from a checkout). The original joint model can now run in JAX/NumPyro for
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

TensorFlow remains the default and retains the eigenbasis modes. The JAX
backend preserves the current model and PSD conventions; it does not yet
include AR prewhitening or eigenbasis inference. See the
[migration and parity report](validation/JAX_MIGRATION.md) for measured
checks, conventions, known issues, and the remaining consolidation work.

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
