[![Coverage Status](https://coveralls.io/repos/github/nz-gravity/sgvb_psd/badge.svg?branch=main)](https://coveralls.io/github/nz-gravity/sgvb_psd?branch=main)
![PyPI version](https://img.shields.io/pypi/v/sgvb-psd.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2409.13224-b31b1b.svg)](https://arxiv.org/abs/2409.13224)


# SGVB PSD Estimator

This repository contains the code for the papers

- ["Variational inference for correlated gravitational wave detector network noise" by Jianan Liu et al. 2024](https://arxiv.org/abs/2409.13224)
- "Variational Bayesian Inference for the Spectral Structure of LISA Noise" (arXiv link to be added)

Documentation is available at https://nz-gravity.github.io/sgvb_psd/

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

*Releasing to PyPI*

1. Manually change the version number in `pyproject.toml`  (has to be higher than previous)
1. Create a tagged commit with the version number
2. Push the tag to GitHub

```
git tag -a v2.0.0 -m "v2.0.0"
git push origin v2.0.0
```
