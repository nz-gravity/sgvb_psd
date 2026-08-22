[![Coverage Status](https://coveralls.io/repos/github/nz-gravity/sgvb_psd/badge.svg?branch=main)](https://coveralls.io/github/nz-gravity/sgvb_psd?branch=main)
![PyPI version](https://img.shields.io/pypi/v/sgvb-psd.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2409.13224-b31b1b.svg)](https://arxiv.org/abs/2409.13224)


# SGVB PSD Estimator

This repository contains the code for the paper 
["Variational inference for correlated gravitational wave detector network noise" by Jianan Liu at al. 2024](https://arxiv.org/abs/2409.13224)

Documentation is available at https://nz-gravity.github.io/sgvb_psd/

## Inference modes

Version 2 keeps the original non-eigenbasis joint SGVB method as the
default and adds joint and factorized eigenbasis methods.

```python
from sgvb_psd.psd_estimator import PSDEstimator

# Original non-eigenbasis joint SGVB
legacy = PSDEstimator(x=data)

# Eigenbasis joint SGVB
eigen_joint = PSDEstimator(
    x=data,
    use_eigenbasis=True,
    posterior_mode="joint",
)

# Eigenbasis factorized SGVB
eigen_factorized = PSDEstimator(
    x=data,
    use_eigenbasis=True,
    posterior_mode="factorized",
)
```

All three methods accept `Nbw`, an effective window-bandwidth correction
computed before constructing the estimator. Its default value is `1.0`.
The eigenbasis methods additionally use `fmin_idx_extension` and
`fmax_idx_extension`, whose defaults are `0` and `32` frequency bins.



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
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```
