[![Coverage Status](https://coveralls.io/repos/github/nz-gravity/sgvb_psd/badge.svg?branch=main)](https://coveralls.io/github/nz-gravity/sgvb_psd?branch=main)
![PyPI version](https://img.shields.io/pypi/v/sgvb-psd.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2409.13224-b31b1b.svg)](https://arxiv.org/abs/2409.13224)


# SGVB PSD Estimator

This repository contains the code for the paper 
["Variational inference for correlated gravitational wave detector network noise" by Jianan Liu at al. 2024](https://arxiv.org/abs/2409.13224)

Documentation is available at https://nz-gravity.github.io/sgvb_psd/



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
