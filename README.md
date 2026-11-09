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
with `uv sync --locked` (Python 3.10+; validation uses 3.12). TensorFlow and
TensorFlow Probability are no longer runtime dependencies or install extras. Existing PyPI
releases may still use the previous runtime.

## Inference modes

The original non-eigenbasis joint SGVB model runs in JAX for one or more
channels. Only JAX/NumPyro is supported in the package, tests and CI.
Eigenbasis joint/factorized JAX inference is unfinished; those requests fail
explicitly. AR whitening/recolouring is not integrated. The old TensorFlow
backend and reference support are removed; historical results remain in Git
history. See [migration scope](validation/JAX_MIGRATION.md).

```python
from sgvb_psd.psd_estimator import PSDEstimator

joint = PSDEstimator(x=data)  # blocked Whittle likelihood
```

## JAX runtime and experimental migration

Install with `uv sync --locked --extra dev` from this checkout (`[jax]` remains a
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

The installed runtime imports no TensorFlow. The JAX
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
The [upgrade benchmark](validation/UPGRADE_BENCHMARK.md) reports cold and
warmed costs alongside posterior agreement and conditional accuracy checks;
full-model calibration is not yet established.
The [VAR2 pointwise coverage study](validation/VAR2_COVERAGE.md) exercises
the full production default guide under a frozen fixed-truth simulation
protocol; it is separate from prior-predictive calibration.

## Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
create the editable development environment from the lockfile:

```bash
uv sync --locked --extra dev
uv run --no-sync pre-commit install
```

Run the same non-slow tests as CI:

```bash
uv run --no-sync pytest -m "not slow"
```

After dependency changes, run `uv lock` and commit the updated `uv.lock`.

Validation scripts and concise summaries are in [validation/README.md](validation/README.md).
Experiment arrays, per-run outputs, plots and logs are ignored local files;
do not commit them. Earlier blobs remain in Git history without a force-push.
