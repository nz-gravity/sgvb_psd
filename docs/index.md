# SGVB PSD

A python package for estimating the power spectral density (PSD)
of correlated multivariate detector noise using variational inference (VI).


## Installation

```bash
pip install sgvb_psd
```

## Examples

The five notebooks include saved plots, so you can read the results without
running them. Their Colab setup installs the JAX/NumPyro package from `main`;
local execution uses the project environment. Data downloads are cached locally.

| Example | Data and reference |
|---|---|
| [Univariate simulation](examples/var1d_example.ipynb) | AR(1) simulation with an approximate AR reference |
| [Bivariate simulation](examples/simulation_example.ipynb) | Correlated VAR(2), analytic truth and parallel Cholesky rows |
| [ET example](examples/ET_example.ipynb) | Short, anti-aliased sample of three-channel ET noise |
| [LISA example](examples/lisa_example.ipynb) | Shared public noise-simulation slice and full reference model in XYZ2 and AET coordinates |
| [LVK example](examples/LVK_example.ipynb) | Pre-event strain with AR order selection on a separate training segment |

Defaults use 1000 MAP steps, 300 VI steps and 96 posterior draws. These small
examples demonstrate the workflow; scientific analyses require convergence
and uncertainty-calibration checks. The LISA example uses a 13.8 MB excerpt
and supports an explicit synthetic mode or a user-supplied HDF5 file.

- [API documentation](api.rst)
- [Credits and papers](papers.md)

## Papers and credits

The [credits and papers page](papers.md) collects the SGVB and related
P-spline papers, citations, software links and data acknowledgements.
