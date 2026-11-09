[![Coverage Status](https://coveralls.io/repos/github/nz-gravity/sgvb_psd/badge.svg?branch=main)](https://coveralls.io/github/nz-gravity/sgvb_psd?branch=main)
![PyPI version](https://img.shields.io/pypi/v/sgvb-psd.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2409.13224-b31b1b.svg)](https://arxiv.org/abs/2409.13224)


# SGVB PSD Estimator

SGVB estimates multivariate spectral density matrices using variational
inference. Research papers and related methods:

- [Variational inference for correlated gravitational wave detector network noise](https://arxiv.org/abs/2409.13224) (2024).
- [Variational Bayesian Inference for the Spectral Structure of LISA Noise](https://arxiv.org/abs/2608.22245) (2026).
- [Multivariate Bayesian P-spline estimation of spectral density matrices, with application to LISA TDI noise](https://arxiv.org/abs/2607.04833) (2026), the related [LogPSplinePSD](https://github.com/nz-gravity/LogPSplinePSD) method.

See [credits and papers](docs/papers.md) for authors, citations and data sources.

Documentation is available at https://nz-gravity.github.io/sgvb_psd/

## Installation

Install the latest release from PyPI:

```bash
pip install sgvb_psd
```

Install the JAX/NumPyro runtime from the source checkout
with `uv sync --locked` (Python 3.10+; validation uses 3.12). TensorFlow and
TensorFlow Probability are no longer runtime dependencies or install extras. Existing PyPI
releases may still use the previous runtime.

## Supported model

SGVB uses one JAX/NumPyro inference implementation for one or more channels;
there is no backend selector. The original modified-Cholesky joint model is
supported. Eigenbasis/factorized inference is unfinished and those requests
fail explicitly. Fit AR models externally and supply their spectra through
`reference_psd`, as demonstrated in the LVK notebook.

```python
from sgvb_psd.psd_estimator import PSDEstimator

joint = PSDEstimator(x=data)  # blocked Whittle likelihood
```

## Running inference

Install with `uv sync --locked --extra dev` from this checkout. For
univariate or multivariate input:

```python
estimator = PSDEstimator(
    x=data,                  # (n,) or (n, p)
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

The guide keeps its means at MAP by default. Set `optimize_mean=True`
to train guide means too. This changes the variational approximation while
keeping the target model.
The default approximation has known uncertainty limitations; stable
optimization and calibrated interval coverage remain under review. The
model preserves inherited prior scales and PSD conventions. In particular,
raw `psd_samples` omit the input standard deviation squared while returned
intervals include it. Investigation notes and experiments are kept locally.

## Reference spectrum inside inference

Supply a fixed reference `S_ref(f)` in original input units squared per
physical-frequency unit. A callable receives the actual analysis frequencies;
full matrices have shape `(nfreq, p, p)`:

```python
estimator = PSDEstimator(
    x=data,
    fs=sample_rate,
    reference_psd=lambda frequencies: reference_spectral_matrix(frequencies),
    seed=0,
)
psd_samples, pointwise_ci, uniform_ci = estimator.run(lr=0.003)
```

Arrays on the analysis grid also work, as does `(values, frequencies)` when
that table contains every analysis bin. Tables are selected without
interpolation/extrapolation so narrow reference features are preserved;
a callable is useful for known AR structure or LISA dip locations. Scalar,
shared-vector and per-channel `(nfreq, p)` diagonal references are supported.

Prepare very small physical reference values in NumPy float64 before they
are normalized to the JAX runtime precision. The reference must be finite,
Hermitian and strictly positive definite.
Exact spectral zeros require an explicit physical noise floor. Use the
package's **two-sided** density convention; divide a one-sided reference by
two at positive frequencies before passing it in.

The model whitens Fourier coefficients with the reference Cholesky factor,
fits the residual SGVB spectrum and reconstructs
`S = L_ref @ S_residual @ L_ref.conj().T`. A zero residual correction returns
`S_ref`. For one channel this is exactly
`log(S) = log(S_ref) + cosine_basis @ coefficients`. For a full matrix the
extension is Cholesky congruence; it is not a matrix-log sum. The fixed
whitening determinant is included in the likelihood. Reference uncertainty
is not inferred. With no reference, the original model is preserved.

`plot(true_psd=(values, frequencies))` remains a separate comparison overlay.

## Optional frequency coarse graining

Opt in to adaptive frequency aggregation for the residual Whittle likelihood:

```python
from sgvb_psd.inference import CoarseGrainConfig

estimator = PSDEstimator(
    x=data,
    fs=sample_rate,
    coarse_grain=CoarseGrainConfig(
        method="adaptive", tolerance=0.15, max_bin_width=32, pilot_window=31,
    ),
    parallel_rows=True,
)
psd_samples, pointwise_ci, uniform_ci = estimator.run(lr=0.003)
```

The default `coarse_grain=None` uses every retained Fourier frequency.
`method="fixed"` pools consecutive groups up to `max_bin_width`; the final
partial group is retained. `max_bin_width=1` gives the native-grid target.
`break_frequencies=(...)` supplies physical-frequency boundaries that bins
cannot cross. The pilot also respects gaps in the retained frequency grid.

Adaptive binning uses a smoothed full complex residual matrix only to choose
boundaries, then freezes the partition before optimization. Inference sums
the **unsmoothed** complex periodograms and uses the original observation
counts; aggregation does not reduce the effective data count. With a full
reference, whitening happens at each native bin first, and its determinant
is retained on that grid. Priors, guide family and frequency coordinates
are unchanged. Both joint and parallel-row execution are supported.

This approximates a residual spectrum that is constant within each aggregate.
The tolerance is a pilot criterion, not a posterior-error guarantee. Compare
with an unbinned fit before using it for a scientific analysis, particularly
near narrow features or rapid cross-spectral phase changes. PSD draws and
reference reconstruction still use the **original** frequency grid; their
cost does not shrink with the likelihood grid. The LISA notebook separately
documents finite-window resolution and its approximate bandwidth correction.

Inspect the chosen frequencies and counts using:

```python
analysis = estimator.inference_runner.data
likelihood_frequencies = analysis.likelihood_freq * estimator.fs
native_bins_per_aggregate = analysis.freq_bin_counts
fourier_observations_per_aggregate = analysis.counts
```

## Parallel Cholesky rows and quick examples

Set `parallel_rows=True` on `PSDEstimator` to fit the independent modified-
Cholesky rows with separate random keys in a single JAX-vmapped program.
The row targets sum to the original joint posterior, including a supplied
full reference matrix. Cross-spectra are retained through each row's regression coefficients.
The guide family is unchanged. This is batched
independent-row computation; CPU/GPU scheduling and speed are device dependent.
The original joint runner remains the default. Eigenbasis inference and the
collapsed conditional-theta method in LogPSplinePSD are distinct algorithms.

The five documentation notebooks cover a univariate simulation, a bivariate
simulation, ET, LISA (XYZ2 and AET together), and LVK strain with training-only
AR order selection. Each notebook includes executed plots. They use 1000 MAP
steps, 300 VI steps and 96 draws for a quick workflow demonstration. Data
downloads are cached and bounded; the LISA example uses a 13.8 MB excerpt
instead of the full 505 MB file. These
budgets are not convergence or uncertainty-calibration guarantees.

Install optional HDF5 support with `uv sync --locked --extra examples`
(or `pip install 'sgvb_psd[examples]'` after this API is released).

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


## Publishing

Merging ordinary changes into `main` does not publish a release. When ready,
choose the exact version and push a commit to `main` with this subject:

```bash
git commit --allow-empty -m "release: v2.1.0"
git push origin main
```

`2.1.0` is an example, not a scheduled release. The version is explicit;
`feat!`/`fix` commits never choose or trigger a release. Only the final pushed
commit subject is checked; a release marker in an older commit or a merge
message body does not publish. If using a squash merge, keep the final commit
subject `release: v2.1.0` (GitHub’s trailing `(#123)` is also accepted).

Alternatively, open **Actions → Release → Run workflow**, select `main`, and
enter `2.1.0`. This works without pushing directly to a protected branch.
For a preview version use, for example, `2.1.0rc1`; GitHub marks it as a
prerelease and ordinary pip installs continue to prefer stable versions.

The workflow validates the version against existing tags, tests the selected
commit, builds and checks one wheel and source distribution, then creates the
`v<version>` tag, publishes those artifacts to PyPI, and creates GitHub release
notes. Versions are derived from the tag by setuptools-scm; do not edit the
generated `_version.py`. Plain tag pushes do not start another publication.

A tag cannot be moved to different source code. Re-run failed workflow jobs
for a transient error; already uploaded PyPI files are skipped, never replaced.
A full retry of the same version must select the same commit (use the original
run if `main` has advanced). Changed code needs a new version. Releases are
serialized and an in-progress upload is not cancelled.

Publishing keeps the existing token-free PyPI Trusted Publisher identity:
`nz-gravity/sgvb_psd`, workflow `publish.yml`. No API-token secret is required;
the corresponding publisher must be configured on the PyPI project. The
workflow must reach `main` before its manual launch option becomes available.
