# Initial JAX migration and parity control

This change adds a usable JAX/NumPyro implementation of the original joint
SGVB model for 1D and multivariate input. It is an incremental migration in
`nz-gravity/sgvb_psd`; it does not combine two duplicate source trees.
TensorFlow remains the default, and both existing eigenbasis modes remain
available through TensorFlow.

## Revisions and environment

- Destination base: `8b0f1f59241b0e5918c962212a4516123d8c8131` (`main` at inspection).
- Work branch: `feature/jax-sgvb`; implementation and evidence are in this branch.
- Related source inspected: `jianan-joseph-liu/cleared_unisgvb`, revision
  `ee515546a389e935b01d1bb1fac26240a52563b0`.
- Run: 2026-11-08 UTC / 2026-11-09 NZDT, host `UOA484808`, macOS 27.0 arm64,
  CPU only (`TFRT_CPU_0`). Python 3.12.16, TensorFlow 2.20.0, TFP 0.25.0,
  tf-keras 2.20.1, JAX/jaxlib 0.6.2, NumPyro 0.19.0, NumPy 2.5.4.
- Dependencies and caches are checkout-local (`.venv`, `.uv-cache`, `.mpl-cache`).
  The existing managed Python interpreter was reused. No system Python,
  network, or security configuration was changed.
- [environment.lock](environment.lock) records installed packages;
  [jax-constraints.txt](jax-constraints.txt) pins the scientific stack for CI.

No pre-existing SGVB checkout or applicable `AGENTS.md`/repository skill was
found in the inspected research/task locations. A fresh clone was created;
no existing research checkout was overwritten. The remote
`univariate-SGVB-PSD` branch was inspected: it still rejects one-channel
input. Its end-to-end tests contain VAR2D tests, but no univariate pytest
was found there or in the fetched main test history. The old `7aedc5a`
"add JAX-jit" change used TensorFlow compilation. Local relevant memory
summaries and filenames did not locate the separate earlier JAX prototype.

## Implementation

Select `PSDEstimator(..., backend="jax", seed=0)` for the JAX implementation.
The input may be `(n,)` or `(n, p)`. The original default API and TensorFlow
modes are preserved. JAX eigenbasis/factorized selection raises an explicit
error; it never silently changes the requested method.

Shared NumPy preprocessing and existing interval summaries were extracted
without changing their formulas. The new backend provides:

- Summed-periodogram blocked Whittle likelihood, modified Cholesky PSD
  reconstruction, and the existing shrinkage prior in log coordinates.
- One log density per variational draw, with block/frequency/channel axes
  reduced internally. This avoids the reference scalar likelihood being
  broadcast across draws when combined with the vector prior.
- Jitted MAP optimization followed by NumPyro mean-field SGVB, including
  empty theta blocks for the one-channel case. The NumPyro improper base
  measure adds no second prior; the factor supplies the full target.
- Independent Normal guide parameters, softplus-constrained positive
  scales initialized at `1e-4`, matching the reference guide family.
- Per-runner PRNG state. Identical seeds reproduce a fixed-learning-rate run; subsequent
  posterior sampling advances keys. No global JAX dtype configuration is
  changed. Model arrays use float32/complex64, matching the reference.
- `elbo_samples` controls NumPyro Monte Carlo particles. Both one- and
  four-particle inference are tested. Public posterior sampling and a small
  learning-rate search are covered.
- PSD plotting retains a two-dimensional axes array even for one channel.

`import sgvb_psd.jax_backend` works without importing TensorFlow. The public
`PSDEstimator` still imports the retained TensorFlow implementation and
TensorFlow remains a base package dependency during this migration.

## Checks and numerical evidence

Commands were run from the repository root:

```bash
UV_CACHE_DIR=.uv-cache uv venv --python 3.12 .venv
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python \
  -e '.[dev,jax]' -c validation/jax-constraints.txt
export MPLBACKEND=Agg MPLCONFIGDIR=.mpl-cache
.venv/bin/python -m pytest -m 'not slow' -q
.venv/bin/python -m pytest tests/test_all_modes.py -q
.venv/bin/python -m pytest -q
.venv/bin/python validation/run_jax_parity.py
```

The initial unchanged baseline passed **20 tests**, with the slow test
deselected ([baseline.log](baseline.log)). The final full suite passed
**54 tests** (33 new JAX tests plus the 21 existing tests, including the slow
all-mode comparison) in 92.84 s. The full run and separate slow comparison are recorded in
[final-tests.log](final-tests.log) and [all-modes.log](all-modes.log).
The added one-channel plotting check initially exposed the scalar axes
return and empty off-diagonal limit reduction; both were fixed and retested
([initial failure](jax-plot-initial.log)). No failing assertion was suppressed.
Existing warnings include single-draw
uniform-band division by zero and plotting/complex-cast warnings.

New coverage checks 1D/2D/3D models, one/three data blocks, one/four
variational draws, TF likelihood/prior/gradient/PSD parity, Hermitian
positive PSDs, independent draws, analytic white-noise normalization,
log-coordinate Jacobians against SciPy, the NumPyro target against the
model target, reproducibility, invalid inputs, and bounded inference.
Because TF `AnalysisData` rejects 1D, the 1D TF oracle restricts its
MIT-licensed BayesianModel data to one channel; independent scalar-prior
and white-noise formulas provide an additional check.

The seeded fixed-parameter VAR2D control (four draws, two blocks) measured:

| Quantity | Result |
|---|---:|
| TF and JAX summed log target | both -2120.20068359375 |
| Maximum absolute gradient difference | 0.000076294 |
| Maximum absolute PSD difference | 0.000000044704 |
| MAP PSD relative L2 difference, JAX versus TF | 0.00000015169 |

The [experiment script](run_jax_parity.py),
[machine-readable results](jax-results.json), and
[fit-control plot](jax-fit-control.svg) are included. The bounded fits used
512 samples, two blocks, six cosine terms, 100 MAP steps, 50 VI steps,
64 posterior draws, and one ELBO particle in each backend. Both JAX fits
had finite ELBO values and interval arrays. The AR(1) fit improved its MAP
target from -268.38 to -198.67; VAR2D improved from -517.27 to -299.73.

The short VAR2D fitted median differs from TensorFlow by about **29.9%**
relative L2 on channel 0 despite nearly identical MAP results. Different
random-number streams and optimizer implementations remain; the fit has
only 50 VI updates. This is **not** evidence of equivalent fitted
posteriors or calibrated uncertainty. The short AR(1) and VAR2D channel-0
median errors against truth were about 18.2% and 27.8%, respectively. A
preliminary four-particle JAX run was also finite and showed a similar
short-fit discrepancy; the recorded paired timing uses equal particle counts.

## Timing boundaries

All timings synchronize device results. First-call timing includes graph
tracing/compilation and execution, but excludes imports. Warm target
evaluation is the median of 30 identical-shape calls. These are small CPU
controls, not general speedup claims.

| Workload | JAX | TensorFlow |
|---|---:|---:|
| First compiled summed-target call | 86.7 ms | 194.2 ms |
| Warm summed-target evaluation | 0.0700 ms | 0.2891 ms |
| Short VAR2D fit including compile/postprocessing | 1.73 s | 3.07 s |

For the JAX VAR2D fit, MAP including compile took 0.196 s and VI including
compile took 0.956 s. Those training timings do not isolate warm training.
Postprocessing remains NumPy-based and has not been optimized.

## Scientific conventions and unresolved work

The migration deliberately preserves the reference conventions:

1. FFT normalization is `fft / sqrt(block_length)`, with no per-block
   demeaning, standardization, taper, or window applied inside preprocessing.
   Remainder samples are truncated. The estimator uses one global input
   standard deviation and does not subtract the mean.
2. The basis uses frequency in cycles/sample; output frequency is rescaled
   by `fs`. Bounds are `fmin <= f < fmax`. DC and even-length Nyquist are
   excluded. The existing odd-length code also drops its highest positive
   bin; tests freeze that behavior rather than quietly correcting it.
3. PSD output is a **two-sided density evaluated at positive frequencies**,
   with a `/fs` conversion. A conventional one-sided interior PSD needs
   an additional factor of two. `Nbw` divides the likelihood; this change
   introduces no new window correction.
4. The bounded quantity named `Sigma2i_diag` in TF is supplied as a Normal
   **standard deviation**, not its variance. Tests preserve this behavior.
   Whether the intended model needs a square root requires author review;
   no scientific correction is included here. Exp-transform Jacobians for
   local/global HalfCauchy scales are included exactly once.
5. The existing `compute_psd` API applies `psd_scaling**2` to interval arrays
   but not to `psd_all`. Both implementations preserve it for parity. A
   future explicit units/API fix should scale samples and summaries
   consistently. Single-draw uniform intervals can contain NaNs, and the
   inherited uniform-band algorithm has not been statistically validated
   by this migration.

The related univariate repository has no `LICENSE` file in its complete
tree and GitHub reports `license: null` at the inspected revision. **No code
or data from it was copied.** AR fitting, residual normalization,
whitening/recolouring, and GW150914 comparison are therefore not ported or
validated. Permission/provenance is needed before importing that code into
this MIT package. This does not block the independently ported MIT model.

Next steps are a matched-draw ELBO/optimizer convergence study with longer
VI runs, simulation-based uncertainty checks, explicit decisions on prior
scale and PSD units, then provenance-cleared AR workflow integration and
the JAX eigenbasis/factorized port. GW150914, GPU behavior, long-run
calibration, and the remote Linux CI result are not claimed by the local
checks. Do not merge or retire TensorFlow on the strength of these controls.
