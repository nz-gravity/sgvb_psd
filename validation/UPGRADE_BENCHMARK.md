# JAX-only upgrade and measured costs

The installed package now uses JAX/NumPyro only. TensorFlow, TFP and
tf-keras are neither runtime dependencies nor install extras. Their MIT
historical source is archived under [reference](reference/README.md), outside
`src` and the wheel. Original-joint 1D/multivariate inference is usable;
eigenbasis joint/factorized JAX modes remain unported and fail explicitly.
This is a breaking upgrade, not complete replacement of every scientific mode.

JAX is faster in the measured CPU workloads. Full-model comparisons do
**not** establish equal posterior accuracy/calibration. A separate
conditional control achieves equivalent quadrature-based accuracy in both
engines and is faster in JAX; its scope is deliberately limited.

## Implementation and environment

Destination base: `8b0f1f59241b0e5918c962212a4516123d8c8131`. Previous guide
investigation: `40a46f571f6001e13435df6f075d11853fb393ba`. This report
accompanies the subsequent runtime/archive/cache changes on `feature/jax-sgvb`.
Measurements: 2026-11-08 UTC / 2026-11-09 NZDT, UOA484808, macOS arm64 CPU,
Python 3.12.16, JAX/jaxlib 0.6.2, NumPyro 0.19.0 versus TensorFlow 2.20.0,
TFP 0.25.0 and tf-keras 2.20.1. Exact packages:
[jax-environment.lock](upgrade-benchmark/jax-environment.lock),
[tf-environment.lock](upgrade-benchmark/tf-environment.lock).

The environments are separate: `.venv-jaxonly` has no TF stack;
`.venv-tfref` has no JAX. Workers confirmed no other-framework imports.
Dependencies/caches are project-local; no system configuration was changed.
Benchmarks ran sequentially, alternating backend order in the second fresh
process trial, before final test suites.

The JAX runner now caches compiled MAP and SVI functions per prepared
runner/data configuration. Learning rates and current MAP locations are
dynamic arguments, preventing stale captured guide means. Regression tests
compare updated means/rates against fresh runners. There is no global
dataset cache. Changing data requires a new runner and is not the warm
scenario below. The estimator's learning-rate search can reuse its runner.

## Full original-joint pipeline

[benchmark_upgrade.py](benchmark_upgrade.py) runs MAP 1,000 steps at .003,
VI 5,000 steps at .05, one ELBO particle, float32/complex64 and 128 public
posterior draws, including interval postprocessing. Both use the inherited
fixed-MAP-mean diagonal family, shrinkage prior and FFT conventions.
Initialization, simulation data and step counts match; backend Adam
epsilon/bias conventions and random streams differ. No likelihood, prior,
window or normalization correction is mixed into this upgrade.

Each case has two seeds/fresh processes per backend, each followed by two
repeated fits. Repeats reset parameters and RNG; they are timing repeats,
not additional independent posterior fits. Cold is parent wall time from
process launch to the first result, including imports, preparation,
compilation, inference and postprocessing. Warm reuses the prepared runner
and kernels and includes fitting/sampling/summaries. Independent accuracy
evaluation is excluded. Historical TF functions are recreated by its
original methods; this compares the existing implementation, not an
optimized TF/XLA ceiling. No GPU result is claimed.

Medians, seconds (two cold observations and four warm observations each):

| Case | JAX cold | TF cold | TF/JAX | JAX warm | TF warm | TF/JAX |
|---|---:|---:|---:|---:|---:|---:|
| AR1, n=512, 1 channel, 2 blocks, 6 basis | 2.343 | 6.889 | 2.94× | .345 | 3.588 | 10.41× |
| VAR2, n=512, 2 channels, 2 blocks, 6 basis | 3.417 | 6.995 | 2.05× | .791 | 3.694 | 4.67× |
| VAR2, n=8192, 2 channels, 8 blocks, 15 basis | 5.367 | 8.138 | 1.52× | 2.733 | 4.839 | 1.77× |

The historical public estimator rejects 1D. Its 1D row uses a lower-level
model adapter, the empty-theta construction used by parity tests; this is
not a benchmark of a previously supported public 1D API.
All observations, phase costs and exit status are preserved in
[summary.json](upgrade-benchmark/summary.json), per-worker JSON and
[processes.json](upgrade-benchmark/processes.json). All 36 fits returned
finite outputs. The exploratory pilot's 28-second TF startup was not
representative of subsequent trials; the table uses the complete full run.

### Posterior quality alongside timing

An independent common NumPy/SciPy evaluator uses 2,048 normal draws per fit
for negative ELBO and PSD functionals. It evaluates the same summed
Whittle target/prior/Jacobians in float64 using float32 Fourier statistics
and basis. PSD functionals explicitly include input std²/fs: each channel's
PSD at .05/.2/.4 cycles/sample, positive-bin integrated power, and VAR2
coherence. Quantiles are 5/50/95%. Common noise pairs guide evaluations;
MC SE is evaluation uncertainty, not optimization/simulation uncertainty.
Raw losses/features/MAP arrays are retained as NPZ artifacts.

| Case | Largest functional median relative difference, seeds 0/1 | JAX/TF 90% width range, all functionals/seeds | Paired negative-ELBO difference JAX−TF, seed 0 / seed 1 (MC SE) |
|---|---:|---:|---:|
| AR1 small | .175% / .268% | .884–1.154 | +.833 (.169) / −1.708 (.190) |
| VAR2 small | .344% / .413% | .907–1.290 | −1.442 (.151) / −.271 (.220) |
| VAR2 large | .269% / .457% | .861–1.194 | −.354 (.367) / −1.073 (.402) |

Medians agree closely, but width differences reach 29%, and some ELBO
differences exceed evaluation error. These bounded runs do not prove
convergence or equal individual posterior accuracy. MAP values also differ
with backend optimizer conventions. Earlier matched-noise objective and
gradient controls in [VI_DIAGNOSIS.md](VI_DIAGNOSIS.md) distinguish target
parity from optimization differences. The inherited diagonal guide has
known conditional undercoverage; faster execution does not repair it.

## Comparable-accuracy conditional control

[benchmark_conditional_accuracy.py](benchmark_conditional_accuracy.py)
uses the existing 64 **prior-predictive** datasets with two unknown log-PSD
coefficients and other nuisance parameters held known. Both engines
optimize the same analytic full-Gaussian Whittle ELBO, with data-based MAP
and Laplace initialization, 2,000 Adam steps (.01 then .001), float32.
Independent float64 quadrature and the deterministic Gaussian optimum
from the earlier calibration study provide accuracy references. This is
an analytic backend control, not the production stochastic SGVB guide.

The script defines its common accuracy gate before running: every dataset
finite; maximum negative-ELBO gap <.005; maximum mean error <.005 quadrature
marginal SD in Euclidean norm; maximum covariance Frobenius relative error
<3%. Both backends pass on every fit. Across both engines:

- Maximum mean error: .000327 quadrature SD; covariance error: 1.408%.
- Maximum ELBO gap to the Gaussian optimum: <8e-12 in independent scoring.
- Nominal 90% coverage: intercept 62/64, slope 58/64, middle-frequency PSD
  61/64, identical in both engines and the earlier full-Gaussian control.
- Replicas repeat the same datasets and measure timing, not new calibration
  simulations. With only 64 datasets, coverage is uncertain (roughly four
  percentage-point binomial SE near 90%). Calibration alone is not proof
  of accurate individual posteriors; the quadrature checks add that control.

Two fresh processes/backend, two warm repeats each:

| Control across 64 datasets | JAX | TF | TF/JAX |
|---|---:|---:|---:|
| Cold result including imports and preparation | .545 s | 2.817 s | 5.17× |
| Warm optimization fit | .0312 s | .2431 s | 7.78× |

This establishes a speed gain at comparable accuracy **for this conditional
control only**, not full-model SBC, general calibration, better accuracy
than TF, or the same gain for richer production guides. See
[conditional processes](upgrade-benchmark/conditional-processes.json) and
[summary](upgrade-benchmark/summary.json).

## Validation and reproduction

- Combined reference environment: **68 passed**, 8 warnings, 109.97 s
  ([log](upgrade-benchmark/tests-reference.log)). Includes original
  VAR2D/eigenbasis reference cases and target/gradient/PSD/multi-draw tests.
- Fresh TensorFlow-free environment: **29 passed, 5 reference modules skipped**,
  22.98 s ([log](upgrade-benchmark/tests-jaxonly.log)). Public JAX 1D/VAR2D,
  RNG, FFT, guide and cache checks pass; reference skips are explicit.
- Offline wheel build passed. Python AST imports and metadata contain no
  TF/TFP/tf-keras imports/dependencies, no `legacy` extra, and no archive/cache
  files ([inspection](upgrade-benchmark/wheel-inspection.json)).
  Extracted-wheel public 1D/VAR2D inference passes with TF imports forbidden
  ([smoke log](upgrade-benchmark/wheel-smoke.log)). The tested development
  wheel records pre-commit HEAD plus dirty-tree date; SHA256 is recorded,
  and this is not a published release version.
- Reference diagnostic cast/degenerate interval and plotting warnings are
  preserved in logs. No remote CI/Linux, GPU or full-model calibration run
  is claimed. The original and new runtime share pure preprocessing helpers;
  they are not independent FFT implementations.

From the checkout, with the isolated environments installed:

```bash
export MPLBACKEND=Agg MPLCONFIGDIR=.mpl-cache
.venv-jaxonly/bin/python validation/benchmark_upgrade.py
.venv-jaxonly/bin/python validation/benchmark_conditional_accuracy.py
.venv-jaxonly/bin/python validation/summarize_upgrade.py
.venv/bin/python -m pytest -q
.venv-jaxonly/bin/python -m pytest -q
UV_CACHE_DIR=.uv-cache uv build --wheel --offline --out-dir /tmp/sgvb-upgrade-wheel
```

The coordinator expects `.venv-tfref` from `reference/requirements.txt` and
`.venv-jaxonly` from `pip install -e '.[dev]'` with scientific constraints.
Serialized data and per-worker logs preserve this bounded experiment.

## Remaining work and publication

Port/validate eigenbasis joint/factorized capabilities; promote a suitably
parameterized row-covariance guide after full-model prior-predictive and
posterior-reference checks; decide prior scale and PSD units separately;
resolve `cleared_unisgvb` licensing before copying AR code. No code from
that unlicensed source has been incorporated.

The PR description is updated locally in [PR_DRAFT.md](PR_DRAFT.md).
No open PR was found by the read-only check. Earlier publication of
`feature/jax-sgvb` to `nz-gravity/sgvb_psd` was denied before execution by
automatic approval review despite explicit user permission in the room.
Its stated reason was that the follow-up addressed calibration/naming,
not publication. The latest delegated instruction says not to retry the
denied push; no push or PR mutation was attempted in this pass. The branch
and complete binary patch remain local and reviewable.
