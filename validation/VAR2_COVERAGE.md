# Production JAX VAR2 fixed-truth pointwise coverage

This is a fixed-truth frequentist coverage experiment across 100 independently
simulated bivariate VAR(2) datasets. It is **not prior-predictive SBC** and
**not simultaneous/uniform coverage**. Each retained frequency gets its own
coverage estimate with realizations as the independent trials. The experiment
measures this configured production pipeline, including its inherited guide,
preprocessing, finite optimization budget and spline approximation.

The protocol and seeds were frozen before evaluating any campaign fit in
[config.json](var2-coverage/config.json) and
[manifest.json](var2-coverage/manifest.json). Implementation revision at
freeze: `927c9b5078a729a5c3f4d00e96a7ead878a5813f`; production runtime from
`b53cf769ec6294a2501170a2cb3fdd43cbc68aa3`. The frozen script SHA256 binds
[the campaign](var2_coverage.py). The separate
[analysis](analyze_var2_coverage.py) never changes inference or retries fits.

## Actual completion and findings

**100 planned, 100 attempted, 100 completed; zero failed/interrupted,
zero pending, no retries or outcome-based tuning.** All 100 retained
functional outputs and optimization traces are finite. This configured
default-guide pipeline substantially undercovers the true spectrum.

The following averages are **frequency averages**, not additional binomial
trials. Their SE uses independent per-realization averages. The range
column describes the separate per-frequency coverage estimates (each has
100 independent trials); uncertainty for every frequency is in the CSV/plot.

| Quantity | Frequency-average coverage ± realization SE | Per-frequency coverage range |
|---|---:|---:|
| PSD00 | 71.6% ± 2.3 percentage points | 47–94% |
| PSD11 | 62.0% ± 1.1 percentage points | 13–89% |
| Re(S01) | 65.5% ± 1.5 percentage points | 18–89% |
| Im(S01) | 53.0% ± 1.3 percentage points | 9–96% |
| Coherence | 68.6% ± 1.6 percentage points | 49–85% |

The requested 90% pointwise coverage is not achieved across the grid.
Interior-bin averages are 72.9%, 60.9%, 65.5%, 51.8% and 67.5%, respectively;
edge-bin averages are 56.1%, 75.9%, 66.1%, 67.8% and 80.4%. The problem is
not confined to the edge bins. These comparisons are descriptive, not a
multiple-testing claim from scanning the frequency grid.

| Quantity | Median 90% interval width, native units | Normalized recovery RMSE |
|---|---:|---:|
| PSD00 | .04375 | .1501 |
| PSD11 | .05315 | .2054 |
| Re(S01) | .04108 | .1398 |
| Im(S01) | .03565 | .1201 |
| Coherence | .10094 | .05657 |

Width medians summarize all realizations/frequencies without treating them
as independent trials. Normalization for recovery is defined below. Native
mean absolute recovery errors are .02678, .04078, .01893, .01638 and .04216.

Optimization has material unresolved stability concerns. **32/100** last
logged MAP objectives are below the logged value 500 trace entries earlier;
changes range from −72.62 to +114.43 (median +1.54). The worst last-logged
objective is 86.75 below its trace maximum. Raw-coordinate gradient RMS
ranges 1.73–20,612 (median 325.9); it is coordinate dependent and is not an
independent convergence criterion. The last versus previous 1,000-step
VI loss-mean difference ranges −2.27 to +3.62, and final-window loss SD has
median 12.28. No fit was filtered using these diagnostics.

This establishes undercoverage for **this finite-budget pipeline**, not
that the guide alone caused it, that a JAX implementation error is proven,
or that a different covariance family will necessarily fix it. Frozen MAP
means, optimization/initialization, prior shrinkage, finite spline
approximation and the Whittle/window approximation remain possible
contributors. The 1.36% finite-window diagonal discrepancy does not by
itself explain or decompose the result. A next comparison should first
establish stable optimization and then prespecify mean/covariance/basis
controls with independent evaluation seeds; these final outcomes must not
be used for tune-until-favorable claims.

Summed fresh-worker wall cost was **1,042.45 s (17.37 min)**; per-worker
median 10.28 s, maximum 14.63 s. Median production fit time was 7.59 s
(including public sampling/interval postprocessing), with MAP .540 s and
VI 1.414 s. These are observed campaign costs, not an isolated cross-engine
benchmark. All workers used `TFRT_CPU_0`; JAX 0.6.2, NumPyro 0.19.0,
NumPy 2.5.4; none imported TensorFlow.

Standalone figures and numerical evidence:

- [Coverage versus frequency, PNG](var2-coverage/coverage.png),
  [PDF](var2-coverage/coverage.pdf), [SVG](var2-coverage/coverage.svg).
- [Native interval widths](var2-coverage/widths-native.pdf),
  [normalized widths](var2-coverage/widths.pdf),
  [median-recovery errors](var2-coverage/recovery.pdf).
- [Optimizer stability](var2-coverage/stability.pdf) and
  [traces selected by fixed indices 0, 49, 99](var2-coverage/traces.pdf).
- [Per-frequency CSV](var2-coverage/frequency-results.csv),
  [complete numerical arrays](var2-coverage/frequency-results.npz),
  [summary](var2-coverage/summary.json),
  [campaign log](var2-coverage/campaign.log).

Focused validation passed **32 tests**, one existing plotting warning,
in 14.58 s ([final log](var2-coverage/focused-tests-final.log)). Before
freezing, 30 production/runtime/mathematical tests passed; five focused
mathematical/summary checks also passed. Tests include independent closed
form spectrum/coherence, covariance integration, direct Toeplitz-window
expectation, and correlated-frequency/failure denominator controls.
No fit was repeated during validation. The
[artifact index](var2-coverage/artifact-index.json) records file hashes and
provenance, checked with `validation/check_var2_coverage.py`.

## Frozen model and procedure

- Existing test VAR(2): A1=diag(.5,−.3), A2=diag(0,−.5), independent Gaussian
  innovations with covariance [[1,.9],[.9,1]]. The existing `SimVARMA`
  simulator and initialization/burn-in are used without copying third-party
  code. The companion spectral radius is .7071, so the process is stable.
- 512 observations, two blocks of 256, six nonlinear spline basis terms
  plus intercept and slope; 127 retained positive-frequency bins, with DC
  and Nyquist excluded. Sampling rate is 2π; plots use cycles/sample.
  This sample size/block/basis choice matches the earlier small VAR2
  performance control; it is one finite configuration, not a sample-size study.
- The **full production original-joint model** is fitted: all 63 log-spectrum,
  complex Cholesky and shrinkage parameters are unknown. No nuisance
  parameters are held at truth. This is not the earlier conditional model.
- Guide: current production default independent Normal coefficients/log
  scales, with means **frozen at the fitted MAP**, trainable softplus scales
  initialized at 1e-4; `optimize_mean=False`, one ELBO particle. The model
  prior is unchanged: Normal base variance 10, inherited bounded
  variance-named expression used as a Normal standard deviation,
  log-HalfCauchy shrinkage/Jacobians, degree_fluctuate=3, Nbw=1.
- MAP: 5,000 Adam steps at .003. VI: 10,000 Adam steps at .05. The standard
  initialization, float32/complex64 policy and production implementation
  are unchanged. No learning-rate search, outcome-based tuning or restarts.
- 2,000 posterior draws per dataset. Intervals are separate scalar 5th/95th
  percentiles for PSD00, PSD11, Re(S01), Im(S01) and coherence
  |S01|²/(S00 S11). The posterior median supplies recovery errors.
- The master seed is 20261109. Per-index data and VI seeds use distinct
  NumPy SeedSequence namespaces 1101 and 2202; all 200 seeds are distinct.
  They were recorded in the manifest before the first fit.
- Every worker runs in a fresh process with a 120-second timeout. Failures
  retain JSON/log/partial numerical diagnostics, remain in counts and are
  not retried. Resume skips terminal results; interrupted unknown outcomes
  are counted as failures instead of receiving a favorable rerun. Stop an
  existing coordinator before resuming; do not launch two coordinators.
- Structured/full covariance options remain research harnesses rather than
  supported production estimator options. No secondary guide was prespecified;
  completing the primary 100-realization campaign takes precedence.

## Spectrum and normalization checks before fitting

Analytic S(f)=H(f) Σ H(f)* / fs, H(f)=(I−A1 exp(−2πif)−A2 exp(−4πif))⁻¹.
The convention is two-sided density per physical-frequency unit evaluated at
positive bins, without doubling. S01=E[X0 conjugate(X1)]/fs. Population
coherence is .9²=.81 at every frequency for these diagonal AR filters.

Before campaign outcomes, independent checks established:

- Exact agreement with the existing analytic spectral helper on the fitting
  grid; integration over the full frequency circle recovers stationary
  covariance [[1.33333333,.70588235],[.70588235,1.38888889]].
- White-noise FFT densities .15932046 and .15907590 versus 1/(2π)=.15915494,
  using a separate preflight seed 917321. FFT scaling is fft/sqrt(block length).
- A constant-spectrum reconstruction checks division by fs and multiplication
  by input scale squared. Production `psd_all` omits input scale squared;
  this campaign explicitly restores it before scalar quantiles. Diagonal
  interval values are cross-checked against the production pointwise output.
- No demeaning or tapering is introduced. The global input sample std is
  estimated separately for each dataset as in the existing pipeline, and
  rescaling is restored. Priors act in those normalized coordinates.
- Exact finite rectangular-window expected periodograms were computed from
  stationary lag covariances. Their largest diagonal deviation from the
  asymptotic analytic spectrum is 1.356%. Coverage is assessed against the
  true infinite-process spectrum, not the expected windowed periodogram.

[Normalization results](var2-coverage/normalization.json),
[truth and window expectation](var2-coverage/truth.npz), and the focused tests
preserve these checks. Edge bins remain included; the first/last five are
also summarized separately and shaded in plots. Short blocks, leakage,
boundary spline behavior and DC/Nyquist omission limit edge interpretation.

## Interpretation of coverage and uncertainty

At each frequency, coverage is the number of independent realizations whose
interval contains truth divided by the number of completed realizations.
The figures include per-frequency 95% Wilson intervals. With 100 independent
runs, binomial standard error near nominal 90% is approximately .03; 90/100
has Wilson interval [.8256,.9448]. These bands are **not simultaneous** across
frequencies or quantities, and scanning many bins produces correlated
fluctuations. Real and imaginary CSD are separate scalar interval checks,
not a joint complex confidence region.

Frequency-average coverage is a separate descriptive quantity. Its SE is
computed from the 100 per-realization frequency averages. No binomial trial
count pools the 127 dependent frequency points. Failure accounting also
provides all-attempted/all-planned coverage bounds: unknown fits could
cover none or all frequencies. Failed fits are never silently excluded.

Widths and recovery errors are saved in physical units. Supplementary
normalized quantities use true PSD for diagonal terms, sqrt(S00 S11) for
CSD, and absolute units for coherence. CSD normalization avoids division by
near-zero real/imaginary truth at edge frequencies. Coverage includes finite
posterior-draw quantile error; no repeated-draw convergence claim is made.
Numerically finite traces alone do not establish optimizer convergence.

## Artifacts and reproduction

The campaign uses `.venv-jaxonly`; no TensorFlow or third-party univariate
source is needed. Hardware available to this session: macOS 27.0 arm64,
18 logical CPUs; Python 3.12.16; actual JAX devices and package versions
are recorded with results. A CPU model-name query was unavailable under
local OS permissions and was not retried. Full dependencies are in
[environment.lock](var2-coverage/environment.lock).

```bash
export MPLBACKEND=Agg MPLCONFIGDIR=.mpl-cache
# freeze was executed exactly once, before any campaign fit:
.venv-jaxonly/bin/python validation/var2_coverage.py freeze
# resume remaining pending entries after stopping any prior coordinator:
.venv-jaxonly/bin/python -u validation/var2_coverage.py run
.venv-jaxonly/bin/python validation/analyze_var2_coverage.py
.venv-jaxonly/bin/python -m pytest tests/test_var2_coverage.py tests/test_var2_coverage_summary.py tests/test_jax_runtime.py -q
```

Do not refreeze/overwrite this final evaluation or change budgets based on
its outcomes. `freeze` rejects an existing configuration. To reproduce a
new identical campaign, preserve this directory and run in a separate
checkout/output location with the recorded revision/configuration.

Every attempted fit has a log and JSON status; completed fits have NPZs
containing the exact data, truth, scalar intervals, coverage indicators,
widths/errors, full MAP/VI traces and variational/MAP parameters. Data and VI
seeds, versions and implementation determine posterior resampling. The final
frequency artifact contains counts, Wilson bands, failure bounds, error and
width curves; PNG/SVG/PDF plots are standalone scientific deliverables.

GitHub publication remains blocked by the earlier automatic-review denial.
No push or PR mutation is attempted for this investigation. Branch and commit
names contain no prohibited naming term. Licensing questions do not block
this independent VAR2 experiment; this does not establish a redistribution
license for the other repository.
