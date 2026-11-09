Title: Review JAX SGVB port and validation

This draft replaces the installed TensorFlow inference path with JAX/NumPyro
for original-joint 1D and multivariate SGVB. The goal is to review whether
the implementation reproduces the intended method and scientific capabilities.
It is not ready for merging or release.

**Review only: no version bump, tag, PyPI release or major release is proposed.**
The existing release workflow triggers on pushes to `main` or `v*` tags,
not on this feature branch/draft PR. This branch includes historical `feat!`
commits and compatibility changes; those must be resolved or explicitly
reconsidered before any eventual merge. Keeping this PR draft does not
make the current code backward compatible.

## What changed

- Original-joint inference supports `(n,)` and `(n, p)` through the public
  `PSDEstimator`. NumPyro runs a custom independent-Normal guide with fixed
  MAP means by default; `optimize_mean=True` explicitly trains means.
- The blocked Whittle target, shrinkage priors, log-coordinate Jacobians,
  Fourier preprocessing and PSD conventions are ported with parity controls.
  Each variational draw has its own target value, avoiding the reference's
  scalar likelihood broadcasting across multiple draws.
- Per-runner JAX compilation is reused; current MAP locations and learning
  rates are dynamic arguments, with tests against stale cached values.
- TensorFlow dependencies and the legacy install extra are removed. The MIT
  reference implementation is archived in `validation/reference`, excluded
  from the installed wheel. No external univariate source was copied.

## Capability and compatibility gaps to review

| Area | Current branch |
|---|---|
| Original-joint 1D / multivariate inference | Implemented in JAX |
| Eigenbasis joint / factorized inference | JAX ports unfinished; requests fail explicitly; historical source is validation-only |
| AR prewhitening / recolouring workflow | Not integrated |
| Default guide | Independent Normal with frozen MAP means; not a production structured-covariance selector |
| Optional TF low-rank guide family | Faithful JAX scale-family building block and research comparisons; not exposed by the production estimator |
| Scientific units / prior questions | Inherited behavior retained and documented; no silent correction |
| Full-model Bayesian calibration | Not established |

## Suggested reading order

1. [Public estimator](../src/sgvb_psd/psd_estimator.py): input validation,
   API changes, unsupported modes, normalization and learning-rate behavior.
2. [Fourier preprocessing](../src/sgvb_psd/_spectral.py) and
   [JAX target](../src/sgvb_psd/jax_backend/model.py): axes, FFT/fs conventions,
   Whittle likelihood, priors/Jacobians, Cholesky reconstruction and units.
   In particular, the reference's variance-named quantity remains a Normal
   **standard deviation**; `psd_all` and interval scaling remain different;
   inherited odd-length FFT bin behavior is documented.
3. [Production guide/optimizer](../src/sgvb_psd/jax_backend/vi_runner.py) and
   [low-rank scale family](../src/sgvb_psd/jax_backend/guides.py): fixed versus
   trainable means, initialization, gradients, per-particle axes, RNG/dtype,
   compilation caching and scale-form versus covariance-form low rank.
4. [Target parity tests](../tests/test_jax_backend.py) and
   [runtime tests](../tests/test_jax_runtime.py): likelihood/prior/gradient/PSD
   agreement, 1D/VAR2D behavior, multiple draws, RNG and cache regression checks.
5. [Migration](JAX_MIGRATION.md), [VI diagnosis](VI_DIAGNOSIS.md),
   [guide comparison](GUIDE_COMPARISON.md), [performance](UPGRADE_BENCHMARK.md)
   and [100-run VAR2 coverage](VAR2_COVERAGE.md). Generated logs, numerical
   artifacts and figures are supporting evidence rather than the core code diff.

## Evidence and its limits

- The full reference-enabled suite passed 68 tests. A TensorFlow-free
  environment passed 29 tests with five reference modules skipped.
  The later coverage/runtime focused suite passed 32 tests.
- A built wheel contains no TF imports/dependencies or archived reference
  source; public 1D and VAR2D smoke inference passes with TF imports forbidden.
- Measured CPU first-result costs improve 1.52–2.94x and repeated-fit costs
  1.77–10.41x across three cases. Warm fits reuse a prepared runner.
  Full-model functional medians differ <.46%, but interval widths differ
  up to 29%; this does not prove equal posterior accuracy.
- The initial ~30% difference was traced to training guide means in one
  implementation while the TF reference froze them. Matched-noise
  objective/gradient controls distinguish target parity from VI optimization.
- A separate 64-dataset conditional full-Gaussian control passes the same
  quadrature accuracy checks in both engines; it is not full SGVB calibration.
- **The full-production fixed-truth VAR2 campaign completed 100/100 fits,
  with no failures, timeouts, retries or outcome-based tuning.** For nominal
  90% equal-tail intervals, frequency-average coverage is 71.6%/62.0% for
  the PSDs, 65.5%/53.0% for real/imaginary CSD and 68.6% for coherence.
  These averages are separate from per-frequency counts/Wilson bands and
  are neither prior-predictive SBC nor simultaneous coverage.
- 32/100 MAP traces have a declining late objective. This identifies an
  unresolved finite-budget pipeline limitation, not an isolated guide-family
  diagnosis or proof of a JAX regression. No matched 100-run TF campaign
  reproducing the original study's exact setup and aggregation is claimed.
- No remote CI result is claimed at PR creation; the local checks and raw
  evidence are recorded with versions, seeds and revisions.

## Review questions before considering a merge

- [ ] Does the JAX target reproduce the intended likelihood, prior scales
      and Jacobians, including all data/draw axes and PSD normalization?
- [ ] Does the custom guide preserve the desired TF behavior, and are
      intentional changes explicit and adequately tested?
- [ ] Do we retain every required scientific capability and public API,
      particularly eigenbasis/factorized inference, without requiring a
      major-release compatibility break?
- [ ] Are optimization stability and the comparison to the original TF
      method adequate for a faithful, faster port? Treat improved uncertainty
      calibration as a separate, explicitly evaluated change.
- [ ] Are documentation, dependencies, tests and migration behavior ready
      for users? Confirm release policy separately; keep this PR draft.
