Title: Review JAX-only SGVB port and validation

Draft for code review only. No merge, version bump, tag or release is
proposed. Existing compatibility gaps and historical `feat!` commits must
be resolved before an eventual merge; this is not release-ready.

## Scope

- JAX/NumPyro original-joint SGVB for 1D and multivariate data, with fixed
  MAP guide means by default and explicit `optimize_mean=True`.
- Per-draw Whittle target, inherited priors/Jacobians and PSD conventions,
  local RNG/dtype policy, and per-runner compilation caches.
- JAX-only runtime, development tests, benchmark and CI. TensorFlow backend,
  archive, parity dependency group, comparison scripts and CI job removed.
- Generated arrays, per-run JSON/logs/plots/CSV and environment dumps are
  untracked and ignored. A verified local copy is preserved. Compact
  historical aggregates live in `validation/summaries`.
- No history rewrite/force-push: old blobs remain in earlier commits, so
  this cleans the current PR diff, not the repository's historical size.

## Reading order

1. [Estimator](../src/sgvb_psd/psd_estimator.py): public API, unsupported modes,
   input normalization and sampling.
2. [Preprocessing](../src/sgvb_psd/_spectral.py) and
   [model](../src/sgvb_psd/jax_backend/model.py): FFT, axes, prior scales,
   log Jacobians, Whittle target, PSD reconstruction and physical units.
3. [Guide/MAP](../src/sgvb_psd/jax_backend/vi_runner.py) and
   [scale-form Gaussian](../src/sgvb_psd/jax_backend/guides.py).
4. [Independent NumPy/SciPy controls](../tests/test_jax_backend.py),
   [runtime tests](../tests/test_jax_runtime.py) and
   [VAR2 end-to-end checks](../tests/test_end_to_end.py).
5. [Reproduction/artifact policy](README.md),
   [migration scope](JAX_MIGRATION.md), [guides](GUIDE_COMPARISON.md),
   [historical costs](UPGRADE_BENCHMARK.md), [VAR2 coverage](VAR2_COVERAGE.md).

## Scientific and compatibility gaps

Eigenbasis joint/factorized JAX inference is unfinished and fails explicitly;
there is no TF fallback. AR whitening/recolouring is not integrated. Richer
covariance guides are research comparisons, not production options. The
variance-named prior quantity remains a Normal standard deviation, odd FFT
bin behavior is retained, and raw-draw/interval input scaling differs as
before. These require explicit scientific/API decisions, not silent fixes.

Historical target parity is recorded without retaining TF support. Current
checks independently validate Fourier residual likelihoods, priors/Jacobians,
directional finite-difference gradients, inverse-precision PSD reconstruction,
scale-family geometry, 1D/VAR2D behavior, RNG, multi-draw targets and caches.

The completed 100-run fixed-truth VAR2 study has no failed/retried fits, but
nominal 90% intervals have frequency-average coverage 71.6%/62.0% (PSDs),
65.5%/53.0% (real/imaginary CSD) and 68.6% (coherence). These averages are
separate from per-frequency Wilson bands, not SBC/simultaneous coverage.
32 late MAP traces decline. This is a finite-budget pipeline limitation,
not an isolated guide/backend diagnosis. No matched 100-run TF campaign
reproducing the original study's exact setup/aggregation is claimed.

Historical measured CPU gains are 1.52–2.94x cold and 1.77–10.41x warm,
but full-model interval widths differ up to 29%; equal accuracy is not
established. The separate conditional quadrature control is not full SGVB.

## Review checklist

- [ ] Target, axes, priors/Jacobians and physical normalization are intended.
- [ ] Guide behavior, optimization and caching match the intended method.
- [ ] Required public/scientific capabilities are retained or gaps resolved
      before any release decision, including compatibility without a major bump.
- [ ] Tests and reproduced evidence justify claims of faithful accuracy/speed.
- [ ] Generated experiment outputs stay local; only useful summaries are tracked.
- [ ] Keep draft: no merge/release until the review and explicit release decision.


Cleanup validation: **55 JAX-only tests passed with no skipped legacy modules**.
This includes independent NumPy/SciPy target/gradient/PSD controls and the
existing bivariate VAR2 model through the public estimator. Concurrent uv
setup changes were preserved, with the TensorFlow parity group removed and
the lockfile refreshed for the JAX stack. Historical raw outputs have a
verified local backup; the 100-run campaign was not rerun or retuned.

Wheel and source-distribution scans passed: no generated experiment outputs,
local backups or TensorFlow reference source are packaged. The refreshed
lockfile check and JAX benchmark smoke passed. See
[cleanup verification](summaries/cleanup-verification.json).
