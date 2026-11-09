# JAX migration: scope and scientific conventions

Draft PR [#13](https://github.com/nz-gravity/sgvb_psd/pull/13) ports the MIT
original-joint SGVB model from base `8b0f1f59241b0e5918c962212a4516123d8c8131`.
The installed package, development tests and CI use JAX/NumPyro only. The
TensorFlow reference package, dependencies, scripts and CI job are removed.
Historical parity evidence was collected before that removal; the current
scientific tests use independent NumPy/SciPy formulas and finite differences.

## Available and missing capabilities

- Public original-joint inference accepts 1D `(n,)` or multivariate `(n,p)` data.
- The production guide has fixed MAP means and independent Normal scales;
  `optimize_mean=True` explicitly enables mean training.
- Eigenbasis joint and factorized inference have **not** been ported to JAX;
  requests fail explicitly. No TF fallback or legacy extra is supported.
- AR whitening/recolouring is not integrated. No external univariate source
  was copied. The user's collaboration context does not itself establish
  redistribution licensing for that source.
- Rich covariance guides are research comparisons, not production options.
  Full-model Bayesian calibration is not established.

## Conventions preserved for backend parity

FFT uses `fft/sqrt(block length)`, no taper or demeaning, tail truncation,
DC/Nyquist exclusion, and cycles/sample for the basis. The inherited odd-block
implementation also omits the final positive-frequency bin. The model uses
float32/complex64 without changing the caller's global dtype configuration.

The Whittle target returns one value per variational draw, reducing data
axes internally. This avoids broadcasting a scalar summed likelihood over
multiple prior values. NumPyro's improper base measure contributes no extra
prior; log-coordinate Jacobians appear once in the factor target.

The shrinkage expression named like a variance in the reference is retained
as a Normal **standard deviation**, since taking its square root would change
the model. Prior-scale interpretation is a separate scientific decision.
PSD density is two-sided at retained positive bins, divided by `fs`. The
historical API's `psd_all` excludes input scale squared while interval arrays
include it; this inconsistency is documented and explicitly handled in
coverage summaries. No scientific correction is hidden in dependency cleanup.

Per-runner compilation caches receive current MAP means and learning rates
dynamically. PRNG keys are local to each runner. Tests cover changed cached
locations/rates, 1D/VAR2D inference, multiple particles, prior Jacobians,
independent Fourier residual likelihoods, directional gradients, inverse-
precision PSD reconstruction, unit scaling and low-rank Gaussian geometry.

## Recorded evidence and limitations

Initial TF/JAX fixed-parameter targets agreed at float32 tolerance; gradient
maximum absolute difference was 7.63e-5 and PSD difference 4.47e-8 in the
recorded controls. An initial ~30% VI difference was traced to trainable JAX
means versus frozen reference means; the JAX default now freezes means.
These historical checks do not prove converged VI, calibrated uncertainty
or completeness of every scientific capability.

The [100-run fixed-truth VAR2 study](VAR2_COVERAGE.md) demonstrates substantial
undercoverage and unresolved optimization stability for its finite-budget
configuration. [Performance](UPGRADE_BENCHMARK.md) gains do not imply equal
accuracy. Current aggregate records are in `summaries/`; original raw
outputs are local ignored files and remain in earlier Git history.

See [validation instructions](README.md) for setup, reproduction, local
backup provenance and artifact policy. This remains a draft for review:
no merge, version bump, tag or release is proposed.
