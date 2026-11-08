Title: Make the default SGVB runtime JAX-only and retain explicit legacy comparisons

Add `PSDEstimator(..., backend="jax")` for univariate and multivariate
original-joint inference. Shared preprocessing and interval summaries retain
the current conventions. TensorFlow is removed from default dependencies
and imports; both eigenbasis modes remain in the explicitly selected
optional `[legacy]` path until their JAX ports are validated. The JAX default matches the reference's fixed
MAP guide means; `optimize_mean=True` explicitly enables trainable means.

The initial roughly 30% short-run PSD difference was traced to the different
guide means, rather than a changed target. Matched-noise ELBO values,
gradients and first optimizer updates agree; longer three-seed comparisons
approach the same held-out objective. See `validation/VI_DIAGNOSIS.md`.

Validation: the complete local suite passed 65 tests (one expected skip), and a fresh TensorFlow-free environment passed 27 tests (five legacy modules skipped), including existing
VAR2D/all-mode cases and new likelihood/prior/gradient/PSD/FFT/RNG/guide tests.
CI gains a pinned-stack JAX parity job; no remote CI result is claimed yet.

Keep this experimental: a 64-simulation conditional prior-predictive screen
finds undercoverage in both diagonal guide variants. Full-covariance
conditional controls agree much better with quadrature, but full-model
calibration and a production covariance-aware guide remain open. AR code
from `cleared_unisgvb` was not copied because its licensing/provenance is
unresolved. No merge or full replacement of every scientific mode is proposed. The
custom optional low-rank scale family has an exact JAX building block, and
bounded comparisons of seven guide options are in `validation/GUIDE_COMPARISON.md`.
