Title: Add experimental JAX joint SGVB inference with parity and calibration controls

Add `PSDEstimator(..., backend="jax")` for univariate and multivariate
original-joint inference. Shared preprocessing and interval summaries retain
the current conventions, and TensorFlow remains the default with both
eigenbasis modes preserved. The JAX default matches the reference's fixed
MAP guide means; `optimize_mean=True` explicitly enables trainable means.

The initial roughly 30% short-run PSD difference was traced to the different
guide means, rather than a changed target. Matched-noise ELBO values,
gradients and first optimizer updates agree; longer three-seed comparisons
approach the same held-out objective. See `validation/VI_DIAGNOSIS.md`.

Validation: the complete local suite passed 58 tests, including existing
VAR2D/all-mode cases and new likelihood/prior/gradient/PSD/FFT/RNG/guide tests.
CI gains a pinned-stack JAX parity job; no remote CI result is claimed yet.

Keep this experimental: a 64-simulation conditional prior-predictive screen
finds undercoverage in both diagonal guide variants. Full-covariance
conditional controls agree much better with quadrature, but full-model
calibration and a production covariance-aware guide remain open. AR code
from `cleared_unisgvb` was not copied because its licensing/provenance is
unresolved. No merge or TensorFlow retirement is proposed.
