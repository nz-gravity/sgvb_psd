# Variational inference diagnosis

The initial roughly 30% short-run discrepancy came from a guide-family
difference: the TF implementation captured guide locations as eager tensors
outside the gradient tape, freezing them at MAP, whereas the initial JAX
implementation trained locations. The current production default freezes
means; `optimize_mean=True` is an explicit alternative approximation with
the same model target. No TensorFlow backend or parity environment is
maintained in the current tree.

Historical matched-noise controls agreed in ELBO and first Adam update at
reported float32 precision; maximum scale-gradient difference was 2.38e-7.
Backend Adam epsilon/bias conventions and stochastic streams differ.
Longer three-seed, matched-family runs approached comparable held-out
objectives, with PSD median differences around .11% and interval-width
variation of 2–10%. These are bounded controls, not convergence proofs.

## Conditional prior-predictive control

`calibration_screen.py` uses 64 prior-predictive 1D datasets with only two
baseline log-PSD coefficients unknown and other nuisance parameters fixed.
Quadrature and analytic Gaussian ELBOs distinguish approximation error from
optimization error. This is **not full SGVB calibration**.

Nominal 90% coefficient coverage was approximately 66–69% for the two
diagonal variants. Full-Gaussian controls covered 62/64 and 58/64, with
middle-frequency PSD coverage 61/64. Full covariance had median covariance
error about 1.36% versus roughly 78% for diagonal covariance. Dense SVI
from tiny scales also had initialization failures in a bounded control;
none was discarded. See the compact
[conditional aggregate](summaries/conditional-calibration.json).

The [production VAR2 campaign](VAR2_COVERAGE.md) is a separate fixed-truth
frequentist experiment. It neither establishes prior-predictive SBC nor
isolates the guide as the cause of undercoverage. Optimization, shrinkage,
finite basis approximation and frozen MAP locations remain possible factors.

Raw draws, per-simulation JSON, traces and logs are ignored local outputs.
See [reproduction and backup details](README.md). Historical TF investigation
scripts remain recoverable from revision `583e4f3`; they are not supported
by current dependencies or CI.
