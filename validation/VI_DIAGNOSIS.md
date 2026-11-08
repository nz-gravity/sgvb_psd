# VI discrepancy and conditional Bayesian calibration

The latest complete regression suite passed **58 tests** in 94.33 seconds,
including the original slow all-mode comparison and 37 JAX tests.
[Final test log](vi-diagnosis/final-tests.log).

The approximately 30% short-run PSD difference had a concrete cause: the
initial JAX guide optimized its Normal means; TensorFlow's existing guide
does not. Both targets use the same likelihood and priors, but these were
different variational approximations. Restoring fixed MAP means resolves
the large short-run difference. Neither backend's diagonal approximation
should be assumed calibrated on that basis.

## Family and implementation controls

TensorFlow constructs each distribution with `loc=params[i][0]` outside a
gradient tape. That slice is an `EagerTensor` snapshot, not a trainable
location. Runtime inspection finds eight `q_z_scale` variables and no
trainable means. Its posterior means remain exactly equal to MAP after VI.
The initial JAX implementation instead registered a NumPyro `_loc` parameter.

The JAX default now keeps means fixed to match the reference. Setting
`PSDEstimator(..., backend="jax", optimize_mean=True)` explicitly enables
mean optimization. This changes the variational family, not the target
likelihood, priors, or Jacobians. TensorFlow rejects that option rather than
ignoring it. This is not a decision to prefer the fixed-mean approximation
scientifically; it makes the comparison and user choice explicit.

The same initial MAP tensors were supplied to both guides for the initial
AR(1) and existing-style VAR2D cases (512 samples, two blocks, six basis
terms, 100 MAP updates at 0.003). Scale initialization remained `1e-4` with
a softplus transform. With four **identical noise draws**, the negative
ELBO values matched at reported precision; maximum scale-coordinate gradient
differences were `2.38e-7`; the first default Adam update matched at float32
precision. Keras uses epsilon `1e-7` and NumPyro Adam `1e-8`, so bitwise
equality of long stochastic paths is not required.

The reference likelihood sums variational draws into a scalar. For the
matched multi-draw objective control, the complete summed target is divided
by the draw count. Using the legacy scalar likelihood plus a vector prior
directly with several draws would multiply the likelihood contribution.
Production-style TF runs retain their default single particle and therefore
do not encounter that broadcasting issue.

At 50 VI steps, fixed-mean VAR2D medians differed by at most `5.6e-6`
relative L2 across seeds 0, 1, and 2. A diagnostic counterfactual that frees
the means reproduces a **29.97%** VAR2D difference and moves a location by
up to 1.585. Its held-out negative ELBO improves substantially. That
explains the initial discrepancy; matching TensorFlow's fitted PSD is not
itself an accuracy criterion.

## Longer runs and stochastic convergence

The same two cases and three seeds were run to 50, 1,000, 5,000, and 10,000
updates. Optimizer state continued across checkpoints. Both backends used
one ELBO particle, Adam at 0.05 for 5,000 updates, then 0.005 for another
5,000. Evaluation used 2,048 common held-out draws, separate from training.
No run was discarded or chosen as a winning seed.

| Final check over three seeds | AR(1) | VAR2D |
|---|---:|---:|
| TF negative ELBO range | 216.581–216.636 | 393.911–394.048 |
| JAX negative ELBO range | 216.581–216.617 | 393.915–394.088 |
| Largest paired ELBO difference | 0.055 | 0.039 |
| Held-out ELBO Monte Carlo SE | 0.178–0.202 | 0.334–0.353 |
| PSD median relative L2 difference | 0.117–0.127% | 0.111–0.113% |
| Pointwise interval-width relative L2 difference | 3.5–9.8% | 2.3–4.0% |

All training and evaluation arrays were finite. Constant 0.05 updates
showed appreciably larger width and ELBO variation at 5,000 steps; lowering
the learning rate reduced it. The final held-out objectives are consistent
within their evaluation uncertainty and across seeds. This is a bounded
stability check, not a proof of asymptotic convergence or posterior accuracy.
Fixed MAP centres here intentionally reproduce the original 100-step
control; this does not establish that the full MAP optimization converged.

Reproduce with:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.mpl-cache \
  .venv/bin/python validation/investigate_vi.py
```

Sources and complete results: [script](investigate_vi.py),
[guide introspection](vi-diagnosis/guide-introspection.log),
[constant-rate results](vi-diagnosis/results-constant-lr.json),
[continued/lower-rate results](vi-diagnosis/results.json),
[run log](vi-diagnosis/run.log).

## Prior-predictive calibration screen

A separate screen evaluates the **conditional univariate Whittle submodel**
with only its intercept and slope unknown. Those two coefficients retain
their actual independent `Normal(0, sqrt(10))` priors. Curvature and shrinkage
nuisance coordinates are held known. For each of 64 prior draws, the summed
power at each frequency is sampled from its exact Gamma distribution under
two independent complex-Gaussian blocks. This corresponds to 128 real time
samples with 31 retained bins per block. Data are not standardized or selected
by the simulated parameter values.

A regression test verifies both the inverse-FFT generator's normalization
and equality of this conditional target to the full JAX model up to a
parameter-independent constant. This is a valid prior-predictive experiment
for the stated conditional model. **It is not full SGVB SBC**, an AR(1)
coverage study, VAR2D calibration, or a GW150914 validation.

The reference posterior is computed by two-dimensional quadrature around
the exact MAP/Hessian. CDFs integrate interpolated cumulative mass rather
than counting grid indicators. Boundary mass was below `2.6e-11`; refining
161 to 241 points per dimension changed CDFs by less than `0.000343` in the
four predefined refinement controls.

Gaussian-family negative ELBOs have analytic expectations for this
log-linear Whittle submodel. Deterministic float64 optimization in a local
context therefore removes stochastic convergence as an explanation for the
calibration result. Production-style NumPyro controls remain float32.

| Nominal 90% interval coverage | Intercept | Slope | log PSD at f=0.25 |
|---|---:|---:|---:|
| Quadrature reference | 62/64 (96.9%) | 58/64 (90.6%) | 61/64 (95.3%) |
| Fixed-mean diagonal Gaussian | 42/64 (65.6%) | 44/64 (68.8%) | 63/64 (98.4%) |
| Trainable-mean diagonal Gaussian | 42/64 (65.6%) | 44/64 (68.8%) | 63/64 (98.4%) |
| Trainable-mean full-covariance Gaussian control | 62/64 (96.9%) | 58/64 (90.6%) | 61/64 (95.3%) |

The diagonal intercept coverage has a 95% Wilson interval of 53.4–76.1%;
for its slope the interval is 56.6–78.8%. Both miss the nominal 90% level.
The quadrature/full-covariance coverage estimates have wide intervals that
include 90%. The finite sample does not establish exact calibration. PIT
histograms and unadjusted diagnostic KS values are recorded, not treated
as independent confirmatory tests or evidence of uniformity.

The oracle coefficient correlation ranges from -0.891 to -0.833. Discarding
that covariance explains why good coefficient means coexist with poor
intervals. Median posterior covariance errors versus quadrature are about
78% for both diagonal families and 1.36% for the full-covariance control.
The latter's maximum error in mean is below 0.00033 oracle marginal
standard deviations. These direct posterior comparisons support the
calibration screen; calibration alone would not establish accurate
individual posteriors. The same diagonal approximation is overconservative
for the middle-frequency PSD functional, demonstrating why one functional's
coverage cannot certify the whole posterior.

Eight predefined float32 NumPyro optimization controls, with 16 particles
and a lower late learning rate, reached analytic negative ELBO gaps below
0.004 for the three families. The dense guide required a per-dataset
Laplace Cholesky initialization. An earlier attempt using a `1e-4` diagonal
dense initialization produced four unusable held-out objectives out of eight,
including two overflows despite finite training losses. Those failures are
retained in the initial results; no failed simulation was silently removed.
This is a warning against introducing a naive dense guide as production-ready.

Reproduce with:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=.mpl-cache \
  .venv/bin/python validation/calibration_screen.py
```

Evidence: [script](calibration_screen.py),
[all simulations and optimization controls](vi-diagnosis/calibration-results.json),
[compact accuracy/calibration summary](vi-diagnosis/calibration-summary.json),
[initial control failures and superseded quadrature](vi-diagnosis/calibration-results-initial.json),
[run log](vi-diagnosis/calibration.log).

## Implications for the migration

The large initial discrepancy was a guide-family mismatch. Likelihood/prior
parity, matched-noise objective gradients, and long-run objectives now support
the backend implementation. They do not rescue the inherited diagonal
guide's uncertainty approximation. Simply training the means did not solve
the covariance problem in this screen.

Keep this change experimental. A covariance-aware or reparameterized guide
is a useful next implementation, followed by full prior-predictive SGVB SBC,
independent posterior references, and multivariate PSD/coherence functionals.
The dense conditional control is promising evidence for that direction,
not a validated full-model implementation. Existing prior-scale, PSD-unit,
AR provenance, and full eigenbasis-port questions remain separate.
