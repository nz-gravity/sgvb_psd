# JAX guide families and bounded comparisons

NumPyro is the inference engine; it supports custom guides and several
families. The production estimator currently uses a custom independent-
Normal guide, fixed at MAP locations by default. Training means is an
explicit option. A covariance-aware production selector is not implemented.

## Scale-form low rank

The original optional custom family used a **scale** matrix
`S = diag(d) + U Uᵀ`, so covariance is `S Sᵀ`. It was independent across
parameter groups, with no cross-group covariance. The default path was
diagonal. Positive integer ranks worked in the bounded reference check;
float ranks and a supplied guide being overwritten were historical issues.

`jax_backend/guides.py` retains a JAX-only scale-form Gaussian building
block. It computes log density through the scale solve rather than squaring
its condition number. NumPyro's covariance-form low-rank distribution uses
`diag(d) + F Fᵀ`, a different family. Current tests compare the scale-form
implementation to SciPy's Gaussian density and covariance, requiring no TF.

## Historical research comparison

Seven guide options were compared at an unchanged target on AR1 (15 latent
coordinates) and VAR2 (63 coordinates, regression-row groups 15+48).
Each used 6,000 steps, 8 particles, clipped Adam .01 then .001 and two seeds.
These are research harnesses, not estimator backend modes. Initial scales
were different for faithful versus richer guides, so this is not a clean
covariance-only ablation or a general speed comparison.

VAR2 negative held-out ELBO ranges:

| Guide | Two-seed range |
|---|---:|
| Frozen diagonal | 393.4–394.2 |
| Frozen scale-form low rank 2 | 390.3–391.5 |
| Trainable diagonal | 69.4–69.6 |
| Covariance low rank 4 | 62.2–62.5 |
| Dense full | 54.8–54.8 |
| Regression-row full | 54.3–54.5 |

A longer MAP initialization changed these objectives materially. Warm cost
measurements were collected alongside other research jobs, so they do not
establish a general cost advantage. Compact records:
[guides](summaries/guide-comparison.json) and
[sampler diagnostics](summaries/sampling-diagnostics.json).

NUTS did **not** qualify as a gold-standard reference: refined AR1 still had
19 divergences; VAR2 saturated tree depth in 1,983/2,000 retained draws and
had poor original-coordinate mixing. Failed diagnostics and draws were
retained. The conditional quadrature study supplies stronger evidence in
its much smaller scope.

Recommended next work is stable initialization/noncentering and a custom
row-covariance guide using the NumPyro engine, followed by independent
posterior-reference and full-model calibration checks. Low rank is useful
when dense row covariance becomes expensive. Flows or DAIS can be explored
after these controls, not assumed more accurate without evidence.

Reproduce with `guide_reference.py`, `compare_guides.py`, `guide_tools.py`
and `summarize_guides.py`; [instructions](README.md). Arrays and raw results
are ignored local outputs. No TensorFlow support or dependency is retained.
