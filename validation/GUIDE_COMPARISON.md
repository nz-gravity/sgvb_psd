# Custom guide, inference engine, and a JAX-only runtime

Use NumPyro as an inference engine while retaining explicit, custom guide
structure. There is no intrinsic "custom guide versus NumPyro" choice:
the same distribution can be implemented in plain JAX or passed to NumPyro.
The faithful manual and NumPyro controls below achieve similar objectives
and cost. Richer covariance and an adequate MAP/initialization stage matter
more here than which engine computes the reparameterized ELBO.

The current default runtime is now JAX-only. The existing guide is preserved
as a reference, including an exact JAX building block for its optional
low-rank **scale** family. A structured Gaussian by Cholesky regression row
is the leading next experiment, not yet a validated production replacement.

## What the TensorFlow custom guide actually is

In [ViRunner.init_surrogate_posterior](../src/sgvb_psd/backend/vi_runner.py#L156):

- Default `variation_factor <= 0`: a product of TFP diagonal multivariate
  Normals, assembled into a joint distribution. Means are MAP tensor
  snapshots; only standard-deviation scales train. Initial scales are
  `1e-4` with a softplus transform.
- Optional positive **integer** `variation_factor=r`: each event vector
  uses `MultivariateNormalDiagPlusLowRank`. Runtime construction and
  sampling work for integer ranks 1 and 2. Passing the float `1.0` fails
  shape construction despite the existing float annotation.
- For a parameter array `(rows, coefficients)`, correlation can occur
  between coefficients within each row. Rows remain independent, and the
  eight parameter groups remain independent. For example, delta coefficients
  cannot correlate with their local/global shrinkage parameters, and theta
  real coefficients cannot correlate with theta imaginary coefficients.
- Both paths keep their means fixed at MAP. Neither is a flow or a general
  hierarchical conditional guide. The eigenbasis joint runner uses the same
  assembly. Factorized inference additionally separates Cholesky components.
- The default `PSDEstimator` did not expose `variation_factor`, and no
  existing test selected the optional low-rank path. Thus "the existing
  guide is diagonal" describes the default, not every available branch.
- Supplying `surrogate_posterior` to the joint runner's constructor does not
  preserve it through training: `run_phase2` unconditionally constructs a
  replacement. This was verified at runtime, not inferred from the name.

The original optimizer is Keras Adam, MAP followed by VI at 0.05; the
reference TFP fit uses one Monte Carlo particle by default. There is no
automatic guarantee of convergence, calibration, or correct guide selection.
See [inspection log](guide-comparison/tf-guide-inspection.log).

## The low-rank distinction must survive a port

The original optional family has

```
S = diag(d) + U @ U.T
Cov = S @ S.T
```

NumPyro `LowRankMultivariateNormal` instead parameterizes covariance as a
diagonal plus a low-rank update. Substituting that class directly changes
the family and its optimization, even though both names say "low rank".
The [TFP scale-family documentation](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiagPlusLowRank)
states the distinction explicitly.

[jax_backend/guides.py](../src/sgvb_psd/jax_backend/guides.py) ports the
original operator and covariance exactly. Batched covariance and log-density
tests match TFP at ranks 1 and 2, and a JAX-only test shows it is distinct
from a covariance-form low-rank update. This building block needs no TFP
or TensorFlow. It can be used by either a custom JAX optimizer or a NumPyro
custom guide; it does not silently change the default production guide.

## Same-target bounded experiment

AR(1) and the existing VAR2D simulation coefficients are reused, with 512
samples, two blocks, six basis terms and the original priors/likelihood.
Latent dimensions are 15 and 63. A fixed diagonal affine preconditioner is
shared by all candidates; its log Jacobian is included. No likelihood,
FFT, PSD unit, prior-scale or window convention is changed.

Each candidate uses two seeds, 6,000 updates, eight particles, clipped Adam
at 0.01 for 4,000 steps and 0.001 thereafter. These settings control this
comparison; they are not claimed to reproduce the original one-particle,
unclipped 0.05 optimizer trajectory. Evaluation uses 2,048 independent draws.

Manual and NumPyro faithful diagonal guides have identical physical
`1e-4` scale initialization. The optional TFP scale-family port preserves its
uniform `[0,1]` factor initialization and fixed means. Richer guides learn
means and start at scale 0.1 in the shared preconditioned coordinates.
Those different initializers are explicit: the comparison is not a clean
ablation of covariance alone. The default 100-step MAP centre is also a
short control, addressed by the additional longer-MAP experiment below.

| Guide | Mean | AR(1) / VAR2D free parameters | What correlates |
|---|---|---:|---|
| Manual faithful JAX diagonal | fixed | 15 / 63 | nothing |
| NumPyro custom faithful diagonal | fixed | 15 / 63 | nothing |
| Faithful optional scale rank 2 | fixed | 45 / 189 | original event vectors only |
| AutoDiagonalNormal | trained | 30 / 126 | nothing |
| AutoLowRankMultivariateNormal, rank 4 | trained | 90 / 378 | covariance across all coordinates |
| AutoMultivariateNormal | trained | 135 / 2,079 | full Gaussian covariance |
| Custom Gaussian by Cholesky row | trained | 135 / 1,359 | full covariance within each regression row |

Here the original joint target factorizes by Cholesky regression row: each
likelihood residual and its prior parameters belong to that row. No inferred
hyperparameter is shared between different rows. The VAR2D blocks have 15
and 48 coordinates. The cross-row Hessian check is exactly zero; the factor
formulas, not that numerical check alone, justify the grouping. This claim
is limited to the current original-joint model and fixed hyperparameters.

| Guide | AR(1) negative ELBO, two seeds | VAR2D negative ELBO, two seeds | Warm VAR2D training |
|---|---:|---:|---:|
| Manual faithful diagonal | 216.72, 216.93 | 394.08, 393.63 | 1.82 s |
| NumPyro faithful diagonal | 216.85, 216.73 | 394.24, 393.38 | 1.73 s |
| Optional scale rank 2 | 215.60, 215.68 | 391.51, 390.32 | 2.09 s |
| AutoDiagonalNormal | 177.58, 177.93 | 69.44, 69.61 | 1.74 s |
| AutoLowRank, covariance rank 4 | 174.02, 173.85 | 62.46, 62.16 | 1.95 s |
| AutoMultivariateNormal | 173.79, 174.04 | 54.80, 54.77 | 1.85 s |
| Gaussian by Cholesky row | 173.76, 173.87 | 54.53, 54.32 | 1.90 s |

Lower negative ELBO is better for the unchanged target. Monte Carlo standard
errors and individual results are retained. Small differences between full
and structured guides are not evidence of a statistically meaningful winner.
All these runs and held-out arrays were finite. Warm timings reuse a compiled
executable; first calls, including compilation, are recorded separately.
These are coarse local CPU costs, with other validation processes sharing
the machine, not controlled general speedup claims or GPU results.

The short MAP centre materially handicaps frozen-mean guides. With 3,000
MAP updates on VAR2D, held-out negative ELBOs were about 66.02/65.97 for the
frozen diagonal, 53.65/53.98 for the trainable diagonal, and 41.40/41.39 for
the structured Gaussian. Raw MAP gradient norms and initializers are
recorded; 3,000 updates are not a certificate of MAP convergence. The
initial large improvement from freeing means should not be generalized to
a fully optimized historical MAP fit. Initialization and covariance both
still matter, and no candidate is selected only because it maximizes ELBO.

## PSD uncertainty and reference limitations

The held-out comparison includes 90% intervals, medians and standardized
Wasserstein distances for PSDs at 0.05, 0.2 and 0.4 cycles/sample, integrated
PSD power for every channel, and VAR2D coherence at the same frequencies.
Physical PSD scaling is applied consistently to every candidate. Arrays and
functional labels are saved, so uncertainty is inspectable beyond coefficient
means or a scalar ELBO.

A noncentered NUTS implementation preserves the same regularized-shrinkage
prior, including use of the bounded quantity as a Normal **scale**. Its
change-of-variables checks differ by less than `8e-7`, consistent with the
inherited float32 prior constants. Noncentering changes sampling coordinates,
not the target PSD model.

The NUTS references **did not qualify as trusted gold standards**:

- Initial AR(1): 27 divergences; a bounded refinement with four chains,
  2,000 warmup and 2,000 draws per chain at target acceptance 0.99 still had
  19 divergences. Functional R-hat was at most 1.0093 and ESS at least 420,
  but residual divergences prevent a clean validation claim.
- VAR2D: no divergences, but 1,983 of 2,000 retained trajectories reached
  depth 10. Original log-coordinate parameter R-hat reached 1.063 and ESS
  fell to 40; functional R-hat reached 1.0123. This was a roughly 174-second
  control including compilation. Longer or better-parameterized reference
  work is necessary.

Consequently, the saved NUTS-relative functional errors are diagnostic only.
For example, the richer VAR2D guides have median interval-width ratios
around 1.18–1.24 relative to that MCMC control, and some median PSD errors
remain 16–22%. Lower ELBO alone has not established more accurate individual
PSD posteriors. No divergent draws or failed diagnostics were hidden or
filtered away to create a reference.

The earlier [conditional quadrature/calibration screen](VI_DIAGNOSIS.md)
remains the cleaner reference: it showed strong intercept/slope correlation,
diagonal undercoverage, and much better full-covariance posterior moments.
It fixes curvature and shrinkage nuisance parameters and does **not** prove
calibration of these full SGVB guides. Full-model prior-predictive SBC and
qualified independent references remain required.

## Recommended direction and stronger alternatives

1. Keep the faithful JAX guide as a regression baseline, including the
   optional scale-family port. Use NumPyro to optimize it; the manual engine
   control gives no material reason to maintain duplicate optimization code.
2. Develop a custom Gaussian by Cholesky row, with trainable means and
   deliberate noncentered/whitened shrinkage coordinates. It captures the
   covariance missing from the default guide while exploiting the current
   target's factorization. The prototype is research code, not yet a public
   calibrated inference mode.
3. Use covariance-form low rank when the dense blocks become too large.
   Rank is an accuracy/cost choice; low rank was not faster than dense at
   this small 63-dimensional size. Preserve the distinction from TFP's scale
   update in names, tests, and migration documentation.
4. If non-Gaussian geometry remains important, evaluate flows or annealed
   HMC-based variational methods after fixing parameterization. NumPyro
   supports custom guides, full/low-rank Gaussian guides, IAF/BNAF flows,
   and DAIS; it is not one approximation family.
   [Official guide overview](https://num.pyro.ai/en/stable/autoguide.html).
5. Laplace or Pathfinder can supply covariance-aware initial proposals.
   Multi-path Pathfinder adds importance resampling across approximations;
   it is an alternative worth testing, not evidence of calibration here.
   [Stan's Pathfinder documentation](https://mc-stan.org/docs/reference-manual/pathfinder.html).
   Retain NUTS or another independently diagnosed sampler for scientific
   controls rather than treating a fast Gaussian approximation as ground truth.

## Runtime change and remaining scientific capabilities

`PSDEstimator` now defaults to JAX, lazily importing the old backend only
for `backend="tensorflow"`. TensorFlow/TFP/tf-keras are removed from normal
dependencies and live in the optional `[legacy]` extra. Python 3.10+ is
required; local validation uses 3.12. Logging and shared test fixtures no
longer force a TensorFlow import. The old `[jax]` extra remains an alias.

A fresh `.venv-jaxonly` installed from project-local cache contains **no
TensorFlow, TFP or tf-keras**. It passed **27 tests**, with five explicitly
skipped legacy-reference modules. The full comparison environment passed
**65 tests**, with one expected missing-extra test skipped. A subprocess
test forbids all legacy imports while importing and running the public
default estimator. Metadata tests also check normal dependencies.

This is a JAX-only **default runtime**, not complete replacement of every
scientific mode. Joint and factorized eigenbasis inference still require
the optional legacy backend and explicit selection; unsupported JAX mode
requests fail clearly. The original-joint 1D/multivariate model, PSD/coherence
postprocessing and benchmark CLI run without TensorFlow. AR integration
still requires provenance clearance. Scientific normalization/prior-scale
questions and the calibration limitations remain separate from dependency
cleanup. No remote publication was retried during this investigation.

Reproduction: [guide experiment](compare_guides.py),
[shared target and row grouping](guide_tools.py),
[NUTS control](guide_reference.py),
[guide results](guide-comparison/guide-results.json),
[longer MAP control](guide-comparison/long-map-results.json),
[reference diagnostics](guide-comparison/reference-results.json),
[full test log](guide-comparison/full-tests-final.log),
[JAX-only test log](guide-comparison/jax-only-tests-final.log).
