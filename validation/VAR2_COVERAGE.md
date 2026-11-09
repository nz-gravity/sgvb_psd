# Full-production VAR2 fixed-truth pointwise coverage

**100 planned, attempted and completed; no failures, timeouts or retries.**
The configuration/seeds were frozen before evaluation. This is fixed-truth
frequentist coverage, **not prior-predictive SBC or simultaneous coverage**.
[Compact protocol, normalization and results](summaries/var2-coverage.json).

The model is the existing test VAR(2): A1=diag(.5,−.3), A2=diag(0,−.5),
innovation covariance [[1,.9],[.9,1]], using `SimVARMA` initialization.
All 63 production latent coordinates are unknown; this is not the earlier
conditional two-parameter control. Each of 100 datasets has 512 samples,
two 256-sample blocks, six nonlinear spline terms, fs=2π and 127 positive
bins excluding DC/Nyquist. The guide is independent Normal with MAP means
frozen, softplus scales initially 1e-4, one particle. MAP uses 5,000 steps
at .003; VI uses 10,000 at .05; intervals use 2,000 posterior draws and
5/95% scalar quantiles. Priors/dtypes/preprocessing are unchanged.

Independent deterministic data/VI seeds use SeedSequence namespaces 1101
and 2202 with master seed 20261109. Each process has a 120-second timeout.
No tuning was performed against the final outcomes. No secondary guide was
prespecified; structured/full covariance is not a production option yet.
Frozen implementation: `927c9b5078a729a5c3f4d00e96a7ead878a5813f`;
raw campaign/evidence snapshot: `583e4f3`.

## Results and uncertainty

| Quantity | Frequency-average coverage ± realization SE | Per-frequency range |
|---|---:|---:|
| PSD00 | 71.6% ± 2.3 pp | 47–94% |
| PSD11 | 62.0% ± 1.1 pp | 13–89% |
| Re CSD01 | 65.5% ± 1.5 pp | 18–89% |
| Im CSD01 | 53.0% ± 1.3 pp | 9–96% |
| Coherence | 68.6% ± 1.6 pp | 49–85% |

Each frequency has 100 independent realization trials. Near 90%, binomial
SE is about 3 pp; 90/100 has Wilson 95% interval [.8256,.9448]. Frequency-wise
bands are not simultaneous bands. Frequency averages and their SE use
per-realization averages; the dependent 127 bins are never pooled as
independent trials. CSD real/imaginary intervals are separate scalar checks,
not a joint complex region. Coherence truth is .81 across the grid.

The first/last five retained bins are flagged as edges. Interior averages
are 72.9%, 60.9%, 65.5%, 51.8%, 67.5%, so undercoverage is not edge-only.
Median interval widths in native density units (coherence dimensionless)
are .04375, .05315, .04108, .03565, .10094. Normalized recovery RMSE is
.1501, .2054, .1398, .1201, .05657; diagonals divide by true PSD, CSD by
sqrt(S00 S11), and coherence uses absolute units. Width/error summaries are
descriptive, not pooled binomial trials.

32/100 late MAP traces decline; final-window objective changes range
−72.62 to +114.43, and raw-coordinate gradient RMS has median 325.9.
Finite outputs do not certify convergence. This demonstrates a limitation
of this finite-budget pipeline; it does not isolate guide family, prior,
optimization, basis approximation or a JAX regression as the cause.

## Normalization and costs

S(f)=H(f)ΣH(f)* / fs, H=(I−A1 exp(−2πif)−A2 exp(−4πif))⁻¹, using two-sided
density at positive bins without doubling. Analytic helper agreement and
full-circle covariance integration passed. Independent white-noise FFT
and constant-spectrum checks verify fft/sqrt(block length), /fs and input
std² reconstruction. The public raw draws omit std², so this campaign
restores it explicitly and cross-checks diagonal interval outputs.

No taper/demeaning was added. The exact finite rectangular-window expected
periodogram differs from the infinite-process diagonal spectrum by up to
1.356%. Boundary spline behavior and finite-block leakage remain caveats.

Fresh-worker wall costs sum to 1,042.45 s (17.37 min), median 10.28 s,
maximum 14.63 s. Median public fit time is 7.59 s, MAP .540 s, VI 1.414 s.
UOA484808, macOS 27 arm64 CPU, 18 logical CPUs, JAX .6.2/NumPyro .19,
Python 3.12.16. The original focused tests passed 32 checks before cleanup;
current JAX-only tests are run separately, not used to rerun this campaign.

The manifest, per-fit data/intervals/traces/parameters, per-frequency CSV,
Wilson bands, width/recovery/stability plots and hash inventory remain
**local ignored outputs**, with an independently verified backup. Run the
scripts in [validation instructions](README.md) to regenerate outputs.
Do not overwrite this frozen final evaluation or tune until favorable.
