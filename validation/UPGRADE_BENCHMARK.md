# Performance evidence and JAX-only reproduction

The supported package, tests and benchmark use JAX/NumPyro only. Current
`benchmark_upgrade.py` measures first-fit/JIT and warm repeated-fit costs,
resetting parameters/RNG on the same prepared runner. It writes ignored
local JSON. It is not an accuracy or calibration benchmark.

```bash
uv sync --locked --extra dev
MPLBACKEND=Agg uv run --no-sync python validation/benchmark_upgrade.py
```

## Historical cross-engine evidence

These results were collected before removing the TensorFlow reference code,
on UOA484808, macOS arm64 CPU, Python 3.12.16, JAX .6.2/NumPyro .19 versus
TF 2.20/TFP .25. Two fresh process trials and four warm repeats per backend
used MAP 1,000 / VI 5,000 steps, fixed MAP means and the same target.
The old code/evidence remains in Git history at `583e4f3`, not supported by
current dependencies or CI. [Compact record](summaries/historical-performance.json).

| Case | JAX cold / TF cold | JAX warm / TF warm |
|---|---:|---:|
| AR1 n512 | 2.343 / 6.889 s | .345 / 3.588 s |
| VAR2 n512 | 3.417 / 6.995 s | .791 / 3.694 s |
| VAR2 n8192 | 5.367 / 8.138 s | 2.733 / 4.839 s |

Cold included imports/preparation/JIT/inference/summaries. Warm reused one
prepared runner; historical TF methods recreated functions. This is not a
comparison against an optimized TF/XLA ceiling or a GPU benchmark. The 1D
historical public estimator required a lower-level adapter.

Functional medians differed <.46%, but 90% widths differed up to 29% and
some held-out objective differences exceeded evaluation MC error. Equal
full-model accuracy/calibration is not established. A 64-dataset analytic
conditional full-Gaussian control passed the same quadrature accuracy gates
in both engines, with 5.17x cold / 7.78x warm gain, but was not full SGVB.

The [production coverage study](VAR2_COVERAGE.md) records undercoverage and
optimization concerns. Local raw results and a verified backup are retained;
large files, plots and logs are not tracked. [Artifact policy](README.md).
