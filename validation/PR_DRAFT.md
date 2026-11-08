Title: Upgrade SGVB to a JAX-only runtime with 1D support and measured parity

Replace the installed TensorFlow runtime with JAX/NumPyro for original-joint
1D and multivariate SGVB. Preserve the blocked Whittle target, priors and
PSD conventions, with one target value per variational draw. The default
guide keeps means at MAP as in TF; `optimize_mean=True` explicitly trains
them. Cache compiled MAP/VI functions per prepared runner, passing current
means and learning rates dynamically.

This is a breaking upgrade: TensorFlow dependencies and the legacy extra
are removed. Historical MIT source lives only in `validation/reference`,
excluded from the installed package/wheel. Eigenbasis joint/factorized JAX
ports are unfinished and requests fail explicitly. AR code from
`cleared_unisgvb` has not been copied because licensing remains unresolved.

Validation: 68 tests pass with the archived reference installed; 29 pass
and five reference modules skip in a fresh TensorFlow-free environment.
An offline wheel has no TF imports/dependencies or reference code, and
passes public 1D/VAR2D smoke inference with TF imports forbidden.

On the UoA CPU, measured full-pipeline first-result costs improve 1.52–2.94×
and repeated fits 1.77–10.41× across three cases. Functional medians differ
<.46%, but interval widths differ up to 29%; these are not equal-accuracy
or full-calibration claims. A separate 64-dataset conditional full-Gaussian
control passes identical quadrature accuracy gates in both engines, with
matching coverage counts and 5.17× cold/7.78× warm speedup. It is not full SGVB.
Protocol, raw artifacts, limits and commands: `validation/UPGRADE_BENCHMARK.md`.

The earlier ~30% discrepancy came from trainable versus frozen guide means.
Matched-noise target/gradient checks and richer custom/NumPyro guide
comparisons are in `VI_DIAGNOSIS.md` and `GUIDE_COMPARISON.md`. The inherited
diagonal approximation has conditional undercoverage. Covariance-aware
production inference and full-model SBC remain open. CI configuration is
updated; no remote CI result or automatic main merge is claimed.
