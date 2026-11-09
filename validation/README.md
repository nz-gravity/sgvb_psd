# Validation and local experiment outputs

The repository tracks validation **scripts and concise summaries**. Raw
experiment outputs (`.npz`, per-run JSON, logs, plots, CSVs, environments)
are local ignored files. Do not commit them. `summaries/` contains deliberately
retained small aggregate records, not raw simulation dumps.

The supported implementation and tests use JAX/NumPyro only. No TensorFlow
backend, dependency group, reference package, skipped legacy test suite or
TF comparison job is maintained. Historical comparison numbers below retain
their provenance; they do not introduce a runtime or development dependency.

## Reproduce

From the repository root (Python 3.12 used in the recorded experiments):

```bash
uv sync --locked --extra dev
MPLBACKEND=Agg uv run --no-sync pytest -q
# Each command writes ignored outputs under validation/:
MPLBACKEND=Agg uv run --no-sync python validation/benchmark_upgrade.py
MPLBACKEND=Agg uv run --no-sync python validation/calibration_screen.py
MPLBACKEND=Agg uv run --no-sync python validation/guide_reference.py
MPLBACKEND=Agg uv run --no-sync python validation/compare_guides.py
MPLBACKEND=Agg uv run --no-sync python validation/summarize_guides.py
```

The guide/NUTS comparisons are research experiments, not production guide
selectors or qualified posterior references. See [GUIDE_COMPARISON.md](GUIDE_COMPARISON.md).
For the full-production fixed-truth coverage campaign:

```bash
MPLBACKEND=Agg uv run --no-sync python validation/var2_coverage.py freeze
MPLBACKEND=Agg uv run --no-sync python validation/var2_coverage.py run
MPLBACKEND=Agg uv run --no-sync python validation/analyze_var2_coverage.py
MPLBACKEND=Agg uv run --no-sync python validation/check_var2_coverage.py
```

Freeze once before evaluating any fit. `run` resumes pending entries, keeps
failures and never retries terminal outcomes. Stop an active coordinator
before resuming. `freeze` refuses an existing configuration. Preserve an old
output directory before starting a new campaign in a separate checkout;
never tune against the final evaluation seeds.

## Existing local evidence and Git history

Before cleanup, 581 validation/test files (27,979,924 bytes) were copied and
SHA256-verified into `.local-validation-backup/583e4f3/` in the UoA checkout.
`backup-manifest.json` records the original revision, file sizes and hashes.
Subsequent concurrent uv edits were also preserved under `concurrent-edits/`.
The generated outputs remain in their original ignored local directories;
the additional verified backup is ignored too.

Untracking cleans the current tree and PR file diff. Earlier commits still
contain the old blobs, accessible through commit `583e4f3`. No history rewrite
or force-push was performed; clone history will not shrink as a result.
Current summaries are historical evidence from that campaign, not fresh
results from a rerun after cleanup. Logs and large figures can be shared
separately when needed; reproducing an experiment regenerates them locally.

[Migration](JAX_MIGRATION.md) · [guide comparison](GUIDE_COMPARISON.md) ·
[VI diagnosis](VI_DIAGNOSIS.md) · [performance](UPGRADE_BENCHMARK.md) ·
[VAR2 coverage](VAR2_COVERAGE.md).

## Cleanup checks

The cleaned tree passed **55 JAX-only tests**, with no skipped legacy modules,
in 89.12 s. Existing plotting/degenerate-MAD warnings were retained in the
local test log. Source code uses no TensorFlow imports and the uv lockfile
contains no TF/TFP/tf-keras package. Package and benchmark smoke checks are
recorded locally; generated logs remain ignored.

[Compact cleanup verification](summaries/cleanup-verification.json) records
current test/package/lock/benchmark results. No raw scientific campaign was
rerun during this cleanup.
