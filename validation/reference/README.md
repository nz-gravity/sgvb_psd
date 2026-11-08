# Isolated TensorFlow reference

This is checkout-only validation code, excluded from the installed package
and wheel. The runtime has no TensorFlow dependency or `legacy` extra.

Source provenance: MIT-licensed `nz-gravity/sgvb_psd`, base revision
`8b0f1f59241b0e5918c962212a4516123d8c8131`. The backend was moved here from
the migration branch; the original estimator was restored from that base.
Relative imports now use the `sgvb_tf` namespace. The pure NumPy Fourier
and interval helpers use the shared, formula-preserving runtime extraction.
Historical plotting and simulation utilities are included so importing the
reference does not import JAX. Its MIT license is included. No source from
`cleared_unisgvb` is included.

The reference retains original joint, eigenbasis joint and factorized
inference. Their presence here does **not** imply a validated JAX port.
It also retains historical limitations, including the public one-channel
input rejection and scalar multi-draw likelihood reduction. The 1D benchmark
uses the lower-level model adapter documented in `benchmark_upgrade.py`.

For parity checks in a development environment:

```bash
pip install -e '.[dev]' -c validation/jax-constraints.txt
pip install -r validation/reference/requirements.txt
MPLBACKEND=Agg pytest -q
```

Tests explicitly opt into the reference namespace in `tests/conftest.py`.
Validation scripts opt in with `reference_bootstrap.py`. No runtime import
automatically selects it. A separate environment with only
`requirements.txt` and `PYTHONPATH=src:validation/reference` can run the
reference without installing JAX. See `UPGRADE_BENCHMARK.md` one level up for
the isolated timing environments and commands.
