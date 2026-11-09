"""JAX-only first-fit and warmed-fit costs; writes ignored local JSON."""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from sgvb_psd.psd_estimator import PSDEstimator
from simulation_cases import cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=["ar1", "var2", "all"], default="all"
    )
    parser.add_argument("--map-steps", type=int, default=1000)
    parser.add_argument("--vi-steps", type=int, default=5000)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "upgrade-benchmark/jax-only-results.json",
    )
    a = parser.parse_args()
    if a.repeats < 1:
        parser.error("--repeats must be positive")
    results = {
        "backend": "jax",
        "version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "scope": "first fit includes JIT; warm reuses the same prepared runner/data; not an accuracy or calibration benchmark",
        "cases": {},
    }
    for name, raw, fs in cases():
        if a.case not in ("all", name):
            continue
        estimator = PSDEstimator(
            raw,
            fs=fs,
            N_theta=6,
            nchunks=2,
            ntrain_map=a.map_steps,
            n_elbo_maximisation_steps=a.vi_steps,
            N_samples=a.draws,
            seed=0,
        )
        runner = estimator.inference_runner
        initial = tuple(
            np.asarray(p).copy() for p in runner.model.trainable_vars
        )
        fits = []
        for i in range(a.repeats):
            runner.model.trainable_vars = tuple(
                jnp.asarray(p) for p in initial
            )
            runner.rng_key = jax.random.PRNGKey(0)
            start = time.perf_counter()
            outputs = estimator.run(lr=0.003)
            elapsed = time.perf_counter() - start
            fits.append(
                {
                    "repeat": i,
                    "fit_s": elapsed,
                    "map_s": runner.map_time,
                    "vi_s": runner.vi_time,
                    "finite": all(
                        bool(np.all(np.isfinite(x))) for x in outputs
                    ),
                }
            )
        results["cases"][name] = {
            "fits": fits,
            "map_steps": a.map_steps,
            "vi_steps": a.vi_steps,
            "draws": a.draws,
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
