"""Rebuild compact comparison tables from saved experimental results."""

import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "guide-comparison"


def main():
    results = json.loads((OUT / "guide-results.json").read_text())
    references = json.loads((OUT / "reference-results.json").read_text())
    summary = {
        "seeds": results["seeds"],
        "steps": results["steps"],
        "particles": results["particles"],
        "cases": {},
    }
    for name, case in results["cases"].items():
        reference = references["cases"][name]
        compact = {
            "dimension": case["dimension"],
            "row_dimensions": case["row_dimensions"],
            "nuts_qualified_strict_reference": reference[
                "qualified_strict_reference"
            ],
            "nuts_artifact_sha256": hashlib.sha256(
                (OUT / (name + "-nuts.npz")).read_bytes()
            ).hexdigest(),
            "families": {},
        }
        for family, rows in case["families"].items():
            compact["families"][family] = {
                "negative_elbo": [r["negative_elbo"] for r in rows],
                "negative_elbo_mc_se": [
                    r["negative_elbo_mc_se"] for r in rows
                ],
                "cold_training_s": rows[0]["training_s"],
                "warm_training_s": rows[1]["training_s"],
                "free_parameters": rows[0]["free_parameters"],
                "training_and_evaluation_finite": all(
                    r["training_finite"] and r["evaluation_finite"]
                    for r in rows
                ),
                "median_nuts_relative_interval_width": float(
                    np.median(
                        [r["nuts_functional_width_ratios"] for r in rows]
                    )
                ),
                "maximum_nuts_relative_median_error": float(
                    np.max(
                        [
                            r["nuts_functional_median_relative_errors"]
                            for r in rows
                        ]
                    )
                ),
            }
        summary["cases"][name] = compact
    (OUT / "guide-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
