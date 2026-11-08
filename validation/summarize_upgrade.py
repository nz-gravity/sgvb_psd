"""Summarize all benchmark trials; retain quality differences, not just speed."""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "upgrade-benchmark"


def read(name):
    return json.loads((OUT / name).read_text())


def timing(rows, workers):
    result = {}
    for backend in ("jax", "tf"):
        cold = [
            r["first_result_wall_s"] for r in rows if r["backend"] == backend
        ]
        warm = [
            f["fit_s"]
            for w in workers
            if w["backend"] == backend
            for f in w["fits"][1:]
        ]
        result[backend] = {
            "cold_seconds": cold,
            "warm_seconds": warm,
            "cold_median_s": float(np.median(cold)),
            "warm_median_s": float(np.median(warm)),
        }
    for kind in ("cold", "warm"):
        result[kind + "_tf_over_jax"] = (
            result["tf"][kind + "_median_s"]
            / result["jax"][kind + "_median_s"]
        )
    return result


def main():
    summary = {"full_model_calibration_validated": False, "cases": {}}
    processes = read("processes.json")
    for case in read("cases.json"):
        workers = [
            read(f"{case}-{backend}-{seed}.json")
            for backend in ("jax", "tf")
            for seed in (0, 1)
        ]
        comparisons = []
        for seed in (0, 1):
            a = read(f"{case}-jax-{seed}.json")["fits"][0]
            b = read(f"{case}-tf-{seed}.json")["fits"][0]
            qa, qb = np.array(a["functional_quantiles"]), np.array(
                b["functional_quantiles"]
            )
            rel = abs(qa[1] / qb[1] - 1)
            widths = (qa[2] - qa[0]) / (qb[2] - qb[0])
            with (
                np.load(OUT / a["artifact"]) as aa,
                np.load(OUT / b["artifact"]) as bb,
            ):
                loss_diff = aa["losses"] - bb["losses"]
                map_diffs = {
                    name: float(np.max(abs(aa[name] - bb[name]), initial=0))
                    for name in aa.files
                    if name.startswith("map_")
                }
            comparisons.append(
                {
                    "seed": seed,
                    "functional_names": a["functional_names"],
                    "functional_median_relative_difference": rel.tolist(),
                    "functional_width_ratio_jax_over_tf": widths.tolist(),
                    "max_functional_median_relative_difference": float(
                        max(rel)
                    ),
                    "functional_width_ratio_range": [
                        float(min(widths)),
                        float(max(widths)),
                    ],
                    "negative_elbo_jax": a["negative_elbo"],
                    "negative_elbo_tf": b["negative_elbo"],
                    "paired_negative_elbo_difference_jax_minus_tf": float(
                        np.mean(loss_diff)
                    ),
                    "paired_difference_mc_se": float(
                        np.std(loss_diff, ddof=1) / np.sqrt(len(loss_diff))
                    ),
                    "map_max_absolute_differences": map_diffs,
                    # Agreement is evidence, not an oracle accuracy or SBC test.
                    "full_model_accuracy_equivalence_established": False,
                }
            )
        summary["cases"][case] = {
            "timing": timing(
                [r for r in processes if r["case"] == case], workers
            ),
            "comparisons": comparisons,
            "all_outputs_finite": all(
                f["finite"] for w in workers for f in w["fits"]
            ),
            "jax_workers_imported_tensorflow": any(
                w["loaded_tensorflow"]
                for w in workers
                if w["backend"] == "jax"
            ),
            "tf_workers_imported_jax": any(
                w["loaded_jax"] for w in workers if w["backend"] == "tf"
            ),
        }
    cond = [
        read(f"conditional-{b}-{r}.json")
        for b in ("jax", "tf")
        for r in (0, 1)
    ]
    summary["conditional_control"] = {
        "timing": timing(read("conditional-processes.json"), cond),
        "all_accuracy_gates_pass": all(
            f["conditional_accuracy_gate"] for w in cond for f in w["fits"]
        ),
        "max_elbo_gap": max(
            f["negative_elbo_gap_max"] for w in cond for f in w["fits"]
        ),
        "max_mean_error_oracle_sd": max(
            f["posterior_mean_error_max_in_oracle_sd"]
            for w in cond
            for f in w["fits"]
        ),
        "max_covariance_relative_error": max(
            f["posterior_covariance_relative_error_max"]
            for w in cond
            for f in w["fits"]
        ),
        "nominal90_coverage_counts": cond[0]["fits"][0][
            "nominal90_coverage_counts"
        ],
        "all_coverage_counts_equal": all(
            f["nominal90_coverage_counts"]
            == cond[0]["fits"][0]["nominal90_coverage_counts"]
            for w in cond
            for f in w["fits"]
        ),
        "full_sgvb_calibration_validated": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
