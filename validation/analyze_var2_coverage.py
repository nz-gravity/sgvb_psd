"""Summarize independent-realization coverage without pooling frequencies."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from var2_coverage import NAMES, OUT, load, dump, wilson


def export_figure(fig, filename):
    """Export standalone figures with clean generated SVG whitespace."""
    path = OUT / (filename + ".svg")
    fig.savefig(path)
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines())
        + "\n"
    )
    fig.savefig(OUT / (filename + ".pdf"))
    fig.savefig(OUT / (filename + ".png"), dpi=130)


def summarize(
    covered,
    widths,
    errors,
    normalized_widths,
    normalized_errors,
    frequency,
    total_planned,
    attempted,
    edge_bins=5,
):
    n = len(covered)
    successes = np.sum(covered, axis=0)
    coverage = successes / n
    intervals = wilson(successes, n)
    binomial_se = np.sqrt(coverage * (1 - coverage) / n)
    freq_average = covered.mean(axis=1)
    interior = covered[:, edge_bins:-edge_bins].mean(axis=1)
    edge = np.concatenate(
        [covered[:, :edge_bins], covered[:, -edge_bins:]], axis=1
    ).mean(axis=1)
    summaries = []
    for k, name in enumerate(NAMES):
        summaries.append(
            {
                "quantity": name,
                "frequency_average_coverage": float(freq_average[:, k].mean()),
                "frequency_average_coverage_realization_se": (
                    float(freq_average[:, k].std(ddof=1) / np.sqrt(n))
                    if n > 1
                    else None
                ),
                "interior_frequency_average_coverage": float(
                    interior[:, k].mean()
                ),
                "edge_frequency_average_coverage": float(edge[:, k].mean()),
                "pointwise_coverage_min": float(coverage[:, k].min()),
                "pointwise_coverage_max": float(coverage[:, k].max()),
                "median_width_over_realizations_and_frequencies": float(
                    np.median(widths[:, :, k])
                ),
                "median_normalized_width": float(
                    np.median(normalized_widths[:, :, k])
                ),
                "mean_absolute_recovery_error": float(
                    np.mean(abs(errors[:, :, k]))
                ),
                "root_mean_squared_normalized_recovery_error": float(
                    np.sqrt(np.mean(normalized_errors[:, :, k] ** 2))
                ),
            }
        )
    arrays = {
        "frequency_cycles": frequency,
        "successes": successes,
        "coverage_completed": coverage,
        "coverage_wilson95": intervals,
        "coverage_binomial_se": binomial_se,
        "coverage_all_planned_lower": successes / total_planned,
        "coverage_all_planned_upper": (successes + total_planned - n)
        / total_planned,
        "coverage_attempted_lower": successes / attempted,
        "coverage_attempted_upper": (successes + attempted - n) / attempted,
        "width_quantiles_over_realizations": np.quantile(
            widths, [0.1, 0.5, 0.9], axis=0
        ),
        "normalized_width_quantiles_over_realizations": np.quantile(
            normalized_widths, [0.1, 0.5, 0.9], axis=0
        ),
        "recovery_mean_error": np.mean(errors, axis=0),
        "recovery_mae": np.mean(abs(errors), axis=0),
        "normalized_recovery_rmse": np.sqrt(
            np.mean(normalized_errors**2, axis=0)
        ),
        "realization_frequency_average_coverage": freq_average,
    }
    return summaries, arrays


def five_panels(
    filename, title, ylabel, lines, bands=None, target=None, percent=False
):
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
    colors = ["#235789", "#d1495b", "#317f43"]
    for k, ax in enumerate(axes.flat[:5]):
        for i, (x, y, label) in enumerate(lines):
            ax.plot(
                x, y[:, k], label=label, color=colors[i % len(colors)], lw=1.6
            )
        if bands:
            x, lo, hi, label = bands
            ax.fill_between(
                x, lo[:, k], hi[:, k], color=colors[0], alpha=0.18, label=label
            )
        if target is not None:
            ax.axhline(target, color="black", ls="--", lw=1)
        if percent:
            ax.set_ylim(0, 1.02)
        ax.axvspan(0, 5 / 256, color="gray", alpha=0.1)
        ax.axvspan(0.5 - 5 / 256, 0.5, color="gray", alpha=0.1)
        ax.set_title(NAMES[k])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 0.5)
        ax.set_xlabel("Frequency (cycles/sample)")
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc="upper left", frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    export_figure(fig, filename)
    plt.close(fig)


def main():
    config = load(OUT / "config.json")
    manifest = load(OUT / "manifest.json")
    ids = [r["index"] for r in manifest["runs"] if r["status"] == "completed"]
    results = [load(OUT / f"run-{i:03d}.json") for i in ids]
    counts = {
        "planned": config["realizations"],
        "attempted": manifest["attempted"],
        "completed": len(ids),
        "failed_or_interrupted": manifest["failed"],
        "pending": sum(r["status"] == "pending" for r in manifest["runs"]),
        "running": sum(r["status"] == "running" for r in manifest["runs"]),
    }
    if not ids:
        dump(
            OUT / "summary.json",
            {"counts": counts, "coverage_estimated": False},
        )
        return
    collected = {
        key: []
        for key in (
            "covered",
            "width",
            "error",
            "normalized_width",
            "normalized_error",
        )
    }
    with np.load(OUT / "truth.npz") as reference:
        reference_truth = reference["functionals"].copy()
        reference_frequency = reference["frequency_cycles"].copy()
    map_final_minus_max = []
    for i in ids:
        with np.load(OUT / f"run-{i:03d}.npz") as row:
            map_trace = row["map_log_posterior"]
            map_final_minus_max.append(float(map_trace[-1] - map_trace.max()))
            frequency = row["frequency_cycles"]
            np.testing.assert_allclose(
                frequency, reference_frequency, atol=0, rtol=0
            )
            np.testing.assert_allclose(
                row["truth"], reference_truth, atol=1e-14
            )
            independently_covered = (
                row["quantiles"][0] <= reference_truth
            ) & (reference_truth <= row["quantiles"][2])
            np.testing.assert_array_equal(
                row["covered"], independently_covered
            )
            for key in collected:
                collected[key].append(
                    independently_covered if key == "covered" else row[key]
                )
    collected = {k: np.array(v) for k, v in collected.items()}
    summaries, arrays = summarize(
        collected["covered"],
        collected["width"],
        collected["error"],
        collected["normalized_width"],
        collected["normalized_error"],
        frequency,
        config["realizations"],
        manifest["attempted"],
        config["edge_bins_each_end"],
    )
    np.savez_compressed(
        OUT / "frequency-results.npz",
        completed_indices=np.array(ids),
        **arrays,
    )
    with (OUT / "frequency-results.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            [
                "frequency_cycles_per_sample",
                "frequency_physical",
                "quantity",
                "completed",
                "successes",
                "coverage",
                "wilson95_low",
                "wilson95_high",
                "binomial_se",
                "median_width",
                "mean_absolute_error",
                "all_attempted_coverage_lower",
                "all_attempted_coverage_upper",
            ]
        )
        for f, frequency_value in enumerate(frequency):
            for k, name in enumerate(NAMES):
                writer.writerow(
                    [
                        frequency_value,
                        frequency_value * config["fs"],
                        name,
                        len(ids),
                        int(arrays["successes"][f, k]),
                        arrays["coverage_completed"][f, k],
                        *arrays["coverage_wilson95"][f, k],
                        arrays["coverage_binomial_se"][f, k],
                        arrays["width_quantiles_over_realizations"][1, f, k],
                        arrays["recovery_mae"][f, k],
                        arrays["coverage_attempted_lower"][f, k],
                        arrays["coverage_attempted_upper"][f, k],
                    ]
                )
    cost = [
        r["process_wall_s"] for r in manifest["runs"] if "process_wall_s" in r
    ]
    stability = {}
    metrics = [
        "fit_s",
        "map_s",
        "vi_s",
        "worker_s",
        "map_last_500_change",
        "map_gradient_rms",
        "map_gradient_abs_max",
        "vi_last_1000_std",
        "guide_scale_min",
        "guide_scale_max",
        "guide_scale_median",
    ]
    for metric in metrics:
        values = np.array([r[metric] for r in results])
        stability[metric] = {
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }
    drift = np.array(
        [r["vi_last_1000_mean"] - r["vi_previous_1000_mean"] for r in results]
    )
    stability["vi_last_vs_previous_1000_mean_difference"] = {
        "min": float(drift.min()),
        "median": float(np.median(drift)),
        "max": float(drift.max()),
    }
    stability["map_last_500_negative_change_count"] = sum(
        r["map_last_500_change"] < 0 for r in results
    )
    stability["map_last_logged_minus_trace_max"] = {
        "min": min(map_final_minus_max),
        "median": float(np.median(map_final_minus_max)),
        "max": max(map_final_minus_max),
    }
    summary = {
        "counts": counts,
        "quantity_summaries": summaries,
        "stability": stability,
        "all_completed_outputs_and_traces_finite": all(
            r["finite"] for r in results
        ),
        "nominal_coverage": 0.9,
        "nominal_binomial_se_at_n100": 0.03,
        "nominal90_wilson95_if_90_of_100": wilson(np.array(90), 100).tolist(),
        "interval_type": "90% equal-tail pointwise marginal intervals, separate real/imaginary CSD; NOT uniform intervals",
        "frequencywise_uncertainty": "Wilson 95% per-frequency intervals across independent realizations; bands not simultaneous",
        "averaged_uncertainty": "SE of per-realization frequency-average coverage across independent realizations; no frequency pooling",
        "not_prior_predictive_sbc": True,
        "not_simultaneous_coverage": True,
        "cost": {
            "attempted_process_wall_seconds_sum": float(sum(cost)),
            "process_seconds_median": float(np.median(cost)),
            "process_seconds_max": float(max(cost)),
        },
        "versions": results[0]["versions"],
        "jax_devices": results[0]["jax_devices"],
        "tensorflow_imported_any": any(
            r["tensorflow_imported"] for r in results
        ),
        "failures": [
            r
            for r in manifest["runs"]
            if r["status"] not in ("completed", "pending", "running")
        ],
    }
    dump(OUT / "summary.json", summary)
    five_panels(
        "coverage",
        "VAR2 fixed-truth pointwise coverage: "
        + str(len(ids))
        + " completed realizations",
        "Coverage",
        [(frequency, arrays["coverage_completed"], "Observed coverage")],
        bands=(
            frequency,
            arrays["coverage_wilson95"][:, :, 0],
            arrays["coverage_wilson95"][:, :, 1],
            "Pointwise Wilson 95%",
        ),
        target=0.9,
        percent=True,
    )
    widths = arrays["normalized_width_quantiles_over_realizations"]
    five_panels(
        "widths",
        "90% interval widths: PSD/true PSD; CSD/sqrt(S00 S11); coherence absolute",
        "Normalized interval width",
        [(frequency, widths[1], "Median across realizations")],
        bands=(frequency, widths[0], widths[2], "10–90% across realizations"),
    )
    native = arrays["width_quantiles_over_realizations"]
    five_panels(
        "widths-native",
        "90% interval widths in physical density units; coherence dimensionless",
        "Interval width",
        [(frequency, native[1], "Median across realizations")],
        bands=(frequency, native[0], native[2], "10–90% across realizations"),
    )
    five_panels(
        "recovery",
        "Posterior median recovery: PSD/true PSD; CSD/sqrt(S00 S11); coherence absolute",
        "Normalized RMSE",
        [
            (
                frequency,
                arrays["normalized_recovery_rmse"],
                "RMSE across realizations",
            )
        ],
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric, label in zip(
        axes.flat[:4],
        [
            "map_last_500_change",
            "map_gradient_rms",
            "vi_last_1000_std",
            "guide_scale_min",
        ],
        [
            "MAP log posterior change over last 500",
            "MAP raw gradient RMS (coordinate dependent)",
            "VI last 1,000 loss SD",
            "Smallest guide scale",
        ],
    ):
        ax.plot(ids, [r[metric] for r in results], ".")
        ax.set_title(label)
        ax.set_xlabel("Realization index")
        ax.grid(alpha=0.2)
    axes.flat[4].plot(ids, drift, ".")
    axes.flat[4].axhline(0, color="black", lw=1)
    axes.flat[4].set_title("VI last vs previous 1,000 mean loss")
    axes.flat[4].set_xlabel("Realization index")
    axes.flat[5].plot(ids, [r["fit_s"] for r in results], ".")
    axes.flat[5].set_title("Full fit wall seconds")
    axes.flat[5].set_xlabel("Realization index")
    fig.suptitle(
        "Optimizer diagnostics: finite is not a convergence certificate"
    )
    fig.tight_layout()
    export_figure(fig, "stability")
    plt.close(fig)
    # Representative traces are selected by index, never by fit quality.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index in [i for i in (0, 49, 99) if i in ids]:
        with np.load(OUT / f"run-{index:03d}.npz") as row:
            axes[0].plot(row["map_log_posterior"], label=f"Run {index}")
            losses = row["vi_negative_elbo"]
            batches = (
                losses[: len(losses) // 100 * 100].reshape(-1, 100).mean(1)
            )
            axes[1].plot(
                np.arange(len(batches)) * 100 + 50,
                batches,
                label=f"Run {index}",
            )
    axes[0].set_title("MAP log posterior")
    axes[1].set_title("VI negative ELBO: 100-step means")
    for ax in axes:
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    export_figure(fig, "traces")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
