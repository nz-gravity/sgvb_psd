"""Check coverage provenance and artifact integrity without rerunning any fit."""

import hashlib
from pathlib import Path

import numpy as np

from var2_coverage import OUT, ROOT, load, dump


def main():
    config = load(OUT / "config.json")
    manifest = load(OUT / "manifest.json")
    assert (
        hashlib.sha256((OUT / "config.json").read_bytes()).hexdigest()
        == manifest["config_sha256"]
    )
    assert (
        hashlib.sha256(
            (ROOT / "validation/var2_coverage.py").read_bytes()
        ).hexdigest()
        == config["script_sha256"]
    )
    assert len(manifest["runs"]) == config["realizations"] == 100
    assert (
        len({r[k] for r in manifest["runs"] for k in ("data_seed", "vi_seed")})
        == 200
    )
    terminal = [
        r
        for r in manifest["runs"]
        if r["status"] not in ("pending", "running")
    ]
    completed = [r for r in terminal if r["status"] == "completed"]
    assert len(completed) == manifest["completed"]
    assert len(terminal) - len(completed) == manifest["failed"]
    assert manifest["attempted"] == len(terminal) + sum(
        r["status"] == "running" for r in manifest["runs"]
    )
    for spec in terminal:
        i = spec["index"]
        result = load(OUT / f"run-{i:03d}.json")
        assert spec["status"] == result["status"]
        assert (OUT / f"run-{i:03d}.log").exists()
        if result["status"] != "completed":
            continue
        assert (
            result["data_seed"] == spec["data_seed"]
            and result["vi_seed"] == spec["vi_seed"]
        )
        assert not result["tensorflow_imported"]
        with np.load(OUT / f"run-{i:03d}.npz") as row:
            assert row["data"].shape == (512, 2)
            assert row["quantiles"].shape == (3, 127, 5)
            assert row["map_log_posterior"].shape == (config["map_steps"],)
            assert row["vi_negative_elbo"].shape == (config["vi_steps"],)
            assert np.all(np.isfinite(row["quantiles"]))
            expected = (row["quantiles"][0] <= row["truth"]) & (
                row["truth"] <= row["quantiles"][2]
            )
            np.testing.assert_array_equal(row["covered"], expected)
    summary = load(OUT / "summary.json")
    assert summary["counts"]["completed"] == manifest["completed"]
    index = {
        "checked_counts": summary["counts"],
        "frozen_source_revision": config["source_revision"],
        "fit_script_sha256": config["script_sha256"],
        "analysis_script_sha256": hashlib.sha256(
            (ROOT / "validation/analyze_var2_coverage.py").read_bytes()
        ).hexdigest(),
        "files": {},
    }
    for path in sorted(OUT.iterdir()):
        if (
            path.is_file()
            and path.name != "artifact-index.json"
            and not path.name.endswith(".tmp")
        ):
            index["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    dump(OUT / "artifact-index.json", index)
    print(
        "Verified",
        len(completed),
        "completed and",
        manifest["failed"],
        "failed/interrupted artifacts;",
        len(index["files"]),
        "file hashes saved; no fits rerun",
    )


if __name__ == "__main__":
    main()
