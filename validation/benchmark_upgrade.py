"""Fresh-process and repeated-fit costs with explicit posterior-agreement checks.

TF lives in a separate validation environment; the distributed runtime is
JAX-only. No network operations. Cold costs include imports; warm costs reuse
one prepared runner. Independent accuracy evaluation is excluded from timing.
"""

import time

BOOT = time.perf_counter()
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/upgrade-benchmark"


def target_and_functionals(raw, fs, chunks, basis, means, scales, seed):
    import numpy as np
    from scipy.special import expit
    from sgvb_psd._spectral import compute_chunked_fft, _compute_Xmatrices

    std = np.std(raw)
    x = raw / std
    y, freq = compute_chunked_fft(x, chunks, None, fs)
    power = (
        np.einsum("bfi,bfj->fij", y, y.conj())
        .astype(np.complex64)
        .astype(np.complex128)
    )
    xd, xt = _compute_Xmatrices(freq, basis, basis)
    xd, xt = xd.astype(np.float32).astype(float), xt.astype(np.float32).astype(
        float
    )
    rng = np.random.default_rng(1000 + seed)
    draws = [
        m + s * rng.normal(size=(2048,) + m.shape)
        for m, s in zip(means, scales)
    ]
    logq = sum(
        np.sum(
            -0.5 * ((z - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi),
            axis=(-2, -1),
        )
        for z, m, s in zip(draws, means, scales)
    )
    gamma = np.einsum("fk,spk->sfp", xd, draws[0])
    theta = np.einsum("fk,sek->sfe", xt, draws[2] + 1j * draws[4])
    p = raw.shape[1]
    row, col = np.tril_indices(p, -1)
    t = np.broadcast_to(
        np.eye(p, dtype=complex), (2048, len(freq), p, p)
    ).copy()
    t[..., row, col] = -theta
    residual = np.diagonal(
        t @ power @ t.conj().swapaxes(-1, -2), axis1=-2, axis2=-1
    ).real
    ll = -np.sum(chunks * gamma + residual * np.exp(-gamma), axis=(-2, -1))
    prior = np.sum(
        -0.5 * draws[0][..., :2] ** 2 / 10 - 0.5 * np.log(20 * np.pi),
        axis=(-2, -1),
    )
    for coef, local, tau in (
        (draws[0][..., 2:], draws[1], draws[6]),
        (draws[2], draws[3], draws[7]),
        (draws[4], draws[5], draws[7]),
    ):
        log_s = np.log(4) - np.logaddexp(0, np.log(4) - 2 * (local + tau))
        prior += np.sum(
            -0.5 * coef**2 * np.exp(-2 * log_s)
            - log_s
            - 0.5 * np.log(2 * np.pi),
            axis=(-2, -1),
        )
        hc = expit(-np.arange(1, local.shape[-1] + 1) + basis / 2)
        prior += np.sum(
            np.log(2 / np.pi)
            - np.log(hc)
            - np.logaddexp(0, 2 * local - 2 * np.log(hc))
            + local,
            axis=(-2, -1),
        )
    for tau in (draws[6], draws[7]):
        prior += np.sum(
            np.log(2 / np.pi)
            - np.log(0.01)
            - np.logaddexp(0, 2 * tau - 2 * np.log(0.01))
            + tau,
            axis=(-2, -1),
        )
    losses = logq - ll - prior
    inv = np.linalg.inv(t)
    psd = (
        (inv * np.exp(gamma)[..., None, :])
        @ inv.conj().swapaxes(-1, -2)
        / fs
        * std**2
    )
    columns = []
    names = []
    for channel in range(p):
        for f in (0.05, 0.2, 0.4):
            idx = int(np.argmin(abs(freq - f)))
            columns.append(psd[:, idx, channel, channel].real)
            names.append(f"PSD[{channel}] at {f} cycles/sample")
        columns.append(
            2
            * fs
            * (freq[1] - freq[0])
            * np.sum(psd[:, :, channel, channel].real, axis=1)
        )
        names.append(f"bandpower[{channel}] positive bins x2")
    if p == 2:
        coherence = abs(psd[:, :, 0, 1]) ** 2 / (
            psd[:, :, 0, 0].real * psd[:, :, 1, 1].real
        )
        for f in (0.05, 0.2, 0.4):
            columns.append(coherence[:, np.argmin(abs(freq - f))])
            names.append(f"coherence at {f} cycles/sample")
    values = np.column_stack(columns)
    return losses, values, names


def tf_estimator(raw, fs, chunks, basis):
    import reference_bootstrap
    import numpy as np
    import tensorflow as tf
    from types import SimpleNamespace
    from sgvb_tf.psd_estimator import PSDEstimator
    from sgvb_tf.backend import BayesianModel, ViRunner
    from sgvb_tf._spectral import compute_chunked_fft, _compute_Xmatrices

    if raw.shape[1] > 1:
        return PSDEstimator(
            raw,
            fs=fs,
            N_theta=basis,
            nchunks=chunks,
            ntrain_map=1000,
            n_elbo_maximisation_steps=5000,
            N_samples=128,
            seed=0,
        )
    # The historical public AnalysisData rejects p=1. Its model supports
    # empty theta blocks; use that same 1D oracle as the parity tests.
    x = raw / np.std(raw)
    y, f = compute_chunked_fft(x, chunks, None, fs)
    xd, xt = _compute_Xmatrices(f, basis, basis)
    data = SimpleNamespace(
        x=x,
        p=1,
        fs=fs,
        nchunks=chunks,
        N_delta=basis,
        N_theta=basis,
        y_ft=y,
        freq=f,
        Xmat_delta=tf.constant(xd, tf.float32),
        Xmat_theta=tf.constant(xt, tf.float32),
        summed_periodogram=tf.constant(
            np.einsum("bfi,bfj->fij", y, y.conj()), tf.complex64
        ),
    )
    runner = ViRunner.__new__(ViRunner)
    runner.data = data
    runner.model = BayesianModel(data)
    runner.variation_factor = 0.0
    runner.surrogate_posterior = None
    e = PSDEstimator.__new__(PSDEstimator)
    e.inference_runner = runner
    e.ntrain_map = 1000
    e.n_elbo_maximisation_steps = 5000
    e.N_samples = 128
    e.psd_scaling = np.std(raw)
    e.fs = fs
    e.model = None
    e.samps = None
    e.n = len(raw)
    e.nchunks = chunks
    e.fmax_for_analysis = None
    e.fmin_for_analysis = None
    return e


def worker(args):
    import numpy as np

    metadata = json.loads((OUT / "cases.json").read_text())[args.case]
    raw = np.load(OUT / (args.case + "-data.npz"))["data"]
    fs, chunks, basis = metadata["fs"], metadata["chunks"], metadata["basis"]
    if args.backend == "jax":
        import jax
        from sgvb_psd.psd_estimator import PSDEstimator

        e = PSDEstimator(
            raw,
            fs=fs,
            N_theta=basis,
            nchunks=chunks,
            ntrain_map=1000,
            n_elbo_maximisation_steps=5000,
            N_samples=128,
            seed=args.seed,
        )
    else:
        import tensorflow as tf

        e = tf_estimator(raw, fs, chunks, basis)
    ready = time.perf_counter()
    runner = e.inference_runner
    initial = [np.asarray(a).copy() for a in runner.model.trainable_vars]
    result = {
        "case": args.case,
        "backend": args.backend,
        "seed": args.seed,
        "import_and_prepare_s": ready - BOOT,
        "config": metadata,
        "fits": [],
    }
    for iteration in range(args.repeats):
        if args.backend == "jax":
            import jax.numpy as jnp

            runner.model.trainable_vars = tuple(
                jnp.asarray(a) for a in initial
            )
            runner.rng_key = jax.random.PRNGKey(args.seed)
        else:
            for a, b in zip(runner.model.trainable_vars, initial):
                a.assign(b)
            tf.random.set_seed(args.seed)
        start = time.perf_counter()
        outputs = e.run(lr=0.003)
        elapsed = time.perf_counter() - start
        if iteration == 0:
            result["first_result_worker_elapsed_s"] = (
                time.perf_counter() - BOOT
            )
            print(
                "FIRST_RESULT "
                + json.dumps(
                    {
                        "worker_elapsed_s": result[
                            "first_result_worker_elapsed_s"
                        ]
                    }
                ),
                flush=True,
            )
        finite = bool(all(np.all(np.isfinite(a)) for a in outputs))
        if args.backend == "jax":
            from sgvb_psd.jax_backend.model import PARAMETER_NAMES

            means = [np.asarray(a[0]) for a in runner.model.trainable_vars]
            scales = [
                (
                    np.asarray(runner.svi_params[n + "_scale"])
                    if a.size
                    else np.zeros(a.shape[1:])
                )
                for n, a in zip(PARAMETER_NAMES, runner.model.trainable_vars)
            ]
        else:
            means = [np.asarray(a) for a in runner.surrogate_posterior.mean()]
            scales = [
                np.asarray(a) for a in runner.surrogate_posterior.stddev()
            ]
        losses, features, names = target_and_functionals(
            raw, fs, chunks, basis, means, scales, args.seed
        )
        artifact = f"{args.case}-{args.backend}-{args.seed}-{iteration}.npz"
        np.savez_compressed(
            OUT / artifact,
            losses=losses,
            features=features,
            **{
                f"map_{i}": np.asarray(a)
                for i, a in enumerate(runner.model.trainable_vars)
            },
        )
        result["fits"].append(
            {
                "iteration": iteration,
                "fit_s": elapsed,
                "map_s": runner.map_time,
                "vi_s": runner.vi_time,
                "postprocess_and_sampling_s": elapsed
                - runner.map_time
                - runner.vi_time,
                "finite": finite
                and bool(np.all(np.isfinite(losses)))
                and bool(np.all(np.isfinite(features))),
                "negative_elbo": float(np.mean(losses)),
                "negative_elbo_mc_se": float(
                    np.std(losses, ddof=1) / np.sqrt(len(losses))
                ),
                "functional_names": names,
                "functional_quantiles": np.quantile(
                    features, [0.05, 0.5, 0.95], axis=0
                ).tolist(),
                "artifact": artifact,
            }
        )
    result["loaded_jax"] = "jax" in sys.modules
    result["loaded_tensorflow"] = "tensorflow" in sys.modules
    if args.backend == "jax":
        assert not result["loaded_tensorflow"]
    (OUT / f"{args.case}-{args.backend}-{args.seed}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def coordinator(args):
    import numpy as np
    from simulation_cases import cases
    from sgvb_psd.utils.sim_varma import SimVARMA

    OUT.mkdir(exist_ok=True)
    common = cases()
    definitions = {
        "ar1-small": dict(data=common[0][1], fs=1.0, chunks=2, basis=6),
        "var2-small": dict(
            data=common[1][1], fs=common[1][2], chunks=2, basis=6
        ),
    }
    sim = SimVARMA(
        n_samples=8192,
        var_coeffs=np.array([[[0.5, 0], [0, -0.3]], [[0, 0], [0, -0.5]]]),
        vma_coeffs=np.eye(2)[None],
        sigma=np.array([[1, 0.9], [0.9, 1]]),
        seed=0,
    )
    definitions["var2-large"] = dict(
        data=sim.data, fs=sim.fs, chunks=8, basis=15
    )
    metadata = {}
    for name, d in definitions.items():
        raw = d.pop("data")
        np.savez_compressed(OUT / (name + "-data.npz"), data=raw)
        metadata[name] = {
            **d,
            "samples": len(raw),
            "channels": raw.shape[1],
            "map_steps": 1000,
            "vi_steps": 5000,
            "output_draws": 128,
            "heldout_draws": 2048,
            "lr_map": 0.003,
            "lr_vi": 0.05,
            "vi_family": "fixed MAP mean diagonal",
            "full_model_calibration_validated": False,
        }
    (OUT / "cases.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cases_to_run = ["var2-small"] if args.quick else list(definitions)
    seeds = [0] if args.quick else [0, 1]
    all_results = []
    for case in cases_to_run:
        for seed in seeds:
            for backend in (("jax", "tf") if seed % 2 == 0 else ("tf", "jax")):
                python = ROOT / (
                    ".venv-jaxonly/bin/python"
                    if backend == "jax"
                    else ".venv-tfref/bin/python"
                )
                cmd = [
                    str(python),
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--case",
                    case,
                    "--backend",
                    backend,
                    "--seed",
                    str(seed),
                    "--repeats",
                    "2" if args.quick else "3",
                ]
                env = os.environ.copy()
                env.update(
                    MPLBACKEND="Agg",
                    MPLCONFIGDIR=str(ROOT / ".mpl-cache"),
                    PYTHONPATH=str(ROOT / "src"),
                )
                start = time.perf_counter()
                cold = None
                with (OUT / f"{case}-{backend}-{seed}.log").open("w") as log:
                    process = subprocess.Popen(
                        cmd,
                        cwd=ROOT,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    for line in process.stdout:
                        log.write(line)
                        if line.startswith("FIRST_RESULT ") and cold is None:
                            cold = time.perf_counter() - start
                    rc = process.wait(timeout=180)
                row = {
                    "case": case,
                    "backend": backend,
                    "seed": seed,
                    "exit_code": rc,
                    "first_result_wall_s": cold,
                    "process_total_s_including_evaluation": time.perf_counter()
                    - start,
                }
                all_results.append(row)
                print(row, flush=True)
                (OUT / "processes.json").write_text(
                    json.dumps(all_results, indent=2) + "\n"
                )
                if rc:
                    raise RuntimeError(
                        f"worker failed: {case} {backend} {seed}"
                    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--case")
    p.add_argument("--backend")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    worker(a) if a.worker else coordinator(a)
