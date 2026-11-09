"""Frozen, resumable fixed-truth production VAR2 pointwise coverage campaign."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/var2-coverage"
A = np.array([[[0.5, 0], [0, -0.3]], [[0, 0], [0, -0.5]]])
SIGMA = np.array([[1.0, 0.9], [0.9, 1.0]])
NAMES = ["PSD 0", "PSD 1", "Re CSD 01", "Im CSD 01", "Coherence 01"]


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text())


def analytic_spectrum(freq, fs):
    w = 2 * np.pi * np.asarray(freq)
    polynomial = np.eye(2) - np.einsum(
        "fk,kij->fij", np.exp(-1j * w[:, None] * np.arange(1, 3)), A
    )
    h = np.linalg.inv(polynomial)
    return h @ SIGMA @ h.conj().swapaxes(-1, -2) / fs


def functionals(spectrum):
    diagonal = np.diagonal(spectrum, axis1=-2, axis2=-1).real
    coherence = abs(spectrum[..., 0, 1]) ** 2 / (
        diagonal[..., 0] * diagonal[..., 1]
    )
    return np.stack(
        [
            diagonal[..., 0],
            diagonal[..., 1],
            spectrum[..., 0, 1].real,
            spectrum[..., 0, 1].imag,
            coherence,
        ],
        axis=-1,
    )


def stationary_covariances(length):
    from scipy.linalg import solve_discrete_lyapunov

    companion = np.block([[A[0], A[1]], [np.eye(2), np.zeros((2, 2))]])
    q = np.zeros((4, 4))
    q[:2, :2] = SIGMA
    covariance = solve_discrete_lyapunov(companion, q)
    lags = []
    power = np.eye(4)
    for _ in range(length):
        lags.append((power @ covariance)[:2, :2])
        power = companion @ power
    return np.array(lags), float(max(abs(np.linalg.eigvals(companion))))


def expected_window_spectrum(freq, block_length, fs):
    lags, _ = stationary_covariances(block_length)
    expected = (
        np.broadcast_to(lags[0], (len(freq), 2, 2)).astype(complex).copy()
    )
    for h in range(1, block_length):
        phase = np.exp(-2j * np.pi * np.asarray(freq) * h)
        expected += (1 - h / block_length) * (
            phase[:, None, None] * lags[h]
            + phase.conj()[:, None, None] * lags[h].T
        )
    return expected / fs


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return np.full(np.shape(successes) + (2,), np.nan)
    p = np.asarray(successes) / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = (
        z
        * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / (1 + z * z / total)
    )
    return np.stack([center - half, center + half], axis=-1)


def source_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def preflight(config):
    from sgvb_psd._spectral import compute_chunked_fft
    from sgvb_psd.utils.sim_varma import _calculate_spec_matrix_helper
    from sgvb_psd.jax_backend import JaxViRunner

    fs = config["fs"]
    block = config["samples"] // config["chunks"]
    freq = np.fft.fftfreq(block)[1 : block // 2]
    truth = analytic_spectrum(freq, fs)
    existing = (
        np.array(
            [
                _calculate_spec_matrix_helper(f, 2, A, np.eye(2)[None], SIGMA)
                for f in freq
            ]
        )
        / fs
    )
    np.testing.assert_allclose(truth, existing, atol=2e-15, rtol=2e-15)
    lags, radius = stationary_covariances(block)
    grid = np.arange(16384) / 16384
    integrated = analytic_spectrum(grid, fs).mean(0) * fs
    np.testing.assert_allclose(integrated, lags[0], atol=2e-13)
    # Independent white-noise FFT check; no campaign seeds or fit outcomes.
    white = np.random.default_rng(917321).normal(size=(256 * 2048, 2))
    y, f = compute_chunked_fft(white, 2048, None, fs)
    mean_power = np.mean(abs(y) ** 2, axis=(0, 1)) / fs
    np.testing.assert_allclose(mean_power, np.full(2, 1 / fs), rtol=0.01)
    # Constant-spectrum reconstruction checks /fs and input scale squared.
    scale = 3.7
    runner = JaxViRunner(
        white[:512] / scale, N_theta=6, nchunks=2, fs=fs, seed=0
    )
    params = list(runner.model.trainable_vars)
    import jax.numpy as jnp

    params[0] = (
        jnp.zeros_like(params[0])
        .at[0, :, 0]
        .set(jnp.log(jnp.array([4.0, 9.0]) / scale**2))
    )
    reconstructed = (
        np.asarray(runner.model.psd_samples(tuple(params), fs=fs)) * scale**2
    )
    np.testing.assert_allclose(
        np.diagonal(reconstructed, axis1=-2, axis2=-1).real,
        np.broadcast_to(
            np.array([4.0, 9.0]) / fs, reconstructed.shape[:2] + (2,)
        ),
        rtol=2e-6,
    )
    finite_window = expected_window_spectrum(freq, block, fs)
    relative = abs(
        np.diagonal(finite_window, axis1=-2, axis2=-1).real
        / np.diagonal(truth, axis1=-2, axis2=-1).real
        - 1
    )
    np.savez_compressed(
        OUT / "truth.npz",
        frequency_cycles=freq,
        frequency_physical=freq * fs,
        spectral_matrix=truth,
        functionals=functionals(truth),
        expected_rectangular_periodogram=finite_window,
    )
    return {
        "passed": True,
        "companion_spectral_radius": radius,
        "stationary_covariance": lags[0].tolist(),
        "integrated_spectrum_covariance_real": integrated.real.tolist(),
        "existing_analytic_helper_max_abs_difference": float(
            np.max(abs(truth - existing))
        ),
        "white_noise_mean_density": mean_power.tolist(),
        "white_noise_expected_density": 1 / fs,
        "white_noise_independent_preflight_seed": 917321,
        "physical_psd_reconstruction_scale_squared_check": True,
        "expected_finite_block_diagonal_relative_bias_max": float(
            relative.max()
        ),
        "density_convention": "two-sided spectral density per physical-frequency unit, evaluated at positive bins; no factor 2",
        "csd_convention": "S[0,1] = E[X0 * conjugate(X1)] / fs",
        "frequency_bins": len(freq),
        "campaign_data_or_vi_used": False,
    }


def freeze():
    if (OUT / "config.json").exists():
        raise RuntimeError(
            "configuration already frozen; use run/resume, never overwrite"
        )
    OUT.mkdir(exist_ok=True)
    clean = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    config = {
        "experiment": "fixed-truth frequentist pointwise coverage, NOT prior-predictive SBC or simultaneous coverage",
        "realizations": 100,
        "samples": 512,
        "channels": 2,
        "chunks": 2,
        "basis": 6,
        "fs": float(2 * np.pi),
        "var_coefficients": A.tolist(),
        "innovation_covariance": SIGMA.tolist(),
        "simulation": "existing SimVARMA; identity MA; existing 101-point initialization/burn-in",
        "guide": "production default independent Normal, frozen MAP means, trainable softplus scales initialized 1e-4",
        "optimize_mean": False,
        "elbo_samples": 1,
        "map_steps": 5000,
        "map_learning_rate": 0.003,
        "vi_steps": 10000,
        "vi_learning_rate": 0.05,
        "posterior_draws": 2000,
        "quantiles": [0.05, 0.5, 0.95],
        "priors": "unchanged production shrinkage/log-HalfCauchy model; degree_fluctuate=N_theta/2=3; Nbw=1; Normal base variance 10; bounded variance-named quantity remains scale",
        "dtype": "production float32/complex64; independent summaries float64",
        "preprocessing": "single global sample std division; no centering/taper; fft/sqrt(block length); discard DC/Nyquist; physical draws multiply std squared and divide fs",
        "fit_timeout_seconds": 120,
        "retry_failed_fits": False,
        "master_seed": 20261109,
        "seed_rule": "numpy.SeedSequence([master_seed, namespace, index]).generate_state(1); data namespace 1101; VI namespace 2202",
        "edge_bins_each_end": 5,
        "source_revision": source_revision(),
        "secondary_method": None,
        "secondary_reason": "structured/full covariance guides are research harnesses, not supported production PSDEstimator options; primary campaign takes precedence",
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_status_at_freeze": clean,
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "logical_cpus": os.cpu_count(),
        },
    }
    seeds = []
    for i in range(100):
        data_seed = int(
            np.random.SeedSequence(
                [config["master_seed"], 1101, i]
            ).generate_state(1)[0]
        )
        vi_seed = int(
            np.random.SeedSequence(
                [config["master_seed"], 2202, i]
            ).generate_state(1)[0]
        )
        seeds.append(
            {
                "index": i,
                "data_seed": data_seed,
                "vi_seed": vi_seed,
                "status": "pending",
            }
        )
    assert (
        len(set(s[k] for s in seeds for k in ("data_seed", "vi_seed"))) == 200
    )
    normalization = preflight(config)
    dump(OUT / "normalization.json", normalization)
    dump(OUT / "config.json", config)
    dump(
        OUT / "manifest.json",
        {
            "config_sha256": hashlib.sha256(
                (OUT / "config.json").read_bytes()
            ).hexdigest(),
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "runs": seeds,
        },
    )
    print("Frozen configuration and independent seeds; normalization passed")


def worker(index):
    started = time.perf_counter()
    config = load(OUT / "config.json")
    manifest = load(OUT / "manifest.json")
    spec = manifest["runs"][index]
    from sgvb_psd.psd_estimator import PSDEstimator
    from sgvb_psd.utils.sim_varma import SimVARMA
    from sgvb_psd.jax_backend.model import PARAMETER_NAMES
    import jax
    import jax.numpy as jnp
    import numpyro

    result = {
        "index": index,
        "data_seed": spec["data_seed"],
        "vi_seed": spec["vi_seed"],
        "status": "failed",
    }
    try:
        sim = SimVARMA(
            n_samples=config["samples"],
            var_coeffs=A,
            vma_coeffs=np.eye(2)[None],
            sigma=SIGMA,
            seed=spec["data_seed"],
        )
        estimator = PSDEstimator(
            sim.data,
            fs=config["fs"],
            N_theta=config["basis"],
            nchunks=config["chunks"],
            ntrain_map=config["map_steps"],
            n_elbo_maximisation_steps=config["vi_steps"],
            N_samples=config["posterior_draws"],
            seed=spec["vi_seed"],
            elbo_samples=config["elbo_samples"],
            optimize_mean=False,
        )
        fit_start = time.perf_counter()
        raw, pointwise, _ = estimator.run(lr=config["map_learning_rate"])
        fit_s = time.perf_counter() - fit_start
        runner = estimator.inference_runner
        spectral = (
            np.asarray(raw, dtype=np.complex128) * estimator.psd_scaling**2
        )
        values = functionals(spectral)
        quantiles = np.quantile(values, config["quantiles"], axis=0)
        frequency = np.asarray(estimator.model.data.freq)
        truth = analytic_spectrum(frequency, config["fs"])
        true_values = functionals(truth)
        np.testing.assert_allclose(
            quantiles[:, :, 0],
            pointwise[:, :, 0, 0].real,
            rtol=2e-5,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            quantiles[:, :, 1],
            pointwise[:, :, 1, 1].real,
            rtol=2e-5,
            atol=1e-8,
        )
        lp = np.asarray(runner.lp)
        loss = np.asarray(runner.kdl_losses)
        # Preserve diagnostics even when a numerical check below fails.
        np.savez_compressed(
            OUT / f"run-{index:03d}-partial.npz",
            data=sim.data,
            quantiles=quantiles,
            map_log_posterior=lp,
            vi_negative_elbo=loss,
        )
        finite = bool(
            np.all(np.isfinite(values))
            and np.all(np.isfinite(lp))
            and np.all(np.isfinite(loss))
        )
        if not finite:
            raise FloatingPointError(
                "nonfinite posterior or optimization trace"
            )
        if (
            np.min(values[:, :, 4]) < -1e-6
            or np.max(values[:, :, 4]) > 1 + 1e-5
        ):
            raise FloatingPointError("invalid coherence")
        covered = (quantiles[0] <= true_values) & (true_values <= quantiles[2])
        width = quantiles[2] - quantiles[0]
        error = quantiles[1] - true_values
        norm = np.column_stack(
            [
                true_values[:, 0],
                true_values[:, 1],
                np.sqrt(true_values[:, 0] * true_values[:, 1]),
                np.sqrt(true_values[:, 0] * true_values[:, 1]),
                np.ones(len(frequency)),
            ]
        )
        grad = jax.grad(lambda p: -jnp.sum(runner.model.logpost(p)))(
            runner.model.trainable_vars
        )
        flatgrad = np.concatenate([np.asarray(g).ravel() for g in grad])
        scales = np.concatenate(
            [
                np.asarray(runner.svi_params[n + "_scale"]).ravel()
                for n, s in zip(PARAMETER_NAMES, runner.model.trainable_vars)
                if s.size
            ]
        )
        np.savez_compressed(
            OUT / f"run-{index:03d}.npz",
            data=sim.data,
            frequency_cycles=frequency,
            truth=true_values,
            quantiles=quantiles,
            covered=covered,
            width=width,
            error=error,
            normalized_error=error / norm,
            normalized_width=width / norm,
            map_log_posterior=lp,
            vi_negative_elbo=loss,
            guide_scales=scales,
            **{
                f"map_{i}": np.asarray(p)
                for i, p in enumerate(runner.model.trainable_vars)
            },
            **{
                f"guide_{k}": np.asarray(v)
                for k, v in runner.svi_params.items()
            },
        )
        (OUT / f"run-{index:03d}-partial.npz").unlink()
        result.update(
            status="completed",
            fit_s=fit_s,
            map_s=runner.map_time,
            vi_s=runner.vi_time,
            input_scale=float(estimator.psd_scaling),
            finite=True,
            map_logpost_last=float(lp[-1]),
            map_last_500_change=float(lp[-1] - lp[-500]),
            map_gradient_rms=float(np.sqrt(np.mean(flatgrad**2))),
            map_gradient_abs_max=float(max(abs(flatgrad))),
            vi_first_1000_mean=float(np.mean(loss[:1000])),
            vi_last_1000_mean=float(np.mean(loss[-1000:])),
            vi_last_1000_std=float(np.std(loss[-1000:])),
            vi_previous_1000_mean=float(np.mean(loss[-2000:-1000])),
            guide_scale_min=float(min(scales)),
            guide_scale_max=float(max(scales)),
            guide_scale_median=float(np.median(scales)),
            realization_frequency_average_coverage=np.mean(
                covered, axis=0
            ).tolist(),
            versions={
                "jax": jax.__version__,
                "numpyro": numpyro.__version__,
                "numpy": np.__version__,
            },
            jax_devices=[str(d) for d in jax.devices()],
            tensorflow_imported="tensorflow" in sys.modules,
        )
    except Exception:
        result["error"] = traceback.format_exc()
        traceback.print_exc()
    result["worker_s"] = time.perf_counter() - started
    dump(OUT / f"run-{index:03d}.json", result)
    if result["status"] != "completed":
        sys.exit(1)


def campaign():
    config = load(OUT / "config.json")
    manifest = load(OUT / "manifest.json")
    assert (
        hashlib.sha256((OUT / "config.json").read_bytes()).hexdigest()
        == manifest["config_sha256"]
    )
    if (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        != config["script_sha256"]
    ):
        raise RuntimeError(
            "frozen script changed; do not run a different evaluation implementation"
        )
    for spec in manifest["runs"]:
        index = spec["index"]
        if spec["status"] == "running":
            # Recover a finished atomic result after coordinator interruption.
            # If its outcome is unknown, count it as interrupted, never rerun.
            path = OUT / f"run-{index:03d}.json"
            result = (
                load(path)
                if path.exists()
                else {
                    "status": "interrupted",
                    "error": "coordinator interrupted; unknown fit outcome; no retry",
                }
            )
            spec.update(status=result["status"], result=path.name)
            if spec["status"] == "completed":
                manifest["completed"] += 1
            else:
                manifest["failed"] += 1
                if not path.exists():
                    dump(path, result)
            dump(OUT / "manifest.json", manifest)
        if spec["status"] != "pending":
            continue
        spec.update(
            status="running",
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        manifest["attempted"] += 1
        dump(OUT / "manifest.json", manifest)
        env = os.environ.copy()
        env.update(
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(ROOT / ".mpl-cache"),
            PYTHONPATH=str(ROOT / "src"),
        )
        start = time.perf_counter()
        with (OUT / f"run-{index:03d}.log").open("w") as log:
            try:
                p = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "worker",
                        "--index",
                        str(index),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=config["fit_timeout_seconds"],
                )
                rc = p.returncode
                result = (
                    load(OUT / f"run-{index:03d}.json")
                    if (OUT / f"run-{index:03d}.json").exists()
                    else {
                        "status": "failed",
                        "error": "worker exited without result",
                    }
                )
            except subprocess.TimeoutExpired:
                rc = None
                result = {
                    "status": "timeout",
                    "error": "bounded worker timeout; not retried",
                }
        spec.update(
            status=result["status"],
            returncode=rc,
            process_wall_s=time.perf_counter() - start,
            result=f"run-{index:03d}.json",
        )
        if spec["status"] == "completed":
            manifest["completed"] += 1
        else:
            manifest["failed"] += 1
            if not (OUT / f"run-{index:03d}.json").exists():
                dump(OUT / f"run-{index:03d}.json", result)
        dump(OUT / "manifest.json", manifest)
        print(
            json.dumps(
                {
                    "index": index,
                    "status": spec["status"],
                    "attempted": manifest["attempted"],
                    "completed": manifest["completed"],
                    "failed": manifest["failed"],
                    "seconds": spec["process_wall_s"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run", "worker"])
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        campaign()
    else:
        worker(args.index)
