"""Reproduce the bounded UoA parity/fit/timing experiment.

Run from the repository root with MPLBACKEND=Agg and a writable MPLCONFIGDIR.
This is a small migration control, not an uncertainty calibration study.
"""

import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import tensorflow as tf
import tensorflow_probability as tfp

from sgvb_psd.backend import AnalysisData, BayesianModel
from sgvb_psd.jax_backend import JaxAnalysisData, JaxBayesianModel
from sgvb_psd.psd_estimator import PSDEstimator
from sgvb_psd.utils.sim_varma import SimVARMA

OUT = Path(__file__).resolve().parent


def time_calls(fn, params, synchronize, repeats=30):
    start = time.perf_counter()
    synchronize(fn(params))
    first = time.perf_counter() - start
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        synchronize(fn(params))
        times.append(time.perf_counter() - start)
    return {
        "first_call_s": first,
        "warm_median_s": statistics.median(times),
        "warm_calls": repeats,
    }


def main():
    results = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "numpyro": numpyro.__version__,
        "tensorflow": tf.__version__,
        "tfp": tfp.__version__,
        "numpy": np.__version__,
        "devices": [str(d) for d in jax.devices()],
        "dtype": "float32/complex64",
        "fits": {},
    }

    simulation = SimVARMA(
        n_samples=512,
        var_coeffs=np.array([[[0.5, 0], [0, -0.3]], [[0, 0], [0, -0.5]]]),
        vma_coeffs=np.eye(2)[None],
        sigma=np.array([[1, 0.9], [0.9, 1]]),
        seed=0,
    )
    x = simulation.data / np.std(simulation.data)
    data_options = dict(x=x, nchunks=2, fs=simulation.fs, N_delta=6, N_theta=6)
    reference = BayesianModel(AnalysisData(**data_options))
    model = JaxBayesianModel(JaxAnalysisData(**data_options))
    rng = np.random.default_rng(45)
    params = tuple(
        np.asarray(a).repeat(4, 0)
        + rng.normal(0, 0.02, (4,) + a.shape[1:]).astype(np.float32)
        for a in model.trainable_vars
    )
    tf_params = tuple(tf.Variable(a) for a in params)
    ja_params = tuple(jnp.asarray(a) for a in params)
    tf_target = lambda z: reference.loglik(z) + tf.reduce_sum(
        reference.logprior(z)
    )
    ja_target = lambda z: jnp.sum(model.logpost(z))
    with tf.GradientTape() as tape:
        tf_value = tf_target(tf_params)
    tf_grads = tape.gradient(tf_value, tf_params)
    ja_value, ja_grads = jax.value_and_grad(ja_target)(ja_params)
    results["fixed_parameter_parity"] = {
        "draws": 4,
        "chunks": 2,
        "channels": 2,
        "tf_log_target_sum": float(tf_value),
        "jax_log_target_sum": float(ja_value),
        "target_abs_error": float(
            abs(tf_value.numpy() - np.asarray(ja_value))
        ),
        "gradient_max_abs_error": max(
            float(np.max(np.abs(a.numpy() - np.asarray(b))))
            for a, b in zip(tf_grads, ja_grads)
        ),
        "psd_max_abs_error": float(
            np.max(
                np.abs(
                    reference.compute_psd(tf_params, fs=simulation.fs)[0]
                    - np.asarray(
                        model.psd_samples(ja_params, fs=simulation.fs)
                    )
                )
            )
        ),
    }
    results["target_timing"] = {
        "jax": time_calls(
            jax.jit(ja_target), ja_params, jax.block_until_ready
        ),
        "tensorflow": time_calls(
            tf.function(tf_target), tf_params, lambda a: a.numpy()
        ),
    }

    eps = np.random.default_rng(0).normal(size=768)
    ar = np.zeros_like(eps)
    for i in range(1, len(ar)):
        ar[i] = 0.6 * ar[i - 1] + eps[i]
    ar = ar[256:]
    cases = [("ar1", ar, 1.0), ("var2", simulation.data, simulation.fs)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (name, data, fs) in zip(axes, cases):
        options = dict(
            x=data,
            nchunks=2,
            fs=fs,
            N_theta=6,
            ntrain_map=100,
            n_elbo_maximisation_steps=50,
            N_samples=64,
            seed=0,
        )
        estimator = PSDEstimator(**options, backend="jax", elbo_samples=1)
        initial_target = float(
            estimator.inference_runner.model.logpost(
                estimator.inference_runner.model.trainable_vars
            )[0]
        )
        start = time.perf_counter()
        _, pointwise, _ = estimator.run(lr=0.003)
        total = time.perf_counter() - start
        runner = estimator.inference_runner
        final_target = float(
            runner.model.logpost(runner.model.trainable_vars)[0]
        )
        fit = {
            "seed": 0,
            "samples": 512,
            "chunks": 2,
            "basis": 6,
            "map_steps": 100,
            "vi_steps": 50,
            "elbo_particles": 1,
            "initial_log_target": initial_target,
            "map_log_target": final_target,
            "map_s_including_compile": runner.map_time,
            "vi_s_including_compile": runner.vi_time,
            "total_s_including_postprocessing": total,
            "finite_intervals": bool(np.all(np.isfinite(pointwise))),
            "finite_elbo": bool(np.all(np.isfinite(runner.kdl_losses))),
        }
        freq = estimator.freq
        if name == "ar1":
            truth = 1 / np.abs(1 - 0.6 * np.exp(-2j * np.pi * freq)) ** 2
        else:
            index = np.searchsorted(simulation.freq, freq)
            truth = simulation.psd[index, 0, 0].real
        median = pointwise[1, :, 0, 0].real
        fit["median_relative_l2_to_truth_channel0"] = float(
            np.linalg.norm(median - truth) / np.linalg.norm(truth)
        )
        ax.plot(freq, truth, "k--", label="True PSD")
        ax.plot(freq, median, label="JAX median")
        ax.fill_between(
            freq,
            pointwise[0, :, 0, 0].real,
            pointwise[2, :, 0, 0].real,
            alpha=0.2,
        )
        if name == "var2":
            tf_estimator = PSDEstimator(**options)
            start = time.perf_counter()
            _, tf_ci, _ = tf_estimator.run(lr=0.003)
            fit["tf_total_s_including_compile_and_postprocessing"] = (
                time.perf_counter() - start
            )
            tf_map = tuple(
                jnp.asarray(a.numpy())
                for a in tf_estimator.model.trainable_vars
            )
            jax_map_psd = np.asarray(
                runner.model.psd_samples(runner.model.trainable_vars, fs=fs)
            )
            tf_map_psd = np.asarray(runner.model.psd_samples(tf_map, fs=fs))
            fit["map_psd_relative_l2_to_tf"] = float(
                np.linalg.norm(jax_map_psd - tf_map_psd)
                / np.linalg.norm(tf_map_psd)
            )
            fit["tf_map_log_target_on_jax"] = float(
                runner.model.logpost(tf_map)[0]
            )
            fit["median_relative_l2_to_tf_channel0"] = float(
                np.linalg.norm(median - tf_ci[1, :, 0, 0])
                / np.linalg.norm(tf_ci[1, :, 0, 0])
            )
            ax.plot(freq, tf_ci[1, :, 0, 0].real, label="TF median", alpha=0.8)
        results["fits"][name] = fit
        ax.set(
            title=name.upper() + " channel 0",
            xlabel="Frequency",
            ylabel="Two-sided PSD",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "jax-fit-control.svg")
    svg = OUT / "jax-fit-control.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines())
        + "\n"
    )
    plt.close(fig)
    (OUT / "jax-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
