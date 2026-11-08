"""Quality-controlled backend timing on the conditional Whittle submodel.

Both engines optimize the same analytic full-Gaussian ELBO on 64 fixed
prior-predictive datasets. This is not a full SGVB calibration benchmark.
"""

import time

BOOT = time.perf_counter()
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/upgrade-benchmark"


def prepare():
    import numpy as np
    from scipy.optimize import minimize

    reference = json.loads(
        (ROOT / "validation/vi-diagnosis/calibration-results.json").read_text()
    )
    rows = reference["simulations_detail"]
    freq = np.fft.fftfreq(64)[1:32]
    x = np.column_stack([np.ones(31), freq])
    power = np.array([r["power_sum"] for r in rows])
    modes = []
    chols = []
    for p in power:

        def f(b):
            return (
                2 * np.sum(x @ b)
                + np.sum(p * np.exp(-x @ b))
                + np.sum(b * b) / 20
            )

        def g(b):
            return x.T @ (2 - p * np.exp(-x @ b)) + b / 10

        def h(b):
            return x.T @ ((p * np.exp(-x @ b))[:, None] * x) + np.eye(2) / 10

        fit = minimize(
            f,
            np.linalg.lstsq(x, np.log(p / 2), rcond=None)[0],
            jac=g,
            hess=h,
            method="trust-exact",
            options={"gtol": 1e-9},
        )
        assert np.linalg.norm(g(fit.x)) < 1e-5
        modes.append(fit.x)
        chols.append(np.linalg.cholesky(np.linalg.inv(h(fit.x))))
    return reference, x, power, np.asarray(modes), np.asarray(chols)


def score(mean, cov, x, power, reference):
    import numpy as np
    from scipy.special import ndtr

    gamma = mean @ x.T
    variance = np.einsum("fi,nij,fj->nf", x, cov, x)
    ll = -2 * gamma.sum(1) - np.sum(
        power * np.exp(-gamma + variance / 2), axis=1
    )
    prior = -(
        np.sum(mean * mean, axis=1) + np.trace(cov, axis1=1, axis2=2)
    ) / 20 - np.log(20 * np.pi)
    entropy = 1 + np.log(2 * np.pi) + np.linalg.slogdet(cov)[1] / 2
    loss = -ll - prior - entropy
    rows = reference["simulations_detail"]
    truth = np.array([r["truth"] for r in rows])
    oracle_mean = np.array([r["quadrature"]["mean"] for r in rows])
    oracle_cov = np.array([r["quadrature"]["covariance"] for r in rows])
    optimum = np.array(
        [r["families"]["full_rank"]["negative_elbo_exact"] for r in rows]
    )
    mean_error = np.linalg.norm(
        (mean - oracle_mean)
        / np.sqrt(np.diagonal(oracle_cov, axis1=1, axis2=2)),
        axis=1,
    )
    covariance_error = np.linalg.norm(
        cov - oracle_cov, axis=(1, 2)
    ) / np.linalg.norm(oracle_cov, axis=(1, 2))
    funcs = np.array([[1, 0], [0, 1], [1, 0.25]])
    sd = np.sqrt(np.einsum("fi,nij,fj->nf", funcs, cov, funcs))
    pit = ndtr((truth @ funcs.T - mean @ funcs.T) / sd)
    coverage = np.sum((pit >= 0.05) & (pit <= 0.95), axis=0)
    finite = bool(
        np.all(np.isfinite(loss))
        and np.all(np.isfinite(mean_error))
        and np.all(np.isfinite(covariance_error))
    )
    # This accuracy gate is predefined for the conditional control only.
    passed = bool(
        finite
        and np.max(loss - optimum) < 0.005
        and np.max(mean_error) < 0.005
        and np.max(covariance_error) < 0.03
    )
    return {
        "negative_elbo_gap_max": float(np.max(loss - optimum)),
        "posterior_mean_error_max_in_oracle_sd": float(np.max(mean_error)),
        "posterior_covariance_relative_error_max": float(
            np.max(covariance_error)
        ),
        "nominal90_coverage_counts": coverage.tolist(),
        "simulations": 64,
        "quantities": [
            "intercept",
            "slope",
            "PSD at 0.25 cycles/sample (monotone log transform)",
        ],
        "finite": finite,
        "conditional_accuracy_gate": passed,
        "full_sgvb_calibration_validated": False,
    }


def worker(a):
    import numpy as np

    reference, x, power, modes, chols = prepare()
    if a.backend == "jax":
        import jax
        import jax.numpy as jnp
        from jax.example_libraries.optimizers import adam

        xx, pp, mm, aa = map(
            lambda q: jnp.asarray(q, jnp.float32), (x, power, modes, chols)
        )

        def physical(p):
            r = p["raw"]
            l = (
                jnp.zeros((64, 2, 2))
                .at[:, 0, 0]
                .set(jnp.exp(r[:, 0]))
                .at[:, 1, 0]
                .set(r[:, 1])
                .at[:, 1, 1]
                .set(jnp.exp(r[:, 2]))
            )
            mean = mm + jnp.einsum("nij,nj->ni", aa, p["mean"])
            b = aa @ l
            return mean, b @ b.swapaxes(-1, -2)

        def loss(p):
            mean, cov = physical(p)
            gamma = mean @ xx.T
            var = jnp.einsum("fi,nij,fj->nf", xx, cov, xx)
            ll = -2 * jnp.sum(gamma, axis=1) - jnp.sum(
                pp * jnp.exp(-gamma + var / 2), axis=1
            )
            prior = -(
                jnp.sum(mean * mean, axis=1) + jnp.trace(cov, axis1=1, axis2=2)
            ) / 20 - jnp.log(20 * jnp.pi)
            entropy = 1 + jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(cov)[1] / 2
            return jnp.sum(-ll - prior - entropy)

        init, update, get = adam(lambda i: jnp.where(i < 1000, 0.01, 0.001))

        @jax.jit
        def fit():
            def step(state, i):
                return update(i, jax.grad(loss)(get(state)), state), None

            state, _ = jax.lax.scan(
                step,
                init({"mean": jnp.zeros((64, 2)), "raw": jnp.zeros((64, 3))}),
                jnp.arange(2000),
            )
            return physical(get(state))

        def run():
            result = fit()
            jax.block_until_ready(result)
            return [np.asarray(r).astype(float) for r in result]

    else:
        import tensorflow as tf

        xx, pp, mm, aa = map(
            lambda q: tf.constant(q, tf.float32), (x, power, modes, chols)
        )
        mean_var = tf.Variable(tf.zeros((64, 2)))
        raw_var = tf.Variable(tf.zeros((64, 3)))
        opt = tf.keras.optimizers.Adam(
            tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                [999], [0.01, 0.001]
            )
        )
        opt.build([mean_var, raw_var])

        def physical():
            r = raw_var
            zero = tf.zeros(64)
            l = tf.stack(
                [
                    tf.stack([tf.exp(r[:, 0]), zero], -1),
                    tf.stack([r[:, 1], tf.exp(r[:, 2])], -1),
                ],
                axis=1,
            )
            mean = mm + tf.einsum("nij,nj->ni", aa, mean_var)
            b = aa @ l
            return mean, b @ tf.transpose(b, [0, 2, 1])

        def loss():
            mean, cov = physical()
            gamma = mean @ tf.transpose(xx)
            var = tf.einsum("fi,nij,fj->nf", xx, cov, xx)
            ll = -2 * tf.reduce_sum(gamma, 1) - tf.reduce_sum(
                pp * tf.exp(-gamma + var / 2), 1
            )
            prior = -(
                tf.reduce_sum(mean * mean, 1) + tf.linalg.trace(cov)
            ) / 20 - tf.math.log(tf.constant(20 * np.pi, tf.float32))
            entropy = (
                1
                + tf.math.log(tf.constant(2 * np.pi, tf.float32))
                + tf.linalg.slogdet(cov)[1] / 2
            )
            return tf.reduce_sum(-ll - prior - entropy)

        @tf.function
        def fit():
            for _ in tf.range(2000):
                with tf.GradientTape() as tape:
                    value = loss()
                opt.apply_gradients(
                    zip(
                        tape.gradient(value, [mean_var, raw_var]),
                        [mean_var, raw_var],
                    )
                )
            return physical()

        def run():
            mean_var.assign(tf.zeros_like(mean_var))
            raw_var.assign(tf.zeros_like(raw_var))
            for v in opt.variables:
                v.assign(tf.zeros_like(v))
            return [r.numpy().astype(float) for r in fit()]

    prepared = time.perf_counter()
    result = {
        "backend": a.backend,
        "process_replica": a.replica,
        "steps": 2000,
        "datasets": 64,
        "family": "full Gaussian, analytic conditional ELBO",
        "prepare_including_import_s": prepared - BOOT,
        "fits": [],
    }
    for repeat in range(3):
        start = time.perf_counter()
        mean, cov = run()
        elapsed = time.perf_counter() - start
        if repeat == 0:
            print(
                "FIRST_RESULT "
                + json.dumps({"worker_elapsed_s": time.perf_counter() - BOOT}),
                flush=True,
            )
        quality = score(mean, cov, x, power, reference)
        result["fits"].append(
            {"iteration": repeat, "fit_s": elapsed, **quality}
        )
        np.savez_compressed(
            OUT / f"conditional-{a.backend}-{a.replica}-{repeat}.npz",
            mean=mean,
            covariance=cov,
        )
    (OUT / f"conditional-{a.backend}-{a.replica}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def coordinator():
    result = []
    for replica in (0, 1):
        for backend in (("jax", "tf") if replica == 0 else ("tf", "jax")):
            python = ROOT / (
                ".venv-jaxonly/bin/python"
                if backend == "jax"
                else ".venv-tfref/bin/python"
            )
            cmd = [
                str(python),
                str(Path(__file__).resolve()),
                "--worker",
                "--backend",
                backend,
                "--replica",
                str(replica),
            ]
            env = os.environ.copy()
            env.update(
                MPLBACKEND="Agg",
                MPLCONFIGDIR=str(ROOT / ".mpl-cache"),
                PYTHONPATH=str(ROOT / "src"),
            )
            start = time.perf_counter()
            cold = None
            with (OUT / f"conditional-{backend}-{replica}.log").open(
                "w"
            ) as log:
                p = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for line in p.stdout:
                    log.write(line)
                    if line.startswith("FIRST_RESULT ") and cold is None:
                        cold = time.perf_counter() - start
                rc = p.wait(timeout=180)
            row = {
                "backend": backend,
                "replica": replica,
                "first_result_wall_s": cold,
                "exit_code": rc,
            }
            result.append(row)
            print(row, flush=True)
            (OUT / "conditional-processes.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            if rc:
                raise RuntimeError("conditional worker failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--backend")
    p.add_argument("--replica", type=int, default=0)
    a = p.parse_args()
    worker(a) if a.worker else coordinator()
