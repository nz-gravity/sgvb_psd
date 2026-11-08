"""Bounded diagnosis of TF/JAX VI differences; no network operations.

Two existing-style cases, three seeds, up to 10000 VI updates, fixed MAP
centres, independent training streams and common held-out evaluation draws.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import tensorflow as tf
import tensorflow_probability as tfp
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

from sgvb_psd.backend import AnalysisData, BayesianModel, ViRunner
from sgvb_psd.jax_backend import JaxViRunner
from sgvb_psd.jax_backend.model import PARAMETER_NAMES
from simulation_cases import cases

OUT = Path(__file__).resolve().parent / "vi-diagnosis"


def reference_runner(x, fs, initial):
    x_tf = np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x
    data = AnalysisData(
        x_tf, nchunks=2, fs=fs, fmax_for_analysis=None, N_theta=6, N_delta=6
    )
    if x.shape[1] == 1:
        data = SimpleNamespace(**vars(data))
        data.p = 1
        data.y_ft = data.y_ft[..., :1]
        data.summed_periodogram = data.summed_periodogram[:, :1, :1]
    runner = ViRunner.__new__(ViRunner)
    runner.data = data
    runner.model = BayesianModel(data, init_params=initial)
    runner.variation_factor = 0.0
    runner.map_time = 0.0
    runner.init_surrogate_posterior(runner.model.trainable_vars)
    return runner


def softplus_inverse(x):
    return np.log(np.expm1(x)).astype(np.float32)


def target_loss(model, locations, rhos, eps):
    scales = tuple(jax.nn.softplus(a) for a in rhos)
    z = tuple(loc + scale * e for loc, scale, e in zip(locations, scales, eps))
    logq = sum(
        jnp.sum(
            -0.5 * e**2 - jnp.log(s) - 0.5 * np.log(2 * np.pi), axis=(-2, -1)
        )
        for s, e in zip(scales, eps)
    )
    return jnp.mean(logq - model.logpost(z))


def matched_noise_diagnostics(model, reference, initial):
    locations = tuple(jnp.asarray(a[0]) for a in initial)
    rho = tuple(
        jnp.full(a.shape[1:], softplus_inverse(np.float32(1e-4)))
        for a in initial
    )
    noise = tuple(
        np.random.default_rng(100 + i)
        .normal(size=(4,) + a.shape[1:])
        .astype(np.float32)
        for i, a in enumerate(initial)
    )
    value, grads = jax.value_and_grad(
        lambda r: target_loss(model, locations, r, noise)
    )(rho)
    tf_rho = tuple(tf.Variable(np.asarray(r)) for r in rho)
    with tf.GradientTape() as tape:
        scales = tuple(tf.nn.softplus(a) for a in tf_rho)
        z = tuple(
            tf.constant(a[0]) + s * e
            for a, s, e in zip(initial, scales, noise)
        )
        logq = sum(
            tf.reduce_sum(
                -0.5 * e**2 - tf.math.log(s) - 0.5 * np.log(2 * np.pi),
                axis=(-2, -1),
            )
            for s, e in zip(scales, noise)
        )
        # Reference loglik sums draws. Dividing the entire sum by n is the
        # correct MC objective, avoiding legacy multi-draw broadcasting.
        tf_value = (
            tf.reduce_mean(logq)
            - (
                reference.model.loglik(z)
                + tf.reduce_sum(reference.model.logprior(z))
            )
            / 4
        )
    tf_grads = tape.gradient(tf_value, tf_rho)
    tf_optimizer = tf.keras.optimizers.Adam(0.05)
    tf_optimizer.apply_gradients(zip(tf_grads, tf_rho))
    ja_optimizer = Adam(0.05)
    ja_state = ja_optimizer.update(grads, ja_optimizer.init(rho))
    ja_after = ja_optimizer.get_params(ja_state)
    return {
        "particles": 4,
        "tf_negative_elbo": float(tf_value),
        "jax_negative_elbo": float(value),
        "loss_abs_difference": float(abs(float(tf_value) - float(value))),
        "rho_gradient_max_abs_difference": max(
            (
                float(np.max(np.abs(a.numpy() - np.asarray(b))))
                if a.shape.num_elements()
                else 0
            )
            for a, b in zip(tf_grads, grads)
        ),
        "rho_after_default_adam_step_max_abs_difference": max(
            (
                float(np.max(np.abs(a.numpy() - np.asarray(b))))
                if a.shape.num_elements()
                else 0
            )
            for a, b in zip(tf_rho, ja_after)
        ),
        "tf_adam_epsilon": 1e-7,
        "jax_adam_epsilon": 1e-8,
    }


def scales_from_numpyro(params, initial):
    return tuple(
        (
            np.asarray(params[name + "_scale"])
            if a.size
            else np.zeros(a.shape[1:], np.float32)
        )
        for name, a in zip(PARAMETER_NAMES, initial)
    )


def summarize(model, locations, scales, noise):
    z = tuple(
        jnp.asarray(loc + scale * e)
        for loc, scale, e in zip(locations, scales, noise)
    )
    logq = sum(
        jnp.sum(
            -0.5 * e**2 - jnp.log(jnp.asarray(s)) - 0.5 * np.log(2 * np.pi),
            axis=(-2, -1),
        )
        for s, e in zip(scales, noise)
    )
    loss = np.asarray(logq - model.logpost(z))
    psd = np.asarray(model.psd_samples(z, fs=model.data.fs))
    med = np.median(psd[..., 0, 0].real, axis=0)
    width = np.quantile(psd[..., 0, 0].real, 0.95, axis=0) - np.quantile(
        psd[..., 0, 0].real, 0.05, axis=0
    )
    return {
        "heldout_negative_elbo": float(np.mean(loss)),
        "heldout_elbo_mc_se": float(np.std(loss, ddof=1) / np.sqrt(len(loss))),
        "finite": bool(np.all(np.isfinite(loss)) and np.all(np.isfinite(psd))),
        "psd_median_channel0": med.tolist(),
        "psd_90pct_width_channel0": width.tolist(),
        "scales_flat": np.concatenate([s.ravel() for s in scales]).tolist(),
    }


def compare_summaries(a, b):
    def rel(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return float(np.linalg.norm(x - y) / np.linalg.norm(y))

    return {
        "negative_elbo_difference": a["heldout_negative_elbo"]
        - b["heldout_negative_elbo"],
        "psd_median_relative_l2": rel(
            a["psd_median_channel0"], b["psd_median_channel0"]
        ),
        "psd_width_relative_l2": rel(
            a["psd_90pct_width_channel0"], b["psd_90pct_width_channel0"]
        ),
        "scale_relative_l2": rel(a["scales_flat"], b["scales_flat"]),
    }


class FreeMeanDiagnosticRunner(JaxViRunner):
    """Reproduce the initial implementation's distinct variational family."""

    def _guide(self):
        for name, a in zip(PARAMETER_NAMES, self.model.trainable_vars):
            if a.size:
                loc = numpyro.param(name + "_loc", a[0])
                scale = numpyro.param(
                    name + "_scale",
                    jnp.full(a.shape[1:], 1e-4),
                    constraint=constraints.softplus_positive,
                )
                numpyro.sample(
                    name, numpyro.distributions.Normal(loc, scale).to_event(2)
                )


def run_case(name, raw, fs, result):
    x = raw / np.std(raw)
    j = JaxViRunner(x, N_theta=6, nchunks=2, fs=fs, seed=0)
    j.run_phase1(lr_map=0.003, ntrain_map=100)
    initial = tuple(np.asarray(a) for a in j.model.trainable_vars)
    locations = tuple(a[0] for a in initial)
    r = reference_runner(x, fs, initial)
    case = {
        "samples": len(x),
        "channels": x.shape[1],
        "chunks": 2,
        "basis": 6,
        "map_steps": 100,
        "map_initialization": "identical JAX MAP tensors in both guides",
        "tf_guide_trainables": [
            {"name": v.name, "shape": list(v.shape)}
            for v in r.surrogate_posterior.trainable_variables
        ],
        "tf_location_types": [
            type(d.distribution.loc).__name__
            for d in r.surrogate_posterior.model
        ],
        "matched_noise": matched_noise_diagnostics(j.model, r, initial),
        "runs": [],
    }
    result["cases"][name] = case
    noise = tuple(
        np.random.default_rng(1000 + i)
        .normal(size=(2048,) + a.shape[1:])
        .astype(np.float32)
        for i, a in enumerate(initial)
    )

    # Direct counterfactual at 50 steps: means remain free only in this
    # diagnostic subclass. Do not expose the changed family in production.
    free = FreeMeanDiagnosticRunner(
        x, N_theta=6, nchunks=2, fs=fs, seed=0, init_params=initial
    )
    free.map_time = 0.0
    free.run_phase2(50)
    free_locations = tuple(
        np.asarray(free.svi_params[n + "_loc"]) if a.size else a[0]
        for n, a in zip(PARAMETER_NAMES, initial)
    )
    free_scales = scales_from_numpyro(free.svi_params, initial)
    case["free_mean_50_step_counterfactual"] = summarize(
        j.model, free_locations, free_scales, noise
    )
    case["free_mean_50_step_counterfactual"][
        "maximum_location_change_from_map"
    ] = max(
        float(np.max(np.abs(a - b))) if a.size else 0
        for a, b in zip(free_locations, locations)
    )

    for seed in (0, 1, 2):
        r = reference_runner(x, fs, initial)
        q = r.surrogate_posterior
        opt = tf.keras.optimizers.Adam(0.05)

        # Only one draw in the reference production objective, as in its
        # default run. No multi-draw likelihood replication enters this fit.
        @tf.function
        def tf_advance(steps, random_seed):
            return tfp.vi.fit_surrogate_posterior(
                target_log_prob_fn=lambda *z: r.model.loglik(z)
                + r.model.logprior(z),
                surrogate_posterior=q,
                optimizer=opt,
                num_steps=steps,
                seed=random_seed,
                sample_size=1,
            )

        ja = JaxViRunner(
            x, N_theta=6, nchunks=2, fs=fs, seed=seed, init_params=initial
        )
        svi = SVI(
            ja._numpyro_model,
            ja._guide,
            Adam(lambda i: jnp.where(i < 5000, 0.05, 0.005)),
            Trace_ELBO(num_particles=1),
        )
        state, last = None, 0
        for step in (50, 1000, 5000, 10000):
            start = time.perf_counter()
            if step > 5000:
                opt.learning_rate.assign(0.005)
            # Preserve optimizer states across checkpoints. The explicit TF
            # seed gives independent, reproducible streams per segment.
            losses = tf_advance(
                tf.constant(step - last), tf.constant([seed, step], tf.int32)
            )
            tf_scales = tuple(np.asarray(a) for a in q.stddev())
            fit = svi.run(
                jax.random.PRNGKey(seed),
                step - last,
                init_state=state,
                progress_bar=False,
            )
            jax.block_until_ready(fit)
            state = fit.state
            ja_scales = scales_from_numpyro(fit.params, initial)
            tf_summary = summarize(j.model, locations, tf_scales, noise)
            ja_summary = summarize(j.model, locations, ja_scales, noise)
            run = {
                "seed": seed,
                "steps": step,
                "elapsed_s": time.perf_counter() - start,
                "tf_training_finite": bool(np.all(np.isfinite(losses))),
                "jax_training_finite": bool(np.all(np.isfinite(fit.losses))),
                "tensorflow": tf_summary,
                "jax": ja_summary,
                "comparison": compare_summaries(ja_summary, tf_summary),
                "tf_mean_change_from_map_max": max(
                    float(np.max(np.abs(a.numpy() - b))) if b.size else 0
                    for a, b in zip(q.mean(), locations)
                ),
            }
            case["runs"].append(run)
            last = step
            print(name, seed, step, run["comparison"], flush=True)
            (OUT / "results.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
    corrected = next(
        a for a in case["runs"] if a["seed"] == 0 and a["steps"] == 50
    )
    case["free_mean_versus_corrected_50_steps"] = compare_summaries(
        case["free_mean_50_step_counterfactual"], corrected["jax"]
    )


def main():
    OUT.mkdir(exist_ok=True)
    result = {
        "purpose": "VI family, objective and convergence diagnosis",
        "seeds": [0, 1, 2],
        "checkpoints": [50, 1000, 5000, 10000],
        "elbo_particles": 1,
        "heldout_common_draws": 2048,
        "vi_learning_rate": "0.05 for 5000 steps, then 0.005",
        "dtype": "float32/complex64",
        "cases": {},
    }
    for name, x, fs in cases():
        run_case(name, x, fs, result)
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
