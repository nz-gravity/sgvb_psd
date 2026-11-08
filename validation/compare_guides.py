"""Same-target guide comparison; research controls, not new public modes."""

import json
import time
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import (
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
)
from numpyro.optim import Adam
from scipy.stats import wasserstein_distance

from guide_tools import Target
from sgvb_psd.jax_backend.guides import diagonal_plus_lowrank_scale

OUT = Path(__file__).resolve().parent / "guide-comparison"
STEPS = 6000
PARTICLES = 8


def schedule(i):
    return jnp.where(i < 4000, 0.01, 0.001)


def custom_guide(target, structured=False):
    def guide():
        for row, indices in enumerate(target.groups):
            n = len(indices)
            if structured:
                loc = numpyro.param(f"loc_{row}", jnp.zeros(n))
                l = numpyro.param(
                    f"chol_{row}",
                    jnp.eye(n) * 0.1,
                    constraint=constraints.scaled_unit_lower_cholesky,
                )
                d = dist.MultivariateNormal(loc, scale_tril=l)
            else:
                # This exactly preserves the TF scale family and physical
                # 1e-4 initialization despite target preconditioning.
                s = numpyro.param(
                    f"scale_{row}",
                    jnp.full(n, 1e-4),
                    constraint=constraints.softplus_positive,
                )
                d = dist.Normal(
                    jnp.zeros(n), s / target.preconditioner[indices]
                ).to_event(1)
            numpyro.sample(f"row_{row}", d)

    return guide


def legacy_scale_distribution(target, row, params=None):
    indices = target.groups[row]
    position = {int(g): i for i, g in enumerate(indices)}
    scale = jnp.zeros((len(indices), len(indices)))
    offset = 0
    for leaf_i, leaf in enumerate(target.leaves):
        for r in range(leaf.shape[0]):
            n = leaf.shape[1]
            global_indices = np.arange(offset + r * n, offset + (r + 1) * n)
            if n and int(global_indices[0]) in position:
                idx = np.array([position[int(g)] for g in global_indices])
                key = f"event_{leaf_i}_{r}"
                if params is None:
                    d = numpyro.param(
                        key + "_scale",
                        jnp.full(n, 1e-4),
                        constraint=constraints.softplus_positive,
                    )
                    u = numpyro.param(
                        key + "_factor",
                        lambda k: jax.random.uniform(k, (n, 2)),
                    )
                else:
                    d, u = params[key + "_scale"], params[key + "_factor"]
                block = diagonal_plus_lowrank_scale(d, u)
                scale = scale.at[idx[:, None], idx[None, :]].set(block)
        offset += leaf.size
    # Apply the common fixed coordinate map after preserving the physical
    # TFP operator. A covariance-form low-rank guide would be different.
    scale = scale / target.preconditioner[indices, None]
    return dist.MultivariateNormal(
        jnp.zeros(len(indices)), covariance_matrix=scale @ scale.T
    )


def legacy_guide(target):
    def guide():
        for row in range(target.p):
            numpyro.sample(
                f"row_{row}", legacy_scale_distribution(target, row)
            )

    return guide


def sample_custom(target, params, key, structured):
    keys = jax.random.split(key, len(target.groups))
    z = jnp.zeros((2048, target.dimension))
    logq = jnp.zeros(2048)
    for row, (indices, key) in enumerate(zip(target.groups, keys)):
        if structured:
            d = dist.MultivariateNormal(
                params[f"loc_{row}"], scale_tril=params[f"chol_{row}"]
            )
        else:
            d = dist.Normal(
                jnp.zeros(len(indices)),
                params[f"scale_{row}"] / target.preconditioner[indices],
            ).to_event(1)
        a = d.sample(key, (2048,))
        z = z.at[:, indices].set(a)
        logq += d.log_prob(a)
    return z, logq


def manual_engine(target):
    initial = jnp.full(target.dimension, jnp.log(jnp.expm1(1e-4)))
    opt = Adam(schedule)

    def loss(rho, key):
        sigma = jax.nn.softplus(rho)
        eps = jax.random.normal(key, (PARTICLES, target.dimension))
        z = eps * sigma / target.preconditioner
        logq = jnp.sum(
            -0.5 * eps**2
            - jnp.log(sigma / target.preconditioner)
            - 0.5 * jnp.log(2 * jnp.pi),
            axis=-1,
        )
        return jnp.mean(logq - jax.vmap(target.logp_z)(z))

    def run(key):
        def body(state, _):
            optim, key = state
            key, drawkey = jax.random.split(key)
            value, grad = jax.value_and_grad(loss)(
                opt.get_params(optim), drawkey
            )
            return (opt.update(jnp.clip(grad, -10, 10), optim), key), value

        (state, key), losses = jax.lax.scan(
            body, (opt.init(initial), key), None, length=STEPS
        )
        return opt.get_params(state), losses

    return jax.jit(run)


def metrics(target, z, logq, save_name):
    logp = jax.jit(jax.vmap(target.logp_z))(z)
    losses = np.asarray(logq - logp)
    flat = np.asarray(target.center + target.preconditioner * z)
    names, features = target.functionals(flat)
    np.savez_compressed(OUT / (save_name + ".npz"), features=features)
    result = {
        "negative_elbo": float(np.mean(losses)),
        "negative_elbo_mc_se": float(
            np.std(losses, ddof=1) / np.sqrt(len(losses))
        ),
        "evaluation_finite": bool(
            np.all(np.isfinite(losses)) and np.all(np.isfinite(features))
        ),
        "functional_names": names,
        "functional_quantiles": np.quantile(
            features, [0.05, 0.5, 0.95], axis=0
        ).tolist(),
    }
    reference = OUT / (target.name + "-nuts.npz")
    if reference.exists():
        ref = np.load(reference)["features"]
        q = np.quantile(features, [0.05, 0.5, 0.95], axis=0)
        r = np.quantile(ref, [0.05, 0.5, 0.95], axis=0)
        result["nuts_functional_median_relative_errors"] = (
            abs(q[1] - r[1]) / np.maximum(abs(r[1]), 1e-12)
        ).tolist()
        result["nuts_functional_width_ratios"] = (
            (q[2] - q[0]) / (r[2] - r[0])
        ).tolist()
        result["nuts_standardized_wasserstein"] = [
            float(
                wasserstein_distance(features[:, i], ref[:, i])
                / np.std(ref[:, i])
            )
            for i in range(ref.shape[1])
        ]
        result["reference_qualification"] = (
            "See reference-results.json; unqualified MCMC is diagnostic only."
        )
    return result


def family_run(target, family):
    if family == "manual_frozen_diagonal":
        run = manual_engine(target)
        count = target.dimension

        def evaluate(params, key):
            sigma = jax.nn.softplus(params)
            eps = jax.random.normal(key, (2048, target.dimension))
            z = eps * sigma / target.preconditioner
            return z, jnp.sum(
                -0.5 * eps**2
                - jnp.log(sigma / target.preconditioner)
                - 0.5 * jnp.log(2 * jnp.pi),
                axis=-1,
            )

    else:
        init = init_to_value(
            values={
                f"row_{i}": jnp.zeros(len(g))
                for i, g in enumerate(target.groups)
            }
        )
        if family == "legacy_scale_lowrank2":
            guide = legacy_guide(target)
            count = 3 * target.dimension
        elif family == "numpyro_frozen_diagonal":
            guide = custom_guide(target)
            count = target.dimension
        elif family == "structured_row_gaussian":
            guide = custom_guide(target, True)
            count = target.dimension + sum(
                len(g) * (len(g) + 1) // 2 for g in target.groups
            )
        elif family == "auto_diagonal":
            guide = AutoDiagonalNormal(
                target.numpyro_model, init_loc_fn=init, init_scale=0.1
            )
            count = 2 * target.dimension
        elif family == "auto_lowrank4":
            guide = AutoLowRankMultivariateNormal(
                target.numpyro_model, init_loc_fn=init, init_scale=0.1, rank=4
            )
            count = 6 * target.dimension
        else:
            guide = AutoMultivariateNormal(
                target.numpyro_model, init_loc_fn=init, init_scale=0.1
            )
            count = (
                target.dimension
                + target.dimension * (target.dimension + 1) // 2
            )
        # Identical clipped Adam schedule/particles in every comparison.
        from numpyro.optim import ClippedAdam

        svi = SVI(
            target.numpyro_model,
            guide,
            ClippedAdam(schedule, clip_norm=10),
            Trace_ELBO(num_particles=PARTICLES),
        )

        def fit(key):
            r = svi.run(key, STEPS, progress_bar=False)
            return r.params, r.losses

        run = jax.jit(fit)
        if family == "legacy_scale_lowrank2":

            def evaluate(params, key):
                z = jnp.zeros((2048, target.dimension))
                logq = jnp.zeros(2048)
                for row, (indices, k) in enumerate(
                    zip(target.groups, jax.random.split(key, target.p))
                ):
                    d = legacy_scale_distribution(target, row, params)
                    v = d.sample(k, (2048,))
                    z = z.at[:, indices].set(v)
                    logq += d.log_prob(v)
                return z, logq

        elif family in ("numpyro_frozen_diagonal", "structured_row_gaussian"):

            def evaluate(params, key):
                return sample_custom(
                    target, params, key, family == "structured_row_gaussian"
                )

        else:

            def evaluate(params, key):
                q = guide.get_posterior(params)
                latent = q.sample(key, (2048,))
                logq = q.log_prob(latent)
                if not q.event_shape:
                    logq = jnp.sum(logq, axis=-1)
                z = jnp.zeros((2048, target.dimension))
                offset = 0
                for indices in target.groups:
                    z = z.at[:, indices].set(
                        latent[:, offset : offset + len(indices)]
                    )
                    offset += len(indices)
                return z, logq

    rows = []
    for seed in (0, 1):
        start = time.perf_counter()
        params, losses = run(jax.random.PRNGKey(seed))
        jax.block_until_ready((params, losses))
        elapsed = time.perf_counter() - start
        z, logq = evaluate(params, jax.random.PRNGKey(100 + seed))
        result = metrics(
            target,
            z,
            logq,
            f"{target.name}-{'longmap-' if '--long-map' in sys.argv else ''}{family}-{seed}",
        )
        result.update(
            seed=seed,
            training_s=elapsed,
            timing=(
                "first call includes compilation"
                if seed == 0
                else "warm executable"
            ),
            free_parameters=count,
            training_finite=bool(np.all(np.isfinite(losses))),
            mean_last500_training_loss=float(
                np.mean(np.asarray(losses)[-500:])
            ),
        )
        rows.append(result)
        print(
            target.name,
            family,
            seed,
            "seconds",
            round(elapsed, 3),
            "ELBO",
            result["negative_elbo"],
            "finite",
            result["evaluation_finite"],
            flush=True,
        )
    return rows


def main():
    result = {
        "dtype": "float64 research comparison",
        "steps": STEPS,
        "particles": PARTICLES,
        "schedule": "0.01 for 4000, then 0.001; per-coordinate gradient clip 10",
        "seeds": [0, 1],
        "initialization": "faithful frozen guides physical sigma 1e-4; richer guides sigma 0.1 in common Hessian-diagonal-scaled coordinates",
        "cases": {},
    }
    long_map = "--long-map" in sys.argv
    for name in (("var2",) if long_map else ("ar1", "var2")):
        target = Target(name, map_steps=3000 if long_map else 100)
        case = {
            "dimension": target.dimension,
            "row_dimensions": [len(g) for g in target.groups],
            "cross_row_hessian_max_abs": target.cross_row_hessian_max_abs,
            "map_steps": 3000 if long_map else 100,
            "map_gradient_norm": target.map_gradient_norm,
            "families": {},
        }
        result["cases"][name] = case
        families = (
            (
                "manual_frozen_diagonal",
                "numpyro_frozen_diagonal",
                "legacy_scale_lowrank2",
                "auto_diagonal",
                "auto_lowrank4",
                "auto_full",
                "structured_row_gaussian",
            )
            if not long_map
            else (
                "numpyro_frozen_diagonal",
                "auto_diagonal",
                "structured_row_gaussian",
            )
        )
        for family in families:
            case["families"][family] = family_run(target, family)
            (
                OUT
                / (
                    "long-map-results.json"
                    if long_map
                    else "guide-results.json"
                )
            ).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
