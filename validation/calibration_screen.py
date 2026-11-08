"""Conditional prior-predictive SBC for the 1D Whittle submodel.

The two baseline coefficients have their actual N(0, 10) priors. Curvature
and shrinkage nuisance coordinates are held known. This is deliberately
not full SGVB calibration. Exact 2D quadrature separates Gaussian-family
error from stochastic optimization error; no simulation is discarded.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import distributions as dist
from numpyro import factor, param, sample
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import logsumexp, ndtr
from scipy.stats import kstest

OUT = Path(__file__).resolve().parent / "vi-diagnosis"
FREQ = np.fft.fftfreq(64)[1:32]
X = np.column_stack([np.ones_like(FREQ), FREQ])
BLOCKS = 2
SIMULATIONS = 64


def mode_and_covariance(power):
    initial = np.linalg.lstsq(X, np.log(power / BLOCKS), rcond=None)[0]

    def objective(beta):
        g = X @ beta
        return BLOCKS * g.sum() + np.sum(power * np.exp(-g)) + beta @ beta / 20

    def gradient(beta):
        return X.T @ (BLOCKS - power * np.exp(-X @ beta)) + beta / 10

    def hessian(beta):
        return (
            X.T @ ((power * np.exp(-X @ beta))[:, None] * X) + np.eye(2) / 10
        )

    fit = minimize(
        objective,
        initial,
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-9, "maxiter": 100},
    )
    if np.linalg.norm(gradient(fit.x)) > 1e-5:
        raise RuntimeError("mode optimization did not converge")
    return (
        fit.x,
        np.linalg.inv(hessian(fit.x)),
        objective,
        float(np.linalg.norm(gradient(fit.x))),
    )


def decode(v, center, family):
    if family == "fixed_diagonal":
        mean, v = center, v
    else:
        mean, v = v[:2], v[2:]
    if family == "full_rank":
        chol = jnp.array([[jnp.exp(v[0]), 0], [v[1], jnp.exp(v[2])]])
    else:
        chol = jnp.diag(jnp.exp(v))
    return mean, chol @ chol.T


def exact_negative_elbo(mean, covariance, power):
    x = jnp.asarray(X)
    gamma_mean = x @ mean
    gamma_variance = jnp.einsum("fi,ij,fj->f", x, covariance, x)
    expected_ll = -BLOCKS * jnp.sum(gamma_mean) - jnp.sum(
        power * jnp.exp(-gamma_mean + gamma_variance / 2)
    )
    expected_prior = -(mean @ mean + jnp.trace(covariance)) / 20 - jnp.log(
        20 * jnp.pi
    )
    entropy = 1 + jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(covariance)[1] / 2
    return -expected_ll - expected_prior - entropy


@jax.experimental.enable_x64()
def optimize_family(power, center, cov, family):
    if family == "full_rank":
        l = np.linalg.cholesky(cov)
        v0 = np.r_[center, np.log(l[0, 0]), l[1, 0], np.log(l[1, 1])]
    else:
        log_sigma = 0.5 * np.log(np.diag(cov))
        v0 = (
            log_sigma
            if family == "fixed_diagonal"
            else np.r_[center, log_sigma]
        )
    fn = FAMILY_FUNCTIONS[family]

    def fun(v):
        value, gradient = fn(
            jnp.asarray(v, jnp.float64),
            jnp.asarray(center, jnp.float64),
            jnp.asarray(power, jnp.float64),
        )
        return float(value), np.asarray(gradient, np.float64)

    fit = minimize(
        fun,
        v0,
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1e-7, "ftol": 1e-14, "maxiter": 300, "maxls": 40},
    )
    # Only this deterministic control uses a local float64 context. The
    # actual NumPyro SVI control below keeps production float32 arrays.
    mean, covariance = decode(jnp.asarray(fit.x), jnp.asarray(center), family)
    return (
        np.asarray(mean),
        np.asarray(covariance),
        float(fit.fun),
        float(np.linalg.norm(fun(fit.x)[1])),
    )


FAMILY_FUNCTIONS = {
    family: jax.jit(
        jax.value_and_grad(
            lambda v, center, power, family=family: exact_negative_elbo(
                *decode(v, center, family), power
            )
        )
    )
    for family in ("fixed_diagonal", "free_diagonal", "full_rank")
}


def quadrature(center, cov, power, truth, grid_n=161):
    axis = np.linspace(-8, 8, grid_n)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    beta = center + grid @ np.linalg.cholesky(cov).T
    gamma = X @ beta.T
    log_density = (
        -BLOCKS * gamma.sum(0)
        - (power[:, None] * np.exp(-gamma)).sum(0)
        - (beta**2).sum(1) / 20
    )
    weights = np.exp(log_density - logsumexp(log_density))
    boundary = np.max(np.abs(grid), axis=1) > 7.9
    functions = np.array([[1, 0], [0, 1], [1, 0.25]])
    density = np.exp(log_density - np.max(log_density)).reshape(grid_n, grid_n)
    norm = trapezoid(trapezoid(density, axis, axis=0), axis)
    l = np.linalg.cholesky(cov)
    pits = []
    for a in functions:
        coefficient = a @ l
        # Integrate to the actual boundary, interpolating cumulative mass.
        # A hard indicator on grid points creates artificial discrete PITs.
        integrate_dim = int(abs(coefficient[1]) > abs(coefficient[0]))
        other_dim = 1 - integrate_dim
        cdf = cumulative_trapezoid(
            density, axis, axis=integrate_dim, initial=0
        )
        if integrate_dim == 1:
            cdf = cdf.T
        boundary_at = (
            a @ (truth - center) - coefficient[other_dim] * axis
        ) / coefficient[integrate_dim]
        mass = np.array(
            [np.interp(b, axis, cdf[:, j]) for j, b in enumerate(boundary_at)]
        )
        if coefficient[integrate_dim] < 0:
            mass = cdf[-1] - mass
        pits.append(float(trapezoid(mass, axis) / norm))
    mean = np.sum(weights[:, None] * beta, axis=0)
    covariance = np.einsum("n,ni,nj->ij", weights, beta - mean, beta - mean)
    return {
        "pit": pits,
        "boundary_mass": float(weights[boundary].sum()),
        "mean": mean.tolist(),
        "covariance": covariance.tolist(),
    }


def wilson(successes, total):
    z = 1.96
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = (
        z
        * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / (1 + z * z / total)
    )
    return [float(center - half), float(center + half)]


def summarize_pits(pits):
    pits = np.asarray(pits)
    result = {}
    for col, name in enumerate(("intercept", "slope", "log_psd_at_0.25")):
        values = pits[:, col]
        covered = int(np.sum((values >= 0.05) & (values <= 0.95)))
        result[name] = {
            "covered_90pct": covered,
            "total": len(values),
            "coverage": covered / len(values),
            "coverage_wilson_95pct": wilson(covered, len(values)),
            "pit_histogram_8_bins": np.histogram(values, np.linspace(0, 1, 9))[
                0
            ].tolist(),
            "pit_mean": float(np.mean(values)),
            "pit_ks_p_unadjusted_diagnostic": float(
                kstest(values, "uniform").pvalue
            ),
        }
    return result


def accuracy_summary(result):
    rows = result["simulations_detail"]
    correlations = [
        np.asarray(r["quadrature"]["covariance"])[0, 1]
        / np.sqrt(np.prod(np.diag(r["quadrature"]["covariance"])))
        for r in rows
    ]
    summary = {
        "scope": result["scope"],
        "simulations": len(rows),
        "oracle_correlation_range": [
            float(min(correlations)),
            float(max(correlations)),
        ],
        "families": {},
    }
    for family in result["families"]:
        mean_errors, covariance_errors = [], []
        for row in rows:
            q, oracle = row["families"][family], row["quadrature"]
            covariance = np.asarray(oracle["covariance"])
            mean_errors.append(
                float(
                    np.linalg.norm(
                        (np.asarray(q["mean"]) - oracle["mean"])
                        / np.sqrt(np.diag(covariance))
                    )
                )
            )
            covariance_errors.append(
                float(
                    np.linalg.norm(np.asarray(q["covariance"]) - covariance)
                    / np.linalg.norm(covariance)
                )
            )
        summary["families"][family] = {
            "coverage": result["families"][family],
            "mean_error_in_oracle_marginal_sd_median": float(
                np.median(mean_errors)
            ),
            "mean_error_in_oracle_marginal_sd_max": float(max(mean_errors)),
            "covariance_relative_error_median": float(
                np.median(covariance_errors)
            ),
            "covariance_relative_error_max": float(max(covariance_errors)),
        }
    return summary


def svi_control(powers, centers, family, initial_cholesky):
    def model(power, center, initial_cholesky):
        beta = sample(
            "beta", dist.Normal(jnp.zeros(2), jnp.sqrt(10.0)).to_event(1)
        )
        gamma = jnp.asarray(X) @ beta
        factor(
            "whittle",
            -BLOCKS * jnp.sum(gamma) - jnp.sum(power * jnp.exp(-gamma)),
        )

    def guide(power, center, initial_cholesky):
        loc = center if family == "fixed_diagonal" else param("loc", center)
        if family == "full_rank":
            l = param(
                "chol",
                initial_cholesky,
                constraint=constraints.lower_cholesky,
            )
            sample("beta", dist.MultivariateNormal(loc, scale_tril=l))
        else:
            scale = param(
                "scale",
                jnp.ones(2) * 1e-4,
                constraint=constraints.softplus_positive,
            )
            sample("beta", dist.Normal(loc, scale).to_event(1))

    svi = SVI(
        model,
        guide,
        Adam(lambda i: jnp.where(i < 3000, 0.01, 0.001)),
        Trace_ELBO(num_particles=16),
    )

    def one(key, power, center, chol):
        fit = svi.run(key, 6000, power, center, chol, progress_bar=False)
        return fit.params, jnp.all(jnp.isfinite(fit.losses))

    params, finite = jax.jit(jax.vmap(one))(
        jax.random.split(jax.random.PRNGKey(123), len(powers)),
        jnp.asarray(powers, jnp.float32),
        jnp.asarray(centers, jnp.float32),
        jnp.asarray(initial_cholesky, jnp.float32),
    )
    jax.block_until_ready(params)
    results = []
    for i, (power, center) in enumerate(zip(powers, centers)):
        mean = (
            center
            if family == "fixed_diagonal"
            else np.asarray(params["loc"][i])
        )
        if family == "full_rank":
            l = np.asarray(params["chol"][i])
            covariance = l @ l.T
        else:
            covariance = np.diag(np.asarray(params["scale"][i]) ** 2)
        with jax.experimental.enable_x64():
            heldout_loss = float(
                exact_negative_elbo(
                    jnp.asarray(mean, jnp.float64),
                    jnp.asarray(covariance, jnp.float64),
                    jnp.asarray(power, jnp.float64),
                )
            )
        results.append(
            {
                "training_losses_finite": bool(finite[i]),
                "finite": bool(finite[i]) and bool(np.isfinite(heldout_loss)),
                "negative_elbo_exact": heldout_loss,
                "mean": mean.tolist(),
                "covariance": covariance.tolist(),
            }
        )
    return results


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261109)
    truths = rng.normal(0, np.sqrt(10), size=(SIMULATIONS, 2))
    powers = rng.gamma(BLOCKS, 1, size=(SIMULATIONS, len(FREQ))) * np.exp(
        truths @ X.T
    )
    result = {
        "scope": "conditional univariate Whittle: baseline intercept/slope only; known curvature and shrinkage coordinates",
        "seed": 20261109,
        "simulations": SIMULATIONS,
        "real_time_samples_equivalent": 128,
        "blocks": BLOCKS,
        "bins": len(FREQ),
        "prior": "independent N(0,10) baseline coefficients",
        "not_full_sgvb_calibration": True,
        "no_simulation_discarded": True,
        "deterministic_optimization_dtype": "float64 in a local context",
        "svi_control_dtype": "float32",
        "full_rank_svi_initialization": "per-dataset Laplace Cholesky; diagonal guides retain 1e-4 scales",
        "grid": 161,
        "simulations_detail": [],
        "families": {},
    }
    family_pits = {f: [] for f in FAMILY_FUNCTIONS}
    oracle_pits = []
    centers = []
    initial_cholesky = []
    functions = np.array([[1, 0], [0, 1], [1, 0.25]])
    for i, (truth, power) in enumerate(zip(truths, powers)):
        center, cov, objective, mode_grad = mode_and_covariance(power)
        centers.append(center)
        initial_cholesky.append(np.linalg.cholesky(cov))
        oracle = quadrature(center, cov, power, truth)
        oracle_pits.append(oracle["pit"])
        row = {
            "index": i,
            "truth": truth.tolist(),
            "power_sum": power.tolist(),
            "map": center.tolist(),
            "map_gradient_norm": mode_grad,
            "quadrature": oracle,
            "families": {},
        }
        if i < 4:
            refined = quadrature(center, cov, power, truth, 241)
            row["quadrature_pit_max_change_refined_grid"] = float(
                np.max(np.abs(np.asarray(refined["pit"]) - oracle["pit"]))
            )
        for family in FAMILY_FUNCTIONS:
            mean, vcov, loss, grad = optimize_family(
                power, center, cov, family
            )
            std = np.sqrt(np.einsum("fi,ij,fj->f", functions, vcov, functions))
            pit = ndtr((functions @ truth - functions @ mean) / std)
            family_pits[family].append(pit.tolist())
            row["families"][family] = {
                "mean": mean.tolist(),
                "covariance": vcov.tolist(),
                "negative_elbo_exact": loss,
                "optimizer_gradient_norm": grad,
                "pit": pit.tolist(),
            }
        result["simulations_detail"].append(row)
        if (i + 1) % 8 == 0:
            print("quadrature/optimal families", i + 1, flush=True)
    result["quadrature_oracle"] = summarize_pits(oracle_pits)
    result["families"] = {f: summarize_pits(p) for f, p in family_pits.items()}
    result["svi_controls_first_8"] = {}
    for family in FAMILY_FUNCTIONS:
        control = svi_control(
            powers[:8],
            np.asarray(centers[:8]),
            family,
            np.asarray(initial_cholesky[:8]),
        )
        for i, row in enumerate(control):
            row["negative_elbo_gap_to_deterministic_optimum"] = (
                row["negative_elbo_exact"]
                - result["simulations_detail"][i]["families"][family][
                    "negative_elbo_exact"
                ]
            )
        result["svi_controls_first_8"][family] = control
        print(
            "SVI controls",
            family,
            "maximum gap",
            max(
                r["negative_elbo_gap_to_deterministic_optimum"]
                for r in control
            ),
            flush=True,
        )
    (OUT / "calibration-results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (OUT / "calibration-summary.json").write_text(
        json.dumps(accuracy_summary(result), indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "oracle": result["quadrature_oracle"],
                "families": result["families"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
