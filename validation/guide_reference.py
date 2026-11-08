"""Bounded noncentered NUTS reference on the unchanged joint SGVB target."""

import json
import time
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # This research process only.
import jax.numpy as jnp
import numpy as np
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import log_density

from guide_tools import Target

OUT = Path(__file__).resolve().parent / "guide-comparison"


def main():
    OUT.mkdir(exist_ok=True)
    refine = "--refine-ar1" in sys.argv
    chains, warmup, draws = (4, 2000, 2000) if refine else (2, 1000, 1000)
    results = (
        json.loads((OUT / "reference-results.json").read_text())
        if refine
        else {"dtype": "float64 reference process", "cases": {}}
    )
    for name in (("ar1",) if refine else ("ar1", "var2")):
        target = Target(name)
        init, jac = target.noncentered_initial()
        nc, _ = log_density(target.noncentered_model, (), {}, init)
        delta = float(nc - target.logpost(target.center) - jac)
        print(name, "noncentered target/Jacobian check", delta, flush=True)
        # The inherited HalfCauchy schedule contains float32 constants;
        # compare at their precision even in this float64 reference.
        assert abs(delta) < 1e-5
        kernel = NUTS(
            target.noncentered_model,
            init_strategy=init_to_value(values=init),
            dense_mass=True,
            target_accept_prob=0.99 if refine else 0.9,
            max_tree_depth=12 if refine else 10,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=warmup,
            num_samples=draws,
            num_chains=chains,
            chain_method="vectorized",
            progress_bar=False,
        )
        start = time.perf_counter()
        mcmc.run(
            jax.random.PRNGKey(20261109),
            extra_fields=("diverging", "num_steps", "accept_prob"),
        )
        samples = mcmc.get_samples(group_by_chain=True)
        jax.block_until_ready(samples)
        elapsed = time.perf_counter() - start
        flat = np.asarray(samples["flat"])
        flat_diagnostics = summary(
            {"original_log_coordinates": flat}, group_by_chain=True
        )["original_log_coordinates"]
        extra = mcmc.get_extra_fields(group_by_chain=True)
        diagnostics = summary(
            {
                ("log_" + k if k.startswith(("lambda_", "tau_")) else k): (
                    jnp.log(v) if k.startswith(("lambda_", "tau_")) else v
                )
                for k, v in samples.items()
                if k != "flat"
            },
            group_by_chain=True,
        )
        maxr = max(float(np.nanmax(v["r_hat"])) for v in diagnostics.values())
        mine = min(float(np.nanmin(v["n_eff"])) for v in diagnostics.values())
        names, features = target.functionals(
            flat.reshape(-1, target.dimension)
        )
        functional_diag = summary(
            {"functionals": features.reshape(chains, draws, -1)},
            group_by_chain=True,
        )["functionals"]
        div = int(np.sum(extra["diverging"]))
        result = {
            "chains": chains,
            "warmup": warmup,
            "samples_per_chain": draws,
            "target_accept": 0.99 if refine else 0.9,
            "diagnostics_positive_scales": "log coordinates (HalfCauchy raw moments may not exist)",
            "dimension": target.dimension,
            "original_log_coordinate_max_rhat": float(
                np.max(flat_diagnostics["r_hat"])
            ),
            "original_log_coordinate_min_ess": float(
                np.min(flat_diagnostics["n_eff"])
            ),
            "row_dimensions": [len(g) for g in target.groups],
            "target_jacobian_abs_error": abs(delta),
            "seconds_including_compile": elapsed,
            "divergences": div,
            "max_parameter_rhat": maxr,
            "min_parameter_ess": mine,
            "max_functional_rhat": float(np.max(functional_diag["r_hat"])),
            "min_functional_ess": float(np.min(functional_diag["n_eff"])),
            "tree_depth_saturations": int(
                np.sum(
                    np.asarray(extra["num_steps"])
                    >= (4095 if refine else 1023)
                )
            ),
            "mean_accept_probability": float(np.mean(extra["accept_prob"])),
            "functional_names": names,
            "functional_quantiles": np.quantile(
                features, [0.05, 0.5, 0.95], axis=0
            ).tolist(),
            "qualified_strict_reference": bool(
                div == 0
                and maxr < 1.01
                and mine > 200
                and float(np.max(functional_diag["r_hat"])) < 1.01
            ),
        }
        results["cases"][name] = result
        np.savez_compressed(
            OUT / (name + "-nuts.npz"), flat=flat, features=features
        )
        (OUT / "reference-results.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(name, result, flush=True)


if __name__ == "__main__":
    main()
