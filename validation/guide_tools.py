"""Shared research controls; target formulas are the existing joint model."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.flatten_util import ravel_pytree
from numpyro import distributions as dist

from simulation_cases import cases
from sgvb_psd.jax_backend import JaxViRunner


class Target:
    def __init__(self, name, map_steps=100):
        name, raw, fs = next(row for row in cases() if row[0] == name)
        self.name, self.fs = name, fs
        self.input_scale = np.std(raw)
        self.runner = JaxViRunner(
            raw / self.input_scale, N_theta=6, nchunks=2, fs=fs, seed=0
        )
        self.runner.run_phase1(lr_map=0.003, ntrain_map=map_steps)
        self.model = self.runner.model
        self.leaves = tuple(
            jnp.asarray(a[0], jnp.float64) for a in self.model.trainable_vars
        )
        self.center, self.unravel = ravel_pytree(self.leaves)
        self.dimension = len(self.center)
        self.p = self.model.data.p
        self.freq = np.asarray(self.model.data.freq) * fs
        self.Xd = self.model.data.Xmat_delta.astype(jnp.float64)
        self.Xt = self.model.data.Xmat_theta.astype(jnp.float64)
        self.power = self.model.data.summed_periodogram.astype(jnp.complex128)
        self.logpost = jax.jit(self.logpost)
        hessian = jax.hessian(lambda v: -self.logpost(v))(self.center)
        self.preconditioner = jnp.clip(
            1 / jnp.sqrt(jnp.maximum(jnp.abs(jnp.diag(hessian)), 0.01)),
            1e-5,
            3.0,
        )
        self.log_jacobian = jnp.sum(jnp.log(self.preconditioner))
        offset = 0
        groups = [[] for _ in range(self.p)]
        lower_rows = np.tril_indices(self.p, -1)[0]
        for i, leaf in enumerate(self.leaves):
            indices = np.arange(offset, offset + leaf.size).reshape(leaf.shape)
            for row in range(self.p):
                selected = (
                    indices[row]
                    if i in (0, 1, 6)
                    else indices[lower_rows == row]
                )
                groups[row].extend(selected.ravel().tolist())
            offset += leaf.size
        self.groups = tuple(np.asarray(g, int) for g in groups)
        self.cross_row_hessian_max_abs = max(
            [
                float(
                    jnp.max(
                        jnp.abs(
                            hessian[jnp.ix_(jnp.asarray(a), jnp.asarray(b))]
                        )
                    )
                )
                for i, a in enumerate(self.groups)
                for j, b in enumerate(self.groups)
                if i < j
            ]
            or [0.0]
        )
        self.map_gradient_norm = float(
            jnp.linalg.norm(jax.grad(self.logpost)(self.center))
        )
        assert sorted(np.concatenate(self.groups).tolist()) == list(
            range(self.dimension)
        )

    def params(self, v):
        return tuple(a[None] for a in self.unravel(v))

    def factors(self, v):
        params = self.params(v)
        gamma = jnp.einsum("fk,spk->sfp", self.Xd, params[0])[0]
        theta = jnp.einsum("fk,sek->sfe", self.Xt, params[2] + 1j * params[4])[
            0
        ]
        t = jnp.broadcast_to(
            jnp.eye(self.p, dtype=jnp.complex128),
            (len(self.freq), self.p, self.p),
        )
        row, col = np.tril_indices(self.p, -1)
        return gamma, t.at[..., row, col].set(-theta)

    def loglik(self, v):
        gamma, t = self.factors(v)
        residual = jnp.diagonal(
            t @ self.power @ t.conj().swapaxes(-1, -2), axis1=-2, axis2=-1
        ).real
        return (
            -jnp.sum(
                self.model.data.nchunks * gamma + residual * jnp.exp(-gamma)
            )
            / self.model.Nbw
        )

    def logpost(self, v):
        return self.loglik(v) + self.model.logprior(self.params(v))[0]

    def logp_z(self, z):
        return (
            self.logpost(self.center + self.preconditioner * z)
            + self.log_jacobian
        )

    def psd(self, v):
        gamma, t = self.factors(v)
        inv_t = jnp.linalg.inv(t)
        return (
            (inv_t * jnp.exp(gamma)[..., None, :])
            @ inv_t.conj().swapaxes(-1, -2)
            / self.fs
            * self.input_scale**2
        )

    def functionals(self, draws):
        psd = np.asarray(jax.jit(jax.vmap(self.psd))(jnp.asarray(draws)))
        fidx = [
            int(np.argmin(abs(self.freq / self.fs - f)))
            for f in (0.05, 0.2, 0.4)
        ]
        names, columns = [], []
        df = self.freq[1] - self.freq[0]
        for p in range(self.p):
            for f, idx in zip((0.05, 0.2, 0.4), fidx):
                names.append(f"PSD[{p}] at {f} cycles/sample")
                columns.append(psd[:, idx, p, p].real)
            names.append(f"integrated PSD[{p}] positive bins x2")
            columns.append(2 * df * np.sum(psd[:, :, p, p].real, axis=1))
        if self.p > 1:
            coherence = abs(psd[:, :, 0, 1]) ** 2 / (
                psd[:, :, 0, 0].real * psd[:, :, 1, 1].real
            )
            for f, idx in zip((0.05, 0.2, 0.4), fidx):
                names.append(f"coherence at {f} cycles/sample")
                columns.append(coherence[:, idx])
        return names, np.column_stack(columns)

    def numpyro_model(self):
        pieces = []
        for row, indices in enumerate(self.groups):
            pieces.append(
                numpyro.sample(
                    f"row_{row}",
                    dist.ImproperUniform(
                        dist.constraints.real, (), (len(indices),)
                    ),
                )
            )
        z = jnp.zeros(self.dimension)
        for indices, piece in zip(self.groups, pieces):
            z = z.at[indices].set(piece)
        numpyro.factor("target", self.logp_z(z))

    def noncentered_model(self):
        p, nd, nt = (
            self.p,
            self.model.data.N_delta,
            self.model.data.N_theta + 2,
        )
        nn = p * (p - 1) // 2
        hc_d = jax.nn.sigmoid(
            -jnp.arange(1, nd + 1, dtype=jnp.float32)
            + self.model.degree_fluctuate
        ).astype(jnp.float64)
        baseline = numpyro.sample(
            "baseline",
            dist.Normal(jnp.zeros((p, 2)), jnp.sqrt(10.0)).to_event(2),
        )
        local = numpyro.sample(
            "lambda_delta",
            dist.HalfCauchy(jnp.broadcast_to(hc_d, (p, nd))).to_event(2),
        )
        tau = numpyro.sample(
            "tau_delta", dist.HalfCauchy(jnp.full((p, 1), 0.01)).to_event(2)
        )
        eta = numpyro.sample(
            "eta_delta", dist.Normal(jnp.zeros((p, nd)), 1).to_event(2)
        )
        scale = (local * tau) ** 2 * 4 / ((local * tau) ** 2 + 4)
        ga = jnp.concatenate([baseline, scale * eta], axis=-1)
        if nn:
            hc_t = jax.nn.sigmoid(
                -jnp.arange(1, nt + 1, dtype=jnp.float32)
                + self.model.degree_fluctuate
            ).astype(jnp.float64)
            tt = numpyro.sample(
                "tau_theta",
                dist.HalfCauchy(jnp.full((nn, 1), 0.01)).to_event(2),
            )
            pieces = []
            for suffix in ("re", "im"):
                la = numpyro.sample(
                    "lambda_" + suffix,
                    dist.HalfCauchy(jnp.broadcast_to(hc_t, (nn, nt))).to_event(
                        2
                    ),
                )
                e = numpyro.sample(
                    "eta_" + suffix,
                    dist.Normal(jnp.zeros((nn, nt)), 1).to_event(2),
                )
                s = (la * tt) ** 2 * 4 / ((la * tt) ** 2 + 4)
                pieces.extend([s * e, jnp.log(la)])
        else:
            pieces = [jnp.zeros((0, nt)) for _ in range(4)]
            tt = jnp.zeros((0, 1))
        vals = (ga, jnp.log(local), *pieces, jnp.log(tau), jnp.log(tt))
        flat, _ = ravel_pytree(vals)
        numpyro.deterministic("flat", flat)
        numpyro.factor("whittle", self.loglik(flat))

    def noncentered_initial(self):
        ga, la, re, lr, im, li, tau, tt = self.leaves
        a, b = jnp.exp(la), jnp.exp(tau)
        s = (a * b) ** 2 * 4 / ((a * b) ** 2 + 4)
        result = {
            "baseline": ga[:, :2],
            "lambda_delta": a,
            "tau_delta": b,
            "eta_delta": ga[:, 2:] / s,
        }
        jac = jnp.sum(jnp.log(s)) - jnp.sum(la) - jnp.sum(tau)
        if self.p > 1:
            result["tau_theta"] = jnp.exp(tt)
            jac -= jnp.sum(tt)
            for suffix, coef, local in (("re", re, lr), ("im", im, li)):
                a, b = jnp.exp(local), jnp.exp(tt)
                s = (a * b) ** 2 * 4 / ((a * b) ** 2 + 4)
                result["lambda_" + suffix] = a
                result["eta_" + suffix] = coef / s
                jac += jnp.sum(jnp.log(s)) - jnp.sum(local)
        return result, jac
