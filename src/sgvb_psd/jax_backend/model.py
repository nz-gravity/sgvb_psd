"""Blocked Whittle model in the legacy unconstrained coordinates.

The first axis of every parameter is a *variational draw*, never a data
block. Log densities return one value per draw. All data-block reductions
happen inside the likelihood.
"""

import jax.numpy as jnp
import numpy as np
from jax.nn import sigmoid
from numpyro import distributions as dist

from .._spectral import _compute_Xmatrices, compute_chunked_fft
from .reference import reference_factor

PARAMETER_NAMES = (
    "ga_delta",
    "lla_delta",
    "ga_theta_re",
    "lla_theta_re",
    "ga_theta_im",
    "lla_theta_im",
    "ltau",
    "ltau_theta",
)


class JaxAnalysisData:
    def __init__(
        self,
        x,
        nchunks=1,
        fmax_for_analysis=None,
        fs=1.0,
        N_theta=15,
        N_delta=15,
        fmin_for_analysis=None,
        reference_psd=None,
        data_scale=1.0,
    ):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[1] < 1 or not np.all(np.isfinite(x)):
            raise ValueError("x must be a finite time series of shape (n, p)")
        if (
            not isinstance(nchunks, (int, np.integer))
            or not 1 <= nchunks <= len(x) // 4
        ):
            raise ValueError(
                "nchunks must leave at least four samples per block"
            )
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError("fs must be finite and positive")
        for name, value in (("N_delta", N_delta), ("N_theta", N_theta)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for value in (fmin_for_analysis, fmax_for_analysis):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(
                    "frequency bounds must be finite and nonnegative"
                )
        self.x, self.p, self.nchunks = x, x.shape[1], nchunks
        self.N_theta, self.N_delta, self.fs = N_theta, N_delta, fs
        self.y_ft, self.freq = compute_chunked_fft(
            x,
            nchunks,
            fmax_for_analysis,
            fs,
            fmin_for_analysis,
        )
        if not self.freq.size:
            raise ValueError("the requested band contains no Fourier bins")
        xd, xt = _compute_Xmatrices(self.freq, N_delta, N_theta)
        # Deliberately match the reference float32/complex64 policy. Do not
        # enable x64 globally on behalf of the caller.
        self.Xmat_delta = jnp.asarray(xd, dtype=jnp.float32)
        self.Xmat_theta = jnp.asarray(xt, dtype=jnp.float32)
        y = jnp.asarray(self.y_ft, dtype=jnp.complex64)
        self.reference_psd = None
        self.reference_factor = None
        self.reference_logdet = None
        if reference_psd is not None:
            physical, factor, logdet = reference_factor(
                reference_psd, self.freq * fs, self.p, fs, data_scale
            )
            self.reference_psd = physical
            self.reference_factor = jnp.asarray(factor)
            self.reference_logdet = jnp.asarray(logdet)
            y = jnp.linalg.solve(self.reference_factor[None], y[..., None])[
                ..., 0
            ]
        self.whitened_y_ft = y
        self.summed_periodogram = jnp.einsum("bfi,bfj->fij", y, y.conj())
        if self.reference_factor is not None and not np.all(
            np.isfinite(np.asarray(self.summed_periodogram))
        ):
            raise ValueError(
                "reference whitening exceeds float32 range; use a physically appropriate positive reference floor"
            )


class JaxBayesianModel:
    def __init__(self, data, degree_fluctuate=None, init_params=None, Nbw=1.0):
        if not np.isfinite(Nbw) or Nbw <= 0:
            raise ValueError("Nbw must be finite and positive")
        self.data, self.Nbw = data, Nbw
        self.degree_fluctuate = (
            data.N_delta / 2 if degree_fluctuate is None else degree_fluctuate
        )
        p, nd, nt = data.p, data.N_delta + 2, data.N_theta + 2
        nn = p * (p - 1) // 2
        self.parameter_shapes = (
            (p, nd),
            (p, nd - 2),
            (nn, nt),
            (nn, nt),
            (nn, nt),
            (nn, nt),
            (p, 1),
            (nn, 1),
        )
        cd = jnp.zeros(nd - 2)
        if nd > 10:
            cd = cd.at[8:].set(1.0)
        co = jnp.full(nt, 0.5).at[10:].set(1.5)
        params = (
            jnp.zeros((1, p, nd)),
            -jnp.broadcast_to(cd, (1, p, nd - 2)),
            jnp.zeros((1, nn, nt)),
            -jnp.broadcast_to(co, (1, nn, nt)),
            jnp.zeros((1, nn, nt)),
            -jnp.broadcast_to(co, (1, nn, nt)),
            jnp.full((1, p, 1), -1.0),
            jnp.full((1, nn, 1), -1.5),
        )
        if init_params is not None:
            if len(init_params) != len(params):
                raise ValueError(
                    "init_params must contain eight parameter arrays"
                )
            params = tuple(
                jnp.asarray(a, dtype=jnp.float32) for a in init_params
            )
            for a, shape in zip(params, self.parameter_shapes):
                if a.shape != (1,) + shape:
                    raise ValueError(
                        f"initial parameter shape must be {(1,) + shape}"
                    )
        self.trainable_vars = tuple(
            jnp.asarray(a, jnp.float32) for a in params
        )
        row, col = np.tril_indices(p, k=-1)
        self.row, self.col = row, col

    def _spectrum_factors(self, params):
        gamma = jnp.einsum("fk,spk->sfp", self.data.Xmat_delta, params[0])
        theta = jnp.einsum(
            "fk,sek->sfe", self.data.Xmat_theta, params[2] + 1j * params[4]
        )
        shape = gamma.shape[:2] + (self.data.p, self.data.p)
        t = jnp.broadcast_to(jnp.eye(self.data.p, dtype=jnp.complex64), shape)
        t = t.at[..., self.row, self.col].set(-theta)
        return gamma, t

    def loglik(self, params):
        gamma, t = self._spectrum_factors(params)
        transformed = (
            t @ self.data.summed_periodogram @ t.conj().swapaxes(-1, -2)
        )
        power = jnp.diagonal(transformed, axis1=-2, axis2=-1).real
        likelihood = -jnp.sum(
            self.data.nchunks * gamma + power * jnp.exp(-gamma),
            axis=(-2, -1),
        )
        if self.data.reference_factor is not None:
            # The whitening determinant is fixed but part of the raw-data
            # density: log det(R_ref) for each complex Fourier observation.
            likelihood -= self.data.nchunks * jnp.sum(
                self.data.reference_logdet
            )
        return likelihood / self.Nbw

    def _prior_components(self, params):
        """Density arrays and their Cholesky-row ownership.

        Preserve the inherited Normal scale and one exp Jacobian per
        log-HalfCauchy coordinate. Pairwise theta terms belong to their
        regression row; no prior hyperparameter couples different rows.
        """
        ga, la, re, lr, im, li, tau, tt = params
        terms = [
            (dist.Normal(0.0, jnp.sqrt(10.0)).log_prob(ga[..., :2]), None)
        ]
        for coef, local, global_scale, owner in (
            (ga[..., 2:], la, tau, None),
            (re, lr, tt, self.row),
            (im, li, tt, self.row),
        ):
            a2tau2 = jnp.exp(2 * local) * jnp.exp(2 * global_scale)
            scale = a2tau2 * 4.0 / (a2tau2 + 4.0)
            terms.append((dist.Normal(0.0, scale).log_prob(coef), owner))
            hc_scale = sigmoid(
                -jnp.arange(1, local.shape[-1] + 1, dtype=jnp.float32)
                + self.degree_fluctuate
            )
            terms.append(
                (
                    dist.HalfCauchy(hc_scale).log_prob(jnp.exp(local)) + local,
                    owner,
                )
            )
        for log_scale, owner in ((tau, None), (tt, self.row)):
            terms.append(
                (
                    dist.HalfCauchy(0.01).log_prob(jnp.exp(log_scale))
                    + log_scale,
                    owner,
                )
            )
        return terms

    def logprior(self, params):
        terms = self._prior_components(params)
        total = jnp.sum(terms[0][0], axis=(-2, -1))
        for density, _ in terms[1:]:
            total += jnp.sum(density, axis=(-2, -1))
        return total

    def logpost_rows(self, params):
        """Independent modified-Cholesky row targets, shape (draw, row).

        Their sum is the original joint posterior. The fixed reference
        determinant is split evenly; this does not change any row gradient.
        """
        gamma, t = self._spectrum_factors(params)
        transformed = (
            t @ self.data.summed_periodogram @ t.conj().swapaxes(-1, -2)
        )
        power = jnp.diagonal(transformed, axis1=-2, axis2=-1).real
        values = (
            -jnp.sum(
                self.data.nchunks * gamma + power * jnp.exp(-gamma), axis=-2
            )
            / self.Nbw
        )
        if self.data.reference_logdet is not None:
            values -= (
                self.data.nchunks
                * jnp.sum(self.data.reference_logdet)
                / (self.Nbw * self.data.p)
            )
        for density, owner in self._prior_components(params):
            term = jnp.sum(density, axis=-1)
            if owner is None:
                values += term
            else:
                values = values.at[:, owner].add(term)
        return values

    def logpost(self, params):
        return self.loglik(params) + self.logprior(params)

    def psd_samples(self, params, fs=None):
        """Two-sided density at positive bins; physical units require /fs."""
        gamma, t = self._spectrum_factors(params)
        inverse_t = jnp.linalg.inv(t)
        psd = (
            inverse_t * jnp.exp(gamma)[..., None, :]
        ) @ inverse_t.conj().swapaxes(-1, -2)
        if self.data.reference_factor is not None:
            factor = self.data.reference_factor
            psd = factor @ psd @ factor.conj().swapaxes(-1, -2)
        return psd if fs is None else psd / fs

    def compute_psd(
        self, vi_samples, quantiles=(0.05, 0.5, 0.95), psd_scaling=1.0, fs=None
    ):
        from .._psd_summary import get_pointwise_ci, get_uniform_ci

        # Match the existing API: raw psd_all excludes psd_scaling, whereas
        # both interval arrays include it. This is a separate scientific/API
        # correction to consider after backend parity.
        psd = np.asarray(self.psd_samples(vi_samples, fs=fs))
        pointwise = get_pointwise_ci(psd, quantiles)
        uniform = get_uniform_ci(psd, pointwise)
        return psd, pointwise * psd_scaling**2, uniform * psd_scaling**2
