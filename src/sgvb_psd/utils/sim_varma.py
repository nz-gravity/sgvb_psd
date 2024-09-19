import matplotlib.pyplot as plt
import numpy as np

from ..postproc.plot_psd import format_axes, plot_peridogram, plot_single_psd
from ..utils.periodogram import get_periodogram, get_welch_periodogram


class SimVARMA:
    """
    Simulate Vector Autoregressive Moving Average (VARMA) processes and compute related spectral properties.
    """

    def __init__(
        self,
        n_samples: int,
        var_coeffs: np.ndarray,
        vma_coeffs: np.ndarray,
        sigma=np.array([1.0]),
        seed=None,
    ):
        """
        Initialize the SimVARMA class.

        Args:
            n_samples (int): Number of samples to generate.
            var_coeffs (np.ndarray): VAR coefficient array.
            vma_coeffs (np.ndarray): VMA coefficient array.
            sigma (np.ndarray): Covariance matrix or scalar variance.
        """
        self.n_samples = n_samples
        self.var_coeffs = var_coeffs
        self.vma_coeffs = vma_coeffs
        self.sigma = sigma
        self.dim = vma_coeffs.shape[1]
        self.psd_scaling = 1
        self.n_freq_samples = n_samples // 2

        self.fs = 2 * np.pi
        self.freq = (
            np.linspace(0, 0.5, self.n_freq_samples, endpoint=False) * self.fs
        )

        self.data = None  # set in "resimulate"
        self.periodogram = None  # set in "resimulate"
        self.welch_psd = None  # set in "resimulate"
        self.welch_f = None  # set in "resimulate"
        self.resimulate(seed=seed)

        self.psd = _calculate_true_varma_psd(
            self.n_freq_samples,
            self.dim,
            self.var_coeffs,
            self.vma_coeffs,
            self.sigma,
        )

    def resimulate(self, seed=None):
        """
        Simulate VARMA process.

        Args:
            seed (int, optional): Random seed for reproducibility.

        Returns:
            np.ndarray: Simulated VARMA process data.
        """
        if seed is not None:
            np.random.seed(seed)

        lag_ma = self.vma_coeffs.shape[0]
        lag_ar = self.var_coeffs.shape[0]

        if self.sigma.shape[0] == 1:
            cov_matrix = np.identity(self.dim) * self.sigma
        else:
            cov_matrix = self.sigma

        x_init = np.zeros((lag_ar + 1, self.dim))
        x = np.empty((self.n_samples + 101, self.dim))
        x[:] = np.NaN
        x[: lag_ar + 1] = x_init
        epsilon = np.random.multivariate_normal(
            np.zeros(self.dim), cov_matrix, size=[lag_ma]
        )

        for i in range(lag_ar + 1, x.shape[0]):
            epsilon = np.concatenate(
                [
                    np.random.multivariate_normal(
                        np.zeros(self.dim), cov_matrix, size=[1]
                    ),
                    epsilon[:-1],
                ]
            )
            x[i] = np.sum(
                np.matmul(
                    self.var_coeffs,
                    x[i - 1 : i - lag_ar - 1 : -1][..., np.newaxis],
                ),
                axis=(0, -1),
            ) + np.sum(
                np.matmul(self.vma_coeffs, epsilon[..., np.newaxis]),
                axis=(0, -1),
            )

        self.data = x[101:]
        self.periodogram = get_periodogram(
            self.data, fs=self.fs, psd_scaling=self.psd_scaling
        )[0]
        self.compute_welch_periodogram(n_chunks=1)

    def compute_welch_periodogram(self, n_chunks=1):
        (self.welch_psd, self.welch_f) = get_welch_periodogram(
            self.data, fs=self.fs, n_chunks=n_chunks
        )

    def plot(self, axs=None, welch_nchunks=None, **kwargs):
        kwargs["off_symlog"] = kwargs.get("off_symlog", False)
        axs = plot_peridogram(self.periodogram, self.freq, axs=axs, **kwargs)
        if welch_nchunks is not None:
            self.compute_welch_periodogram(n_chunks=welch_nchunks)
            axs = plot_peridogram(
                self.welch_psd,
                self.welch_f,
                axs=axs,
                **kwargs,
                color="gray",
                alpha=0.5,
                zorder=-1,
            )
        plot_single_psd(self.psd, self.freq, axs=axs, **kwargs)
        format_axes(axs, **kwargs)
        return axs

    def _repr_html_(self):
        """
        Return an HTML representation of the VARMA process for rendering in notebooks and Sphinx documentation.
        """
        p = self.var_coeffs.shape[0]  # VAR order
        q = self.vma_coeffs.shape[0]  # VMA order

        html = f"""
        <div style="font-family: Arial, sans-serif;">
            <h3>VARMA({p}, {q}) Process</h3>
            <p>X<sub>t</sub> = """

        # VAR part
        if p > 0:
            for i in range(p):
                html += f"Φ<sub>{i + 1}</sub>X<sub>t-{i + 1}</sub> + "

        # VMA part
        html += "ε<sub>t</sub> + "
        if q > 0:
            for i in range(q):
                html += f"Θ<sub>{i + 1}</sub>ε<sub>t-{i + 1}</sub>"
                if i < q - 1:
                    html += " + "

        html += "</p>"
        html += "<p>ε<sub>t</sub> ~ N(0, Σ)</p>"

        # VAR coefficients
        html += "<h4>VAR coefficients:</h4>"
        for i in range(p):
            html += f"<p>Φ<sub>{i + 1}</sub> = {self._matrix_to_html(self.var_coeffs[i])}</p>"

        # VMA coefficients
        html += "<h4>VMA coefficients:</h4>"
        for i in range(q):
            html += f"<p>Θ<sub>{i + 1}</sub> = {self._matrix_to_html(self.vma_coeffs[i])}</p>"

        # Sigma
        html += "<h4>Covariance matrix:</h4>"
        html += f"<p>Σ = {self._matrix_to_html(self.sigma)}</p>"

        html += "</div>"
        return html

    def _matrix_to_html(self, matrix):
        """
        Convert a numpy array to an HTML table representation.
        """
        html = "<table style='border-collapse: collapse;'>"
        for row in matrix:
            html += "<tr>"
            for val in row:
                html += f"<td style='border: 1px solid black; padding: 4px;'>{val:.2f}</td>"
            html += "</tr>"
        html += "</table>"
        return html


def _calculate_true_varma_psd(
    n_samples: int,
    dim: int,
    var_coeffs: np.ndarray,
    vma_coeffs: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Calculate the spectral matrix for given frequencies.

    Args:
        n_samples int: Number of samples to generate for the true PSD (up to 0.5).
        var_coeffs (np.ndarray): VAR coefficient array.
        vma_coeffs (np.ndarray): VMA coefficient array.
        sigma (np.ndarray): Covariance matrix or scalar variance.

    Returns:
        np.ndarray: VARMA spectral matrix (PSD) for freq from 0 to 0.5.
    """
    freq = np.linspace(0, 0.5, n_samples, endpoint=False)
    spec_matrix = np.apply_along_axis(
        lambda f: _calculate_spec_matrix_helper(
            f, dim, var_coeffs, vma_coeffs, sigma
        ),
        axis=1,
        arr=freq.reshape(-1, 1),
    )
    return spec_matrix / (2 * np.pi)


def _calculate_spec_matrix_helper(f, dim, var_coeffs, vma_coeffs, sigma):
    """
    Helper function to calculate spectral matrix for a single frequency.

    Args:
        f (float): Single frequency value.
        var_coeffs (np.ndarray): VAR coefficient array.
        vma_coeffs (np.ndarray): VMA coefficient array.
        sigma (np.ndarray): Covariance matrix or scalar variance.

    Returns:
        np.ndarray: Calculated spectral matrix for the given frequency.
    """
    if sigma.shape[0] == 1:
        cov_matrix = np.identity(dim) * sigma
    else:
        cov_matrix = sigma

    k_ar = np.arange(1, var_coeffs.shape[0] + 1)
    A_f_re_ar = np.sum(
        var_coeffs * np.cos(np.pi * 2 * k_ar * f)[:, np.newaxis, np.newaxis],
        axis=0,
    )
    A_f_im_ar = -np.sum(
        var_coeffs * np.sin(np.pi * 2 * k_ar * f)[:, np.newaxis, np.newaxis],
        axis=0,
    )
    A_f_ar = A_f_re_ar + 1j * A_f_im_ar
    A_bar_f_ar = np.identity(dim) - A_f_ar
    H_f_ar = np.linalg.inv(A_bar_f_ar)

    k_ma = np.arange(vma_coeffs.shape[0])
    A_f_re_ma = np.sum(
        vma_coeffs * np.cos(np.pi * 2 * k_ma * f)[:, np.newaxis, np.newaxis],
        axis=0,
    )
    A_f_im_ma = -np.sum(
        vma_coeffs * np.sin(np.pi * 2 * k_ma * f)[:, np.newaxis, np.newaxis],
        axis=0,
    )
    A_f_ma = A_f_re_ma + 1j * A_f_im_ma
    A_bar_f_ma = A_f_ma
    H_f_ma = A_bar_f_ma

    spec_mat = H_f_ar @ H_f_ma @ cov_matrix @ H_f_ma.conj().T @ H_f_ar.conj().T
    return spec_mat
