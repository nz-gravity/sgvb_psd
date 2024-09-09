import matplotlib.pyplot as plt
import numpy as np

from ..utils.periodogram import get_periodogram
from ..postproc.plot_psd import plot_peridogram, plot_single_psd, format_axes


class SimVARMA:
    """
    Simulate Vector Autoregressive Moving Average (VARMA) processes and compute related spectral properties.
    """

    def __init__(self, n_samples: int, var_coeffs: np.ndarray, vma_coeffs: np.ndarray, sigma=np.array([1.]), seed=None):
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
        self.freq = np.arange(0,np.floor_divide(n_samples, 2), 1) / (n_samples)
        self.fs = self.freq[-1]
        self.data = None
        self.periodogram = None
        self.resimulate(seed=seed)

        self.psd = self._compute_psd()

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
        x[:lag_ar + 1] = x_init
        epsilon = np.random.multivariate_normal(np.zeros(self.dim), cov_matrix, size=[lag_ma])

        for i in range(lag_ar + 1, x.shape[0]):
            epsilon = np.concatenate(
                [np.random.multivariate_normal(np.zeros(self.dim), cov_matrix, size=[1]), epsilon[:-1]])
            x[i] = np.sum(np.matmul(self.var_coeffs, x[i - 1:i - lag_ar - 1:-1][..., np.newaxis]), axis=(0, -1)) + \
                   np.sum(np.matmul(self.vma_coeffs, epsilon[..., np.newaxis]), axis=(0, -1))

        self.data = x[101:]
        self.periodogram = self._compute_periodogram()

    def _compute_periodogram(self):
        """
        Compute the periodogram of the simulated data.

        Returns:
            np.ndarray: Computed periodogram.
        """
        # Implement periodogram computation here using self.frequencies
        # This is a placeholder implementation
        return get_periodogram(self.data, fs=self.fs)[0]

    def _compute_psd(self):
        """
        Compute the true Power Spectral Density (PSD) of the VARMA process.

        Returns:
            np.ndarray: Computed PSD.
        """
        return _calculate_true_varma_psd(self.freq, self.dim, self.var_coeffs, self.vma_coeffs, self.sigma)

    def plot(self, axs=None, **kwargs):
        kwargs['off_symlog']=kwargs.get('off_symlog', False)
        axs=plot_peridogram(self.periodogram, self.freq, axs=axs, **kwargs)
        plot_single_psd(self.psd, self.freq, axs=axs, **kwargs)
        format_axes(axs, **kwargs)
        return axs

    def __repr__(self):
        """
        Return a LaTeX representation of the VARMA process.
        """
        p = self.var_coeffs.shape[0]  # VAR order
        q = self.vma_coeffs.shape[0]  # VMA order

        latex_repr = r"$\mathbf{X}_t = "

        # VAR part
        if p > 0:
            for i in range(p):
                latex_repr += r"\mathbf{\Phi}_{" + str(i+1) + r"}\mathbf{X}_{t-" + str(i+1) + r"} + "

        # VMA part
        latex_repr += r"\mathbf{\epsilon}_t + "
        if q > 0:
            for i in range(q):
                latex_repr += r"\mathbf{\Theta}_{" + str(i+1) + r"}\mathbf{\epsilon}_{t-" + str(i+1) + r"}"
                if i < q - 1:
                    latex_repr += " + "

        # Noise term
        latex_repr += r"$"
        latex_repr += "\n"
        latex_repr += r"$\mathbf{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma})$"

        # VAR coefficients
        latex_repr += "\n\nVAR coefficients:\n"
        for i in range(p):
            latex_repr += r"$\mathbf{\Phi}_{" + str(i+1) + r"} = " + np.array2string(self.var_coeffs[i], precision=2, separator=',') + "$\n"

        # VMA coefficients
        latex_repr += "\nVMA coefficients:\n"
        for i in range(q):
            latex_repr += r"$\mathbf{\Theta}_{" + str(i+1) + r"} = " + np.array2string(self.vma_coeffs[i], precision=2, separator=',') + "$\n"

        # Sigma
        latex_repr += "\nCovariance matrix:\n"
        latex_repr += r"$\mathbf{\Sigma} = " + np.array2string(self.sigma, precision=2, separator=',') + "$"

        return latex_repr




def _calculate_true_varma_psd(freq, dim, var_coeffs, vma_coeffs, sigma):
    """
    Calculate the spectral matrix for given frequencies.

    Args:
        frequencies (np.ndarray): Array of frequencies.
        var_coeffs (np.ndarray): VAR coefficient array.
        vma_coeffs (np.ndarray): VMA coefficient array.
        sigma (np.ndarray): Covariance matrix or scalar variance.

    Returns:
        np.ndarray: Calculated spectral matrix.
    """
    spec_matrix = np.apply_along_axis(
        lambda f: _calculate_spec_matrix_helper(f, dim, var_coeffs, vma_coeffs, sigma),
        axis=1,
        arr=freq.reshape(-1, 1)
    )
    return spec_matrix *2


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
    A_f_re_ar = np.sum(var_coeffs * np.cos(np.pi * 2 * k_ar * f)[:, np.newaxis, np.newaxis], axis=0)
    A_f_im_ar = -np.sum(var_coeffs * np.sin(np.pi * 2 * k_ar * f)[:, np.newaxis, np.newaxis], axis=0)
    A_f_ar = A_f_re_ar + 1j * A_f_im_ar
    A_bar_f_ar = np.identity(dim) - A_f_ar
    H_f_ar = np.linalg.inv(A_bar_f_ar)

    k_ma = np.arange(vma_coeffs.shape[0])
    A_f_re_ma = np.sum(vma_coeffs * np.cos(np.pi * 2 * k_ma * f)[:, np.newaxis, np.newaxis], axis=0)
    A_f_im_ma = -np.sum(vma_coeffs * np.sin(np.pi * 2 * k_ma * f)[:, np.newaxis, np.newaxis], axis=0)
    A_f_ma = A_f_re_ma + 1j * A_f_im_ma
    A_bar_f_ma = A_f_ma
    H_f_ma = A_bar_f_ma

    spec_mat = H_f_ar @ H_f_ma @ cov_matrix @ H_f_ma.conj().T @ H_f_ar.conj().T
    return spec_mat
