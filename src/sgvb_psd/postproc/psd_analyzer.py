import numpy as np
import pandas as pd


class PSDAnalyzer:
    """
    This is a utility class to analyze the estimated PSDs, given the true PSDs.
    """

    def __init__(
        self,
        spec_true: np.ndarray,
        spectral_density_q: np.ndarray,
        task_id=1,
        csv_file=None,
    ):
        """

        :param spec_true: the true psd
        :param spectral_density_q: An array containing the estimated PSD, including the lower bound (5th percentile),
        median (50th percentile), and upper bound (95th percentile) for the 90% pointwise confidence interval (CI).
        :param task_id: the number of the analysis task, the default is set to 1, which is one realization
        """
        self.spec_true = spec_true
        self.spectral_density_q = spectral_density_q

        if self.spec_true.shape != spectral_density_q[0].shape:
            raise ValueError(
                "The true PSD and the estimated PSD should have the same shape. "
                f" True PSD shape: {self.spec_true.shape}, "
                f"Estimated PSD shape: {spectral_density_q[0].shape}"
            )

        self.n_freq = spectral_density_q.shape[1]
        self.task_id = task_id
        self.real_spec_true = self._transform_spec_true_to_real()

        # RUN ANALYSIS
        self.l2_error = self._calculate_L2_error()
        (
            self.len_point_CI_f11,
            self.len_point_CI_re_f12,
            self.len_point_CI_im_f12,
            self.len_point_CI_f22,
        ) = self._calculate_CI_length()
        self.coverage_point_CI = self._calculate_coverage()

        csv_file = f"results_{task_id}.csv" if csv_file is None else csv_file
        self._save_data(csv_file)

    def _transform_spec_true_to_real(self):
        real_spec_true = np.zeros_like(self.spec_true, dtype=float)
        for j in range(self.n_freq):
            real_spec_true[j] = self._complex_to_real(self.spec_true[j])
        return real_spec_true

    @staticmethod
    def _complex_to_real(matrix):
        n = matrix.shape[0]
        real_matrix = np.zeros_like(matrix, dtype=float)
        real_matrix[np.triu_indices(n)] = np.real(matrix[np.triu_indices(n)])
        real_matrix[np.tril_indices(n, -1)] = np.imag(
            matrix[np.tril_indices(n, -1)]
        )
        return real_matrix

    def _calculate_L2_error(self):
        spec_mat_median = self.spectral_density_q[1]
        N2_VI = np.empty(self.n_freq)
        for i in range(self.n_freq):
            N2_VI[i] = np.sum(
                np.diag(
                    (spec_mat_median[i, :, :] - self.spec_true[i, :, :])
                    @ (spec_mat_median[i, :, :] - self.spec_true[i, :, :])
                )
            )
        L2_VI = np.sqrt(np.mean(N2_VI))
        return L2_VI

    def _calculate_CI_length(self):
        spec_mat_upper = self.spectral_density_q[2]
        spec_mat_lower = self.spectral_density_q[0]

        len_point_CI_f11 = np.median(
            np.real(spec_mat_upper[:, 0, 0])
        ) - np.median(np.real(spec_mat_lower[:, 0, 0]))
        len_point_CI_re_f12 = np.median(
            np.real(spec_mat_upper[:, 0, 1])
        ) - np.median(np.real(spec_mat_lower[:, 0, 1]))
        len_point_CI_im_f12 = np.median(
            np.imag(spec_mat_upper[:, 1, 0])
        ) - np.median(np.imag(spec_mat_lower[:, 1, 0]))
        len_point_CI_f22 = np.median(
            np.real(spec_mat_upper[:, 1, 1])
        ) - np.median(np.real(spec_mat_lower[:, 1, 1]))

        return (
            len_point_CI_f11,
            len_point_CI_re_f12,
            len_point_CI_im_f12,
            len_point_CI_f22,
        )

    def _calculate_coverage(self):
        spec_mat_lower_real = np.zeros_like(
            self.spectral_density_q[0], dtype=float
        )
        for j in range(self.n_freq):
            spec_mat_lower_real[j] = self._complex_to_real(
                self.spectral_density_q[0][j]
            )

        spec_mat_upper_real = np.zeros_like(
            self.spectral_density_q[2], dtype=float
        )
        for j in range(self.n_freq):
            spec_mat_upper_real[j] = self._complex_to_real(
                self.spectral_density_q[2][j]
            )

        coverage_point_CI = np.mean(
            (spec_mat_lower_real <= self.real_spec_true)
            & (self.real_spec_true <= spec_mat_upper_real)
        )
        return coverage_point_CI

    def _save_data(self, fname):
        results = {
            "task_id": self.task_id,
            "L2_errors_VI": self.l2_error,
            "len_point_CI_S11": self.len_point_CI_f11,
            "len_point_CI_re_S12": self.len_point_CI_re_f12,
            "len_point_CI_im_S12": self.len_point_CI_im_f12,
            "len_point_CI_S22": self.len_point_CI_f22,
            "coverage_pointwise": self.coverage_point_CI,
        }
        result_df = pd.DataFrame([results])
        result_df.to_csv(fname, index=False)
        return result_df
