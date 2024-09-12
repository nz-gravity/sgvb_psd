import time

import numpy as np
import pandas as pd


class PSDAnalyzer:
    """
    This is a utility class to analyze the estimated PSDs, given the true PSDs.
    """

    def __init__(
        self,
        spec_true,
        spectral_density_q,
#        psd_estimator,
        task_id=1,
    ):
        """

        :param spec_true: the true psd
        :param spectral_density_q: An array containing the estimated PSD, including the lower bound (5th percentile), 
        median (50th percentile), and upper bound (95th percentile) for the 90% pointwise confidence interval (CI).
        :param task_id: the number of the analysis task, the default is set to 1, which is one realization
        """
        self.spec_true = spec_true
        self.spectral_density_q = spectral_density_q
        self.n_freq = spectral_density_q.shape[1]
        self.task_id = task_id
#        self.psd_estimator = psd_estimator
        self.real_spec_true = self._transform_spec_true_to_real()

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

    def calculate_L2_error(self):
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

    def calculate_CI_length(self):
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

    def calculate_coverage(self):
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

    def run_analysis(self):
        """What does this do?"""
        L2_VI = self.calculate_L2_error()
        (
            len_point_CI_f11,
            len_point_CI_re_f12,
            len_point_CI_im_f12,
            len_point_CI_f22,
        ) = self.calculate_CI_length()
        coverage_point_CI = self.calculate_coverage()

#        iteration_end_time = time.time()
#        total_iteration_time = iteration_end_time - iteration_start_time

        results = {
            "task_id": self.task_id,
            "L2_errors_VI": L2_VI,
            "len_point_CI_f11": len_point_CI_f11,
            "len_point_CI_re_f12": len_point_CI_re_f12,
            "len_point_CI_im_f12": len_point_CI_im_f12,
            "len_point_CI_f22": len_point_CI_f22,
            "coverage_pointwise": coverage_point_CI,
#            "optimal_lr": self.psd_estimator.optimal_lr,
#            "hyperopt_time": self.psd_estimator.hyperopt_time,
#            "total_iteration_time": total_iteration_time,
        }

        result_df = pd.DataFrame([results])
        csv_file = f"estimated_psd_analysis{self.task_id}.csv"
        result_df.to_csv(csv_file, index=False)
        return result_df
