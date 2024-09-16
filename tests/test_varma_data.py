from sgvb_psd.utils.sim_varma import SimVARMA
from sgvb_psd.utils.tf_utils import set_seed
import numpy as np


def test_data_generation(plot_dir):
    sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
    varCoef = np.array([[[0.5, 0.0], [0.0, -0.3]], [[0.0, 0.0], [0.0, -0.5]]])
    vmaCoef = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    n = 126
    set_seed(0)
    var = SimVARMA(
        n_samples=n, var_coeffs=varCoef, vma_coeffs=vmaCoef, sigma=sigma
    )
    set_seed(0)
    var2 = SimVARMA(
        n_samples=n, var_coeffs=varCoef, vma_coeffs=vmaCoef, sigma=sigma
    )
    assert np.all(var.data == var2.data)
    var2.plot()
    plt.savefig(f"{plot_dir}/pdgrm.png")
