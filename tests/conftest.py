import pytest
import os
import numpy as np
from sgvb_psd.utils import SimVARMA
from collections import namedtuple
import tensorflow as tf
import tensorflow_probability as tfp


# set random seed
np.random.seed(0)
tf.random.set_seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def plot_dir():
    plot_dir = os.path.join(HERE, "out_plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


@pytest.fixture
def var2_data():
    sigma = np.array([[1., 0.9], [0.9, 1.]])  
    varCoef = np.array([[[0.5, 0.], [0., -0.3]], [[0., 0.], [0., -0.5]]])
    vmaCoef = np.array([[[1.,0.],[0.,1.]]])
    n = 1024
    d = 2
    var2 = SimVARMA(n_samples=n, var_coeffs=varCoef, vma_coeffs=vmaCoef, sigma=sigma)


    # true_var2 = VarmaSim(n=n)
    # freq = (np.arange(0,np.floor_divide(n, 2), 1) / (n)).ravel()
    # true_spec = true_var2.calculateSpecMatrix(freq, varCoef, vmaCoef, sigma) / (2 * np.pi)
    # freq = np.linspace(0, np.pi, n//2)
    # true_psd = [true_spec, freq]


    return var2


@pytest.fixture
def et_data():
    # download and cache the dataset (small) if it doesn't exist
    pass