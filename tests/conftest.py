import pytest
import os
import numpy as np
from sgvb_psd.utils import sim_varma
from collections import namedtuple
import tensorflow as tf
import tensorflow_probability as tfp

# set random seed
np.random.seed(0)
tf.random.set_seed(0)
tfp.random.set_seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def plot_dir():
    plot_dir = os.path.join(HERE, "out_plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


@pytest.fixture
def var2_data():
    sigma = np.array([[1, 0.9], [0.9, 1]])
    ar = np.array([[0.5, 0, 0, 0], [0, -0.3, 0, -0.5]])
    n = 256
    d = 2
    x = sim_varma(model='ar', coeffs=ar, n=n, d=d, sigma=sigma)

    return namedtuple(
        'Data',
        ['ar', 'sigma', 'n', 'd', 'x']
    )(ar, sigma, n, d, x)
