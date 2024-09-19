import os
from collections import namedtuple

import numpy as np
import pytest
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def plot_dir():
    plot_dir = os.path.join(HERE, "out_plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


@pytest.fixture
def et_data():
    # download and cache the dataset (small) if it doesn't exist
    pass
