import os
import sys
from pathlib import Path
from collections import namedtuple

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "validation/reference")
)

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def output_dir():
    d = os.path.join(HERE, "out")
    os.makedirs(plot_dir, exist_ok=True)
    return d


@pytest.fixture
def plot_dir():
    plot_dir = os.path.join(HERE, "out_plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


@pytest.fixture
def et_data():
    # download and cache the dataset (small) if it doesn't exist
    pass
