"""SGVB models and variational inference implemented with JAX/NumPyro.

Importing this module leaves JAX's global dtype configuration unchanged.
"""

from .coarse_grain import CoarseGrainConfig
from .model import AnalysisData, BayesianModel
from .vi_runner import ViRunner
from .row_vi_runner import RowViRunner

__all__ = [
    "CoarseGrainConfig",
    "AnalysisData",
    "BayesianModel",
    "ViRunner",
    "RowViRunner",
]
