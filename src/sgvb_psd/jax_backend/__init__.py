"""Experimental JAX implementation of the original joint SGVB model.

Importing this module does not import TensorFlow or alter JAX's global dtype
configuration. JAX/NumPyro is the package runtime.
"""

from .model import JaxAnalysisData, JaxBayesianModel
from .vi_runner import JaxViRunner

__all__ = ["JaxAnalysisData", "JaxBayesianModel", "JaxViRunner"]
