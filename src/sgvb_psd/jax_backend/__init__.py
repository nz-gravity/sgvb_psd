"""Experimental JAX implementation of the original joint SGVB model.

Importing this module does not import TensorFlow or alter JAX's global dtype
configuration. Install the ``jax`` extra before using it.
"""

from .model import JaxAnalysisData, JaxBayesianModel
from .vi_runner import JaxViRunner

__all__ = ["JaxAnalysisData", "JaxBayesianModel", "JaxViRunner"]
