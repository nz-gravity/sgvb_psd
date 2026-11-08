"""JAX backend aliases. TensorFlow is no longer part of the runtime."""

from ..jax_backend import (
    JaxAnalysisData as AnalysisData,
    JaxBayesianModel as BayesianModel,
    JaxViRunner as ViRunner,
)

__all__ = ["AnalysisData", "BayesianModel", "ViRunner"]
