import numpy as np
import pytest
import tensorflow as tf

from sgvb_psd.backend import AnalysisData, BayesianModel, ViRunner
from sgvb_psd.backend.block_bayesian_model import BlockBayesianModel
from sgvb_psd.backend.factorized_vi_runner import FactorizedViRunner
from sgvb_psd.psd_estimator import PSDEstimator


def _standardized_data(n_samples=256, dimension=3):
    rng = np.random.default_rng(1234)
    data = rng.normal(size=(n_samples, dimension)).astype(np.float32)
    return data / np.std(data)


def test_joint_loglik_matches_sum_of_factorized_blocks():
    data = _standardized_data()
    analysis_kwargs = {
        "x": data,
        "nchunks": 4,
        "fmax_for_analysis": 6.0,
        "fs": 16.0,
        "N_theta": 6,
        "fmin_for_analysis": 1.0,
        "fmin_idx_extension": 1,
        "fmax_idx_extension": 2,
    }
    analysis_data = AnalysisData(N_delta=6, **analysis_kwargs)
    joint_model = BayesianModel(analysis_data, Nbw=1.5)
    factorized_runner = FactorizedViRunner(
        init_params=joint_model.trainable_vars,
        Nbw=1.5,
        **analysis_kwargs,
    )

    block_loglik = tf.add_n(
        [
            BlockBayesianModel(
                factorized_runner.data,
                block_index=block_index,
                init_params=factorized_runner._slice_init_params(block_index),
                Nbw=1.5,
            ).loglik(factorized_runner._slice_init_params(block_index))
            for block_index in range(data.shape[1])
        ]
    )

    np.testing.assert_allclose(
        joint_model.loglik(joint_model.trainable_vars).numpy(),
        block_loglik.numpy(),
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    ("posterior_mode", "runner_type"),
    [("joint", ViRunner), ("factorized", FactorizedViRunner)],
)
def test_psd_estimator_selects_requested_runner(posterior_mode, runner_type):
    estimator = PSDEstimator(
        x=_standardized_data(n_samples=128, dimension=2),
        N_theta=6,
        nchunks=2,
        fs=16.0,
        fmax_for_analysis=6.0,
        posterior_mode=posterior_mode,
    )

    assert isinstance(estimator.inference_runner, runner_type)


def test_psd_estimator_rejects_unknown_posterior_mode():
    with pytest.raises(ValueError, match="posterior_mode"):
        PSDEstimator(
            x=_standardized_data(n_samples=128, dimension=2),
            N_theta=6,
            nchunks=2,
            fs=16.0,
            posterior_mode="unknown",
        )


def test_factorized_estimator_smoke_run():
    estimator = PSDEstimator(
        x=_standardized_data(n_samples=128, dimension=2),
        N_theta=6,
        N_samples=3,
        nchunks=2,
        ntrain_map=1,
        n_elbo_maximisation_steps=1,
        fs=16.0,
        fmin_for_analysis=1.0,
        fmax_for_analysis=6.0,
        fmin_idx_extension=1,
        fmax_idx_extension=2,
        posterior_mode="factorized",
        seed=4321,
    )

    psd_all, pointwise_ci, uniform_ci = estimator.run(lr=1e-3)

    expected_shape = (3, estimator.freq.size, 2, 2)
    assert psd_all.shape == expected_shape
    assert pointwise_ci.shape == expected_shape
    assert uniform_ci.shape == expected_shape
    assert np.all(np.isfinite(psd_all))
