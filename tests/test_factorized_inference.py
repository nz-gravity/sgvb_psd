import numpy as np
import pytest
import tensorflow as tf

from sgvb_psd.backend import (
    AnalysisData,
    BayesianModel,
    EigenbasisAnalysisData,
    EigenbasisBayesianModel,
    EigenbasisViRunner,
    FactorizedViRunner,
    ViRunner,
)
from sgvb_psd.backend.block_bayesian_model import BlockBayesianModel
from sgvb_psd.psd_estimator import PSDEstimator


def _standardized_data(n_samples=256, dimension=3):
    rng = np.random.default_rng(1234)
    data = rng.normal(size=(n_samples, dimension)).astype(np.float32)
    return data / np.std(data)


def _analysis_kwargs(data):
    return {
        "x": data,
        "nchunks": 4,
        "fmax_for_analysis": 6.0,
        "fs": 16.0,
        "N_theta": 6,
        "N_delta": 6,
        "fmin_for_analysis": 1.0,
    }


def test_legacy_and_eigenbasis_joint_loglik_match():
    data = _standardized_data()
    analysis_kwargs = _analysis_kwargs(data)
    legacy_data = AnalysisData(**analysis_kwargs)
    eigenbasis_data = EigenbasisAnalysisData(
        **analysis_kwargs,
        fmin_idx_extension=0,
        fmax_idx_extension=0,
    )
    legacy_model = BayesianModel(legacy_data, Nbw=1.5)
    eigenbasis_model = EigenbasisBayesianModel(eigenbasis_data, Nbw=1.5)

    np.testing.assert_allclose(
        legacy_model.loglik(legacy_model.trainable_vars).numpy(),
        eigenbasis_model.loglik(legacy_model.trainable_vars).numpy(),
        rtol=1e-5,
        atol=1e-4,
    )


def test_eigenbasis_joint_loglik_matches_factorized_blocks():
    data = _standardized_data()
    analysis_kwargs = _analysis_kwargs(data)
    eigenbasis_data = EigenbasisAnalysisData(
        **analysis_kwargs,
        fmin_idx_extension=1,
        fmax_idx_extension=2,
    )
    joint_model = EigenbasisBayesianModel(eigenbasis_data, Nbw=1.5)
    factorized_runner = FactorizedViRunner(
        x=data,
        nchunks=analysis_kwargs["nchunks"],
        fmax_for_analysis=analysis_kwargs["fmax_for_analysis"],
        fs=analysis_kwargs["fs"],
        N_theta=analysis_kwargs["N_theta"],
        fmin_for_analysis=analysis_kwargs["fmin_for_analysis"],
        fmin_idx_extension=1,
        fmax_idx_extension=2,
        init_params=joint_model.trainable_vars,
        Nbw=1.5,
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
    ("use_eigenbasis", "posterior_mode", "runner_type"),
    [
        (False, "joint", ViRunner),
        (True, "joint", EigenbasisViRunner),
        (True, "factorized", FactorizedViRunner),
    ],
)
def test_psd_estimator_selects_requested_runner(
    use_eigenbasis, posterior_mode, runner_type
):
    estimator = PSDEstimator(
        x=_standardized_data(n_samples=128, dimension=2),
        N_theta=6,
        nchunks=2,
        fs=16.0,
        fmax_for_analysis=6.0,
        use_eigenbasis=use_eigenbasis,
        posterior_mode=posterior_mode,
    )

    assert isinstance(estimator.inference_runner, runner_type)


def test_legacy_mode_does_not_use_eigenbasis_or_frequency_extension():
    estimator = PSDEstimator(
        x=_standardized_data(n_samples=128, dimension=2),
        N_theta=6,
        nchunks=2,
        fs=16.0,
        fmax_for_analysis=6.0,
    )

    assert isinstance(estimator.inference_runner, ViRunner)
    assert not hasattr(estimator.inference_runner.data, "u")
    assert not hasattr(estimator.inference_runner.data, "output_keep_mask")


def test_eigenbasis_frequency_extension_stops_at_highest_available_bin():
    data = _standardized_data(n_samples=128, dimension=2)
    analysis_data = EigenbasisAnalysisData(
        x=data,
        nchunks=2,
        fmax_for_analysis=6.0,
        fs=16.0,
        N_theta=6,
        N_delta=6,
        fmin_for_analysis=1.0,
        fmin_idx_extension=1,
        fmax_idx_extension=1000,
    )

    n_per_chunk = data.shape[0] // 2
    highest_positive_bin_count = n_per_chunk // 2 - 1
    assert analysis_data.freq.size <= highest_positive_bin_count
    assert analysis_data.freq[-1] < 16.0 / 2


@pytest.mark.parametrize("Nbw", [0.0, -1.0, np.inf, np.nan])
def test_psd_estimator_rejects_invalid_bandwidth_correction(Nbw):
    with pytest.raises(ValueError, match="Nbw"):
        PSDEstimator(
            x=_standardized_data(n_samples=128, dimension=2),
            N_theta=6,
            nchunks=2,
            fs=16.0,
            Nbw=Nbw,
        )


def test_factorized_mode_requires_eigenbasis():
    with pytest.raises(ValueError, match="requires use_eigenbasis=True"):
        PSDEstimator(
            x=_standardized_data(n_samples=128, dimension=2),
            N_theta=6,
            nchunks=2,
            fs=16.0,
            posterior_mode="factorized",
        )


def test_psd_estimator_rejects_unknown_posterior_mode():
    with pytest.raises(ValueError, match="posterior_mode"):
        PSDEstimator(
            x=_standardized_data(n_samples=128, dimension=2),
            N_theta=6,
            nchunks=2,
            fs=16.0,
            posterior_mode="unknown",
        )


@pytest.mark.parametrize("posterior_mode", ["joint", "factorized"])
def test_eigenbasis_estimator_smoke_run(posterior_mode):
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
        use_eigenbasis=True,
        posterior_mode=posterior_mode,
        seed=4321,
    )

    psd_all, pointwise_ci, uniform_ci = estimator.run(lr=1e-3)

    expected_shape = (3, estimator.freq.size, 2, 2)
    assert psd_all.shape == expected_shape
    assert pointwise_ci.shape == expected_shape
    assert uniform_ci.shape == expected_shape
    assert np.all(np.isfinite(psd_all))
