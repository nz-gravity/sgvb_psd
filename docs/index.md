# SGVB PSD

A python package for estimating the power spectral density (PSD)
of correlated multivariate detector noise using variational inference (VI).


## Installation

```bash
pip install sgvb_psd
```

## Quick notebooks

All notebooks install the JAX PR branch explicitly while it is under review.
Their default demonstration budget is 1000 MAP steps, 300 VI steps and 96
draws, with an explicit learning rate. They are intended for a CPU Colab
runtime; local CPU execution is tested, not a guarantee of Colab timings or
publication-level accuracy. Notebook outputs and data caches are not committed.

| Notebook | Default data and reference |
|---|---|
| [1D AR](examples/var1d_example.ipynb) | 1024 simulated samples; approximate AR reference |
| [VAR2](examples/simulation_example.ipynb) | 1024 correlated samples; known analytic truth; parallel Cholesky rows |
| [ET](examples/ET_example.ipynb) | Short slice of the 16 MB sample; anti-aliased downsampling |
| [LISA XYZ2](examples/lisa_example.ipynb) | 13.8 MB cached real excerpt; light-travel-time notch reference |
| [LISA AET](examples/lisa_example_v2.ipynb) | Same excerpt transformed to A/E/T; full matrix reference |
| [LVK + AR](examples/LVK_example.ipynb) | 1 MB H1 strain file; AR order optimized on disjoint pre-event training data; optional H1/L1 |

LISA also supports an explicit no-download synthetic mode or a user-supplied
HDF5 file. Detector examples are short workflow demonstrations, not the
original full-paper analyses. Increase duration, basis and optimization budgets
under a separate validation plan for a scientific result.

- [API documentation](api.rst)

## Paper



The paper for SGVB PSD can be found [here](https://arxiv.org/abs/2409.13224).
Code to generate the plots for the paper are available [here](https://github.com/nz-gravity/sgvb_psd_paper)


## Acknowledging SGVB PSD
If you use this SGVB PSD code in your research, please cite the following papers:
```bibtex
@article{Liu2024,
      title={Variational Inference for Correlated Gravitational Wave Detector Network Noise},
      author={Jianan Liu and Avi Vajpeyi and Renate Meyer and Kamiel Janssens and Jeung Eun Lee and Patricio Maturana-Russel and Nelson Christensen and Yixuan Liu},
      year={2024},
      journal={arXiv preprint},
      volume={2409.13224},
      archivePrefix={arXiv},
      primaryClass={gr-qc},
      url={https://arxiv.org/abs/2409.13224}
}
@article{Hu2023,
	title        = {Fast Bayesian inference on spectral analysis of multivariate stationary time series},
	author       = {Zhixiong Hu and Raquel Prado},
	year         = 2023,
	journal      = {Computational Statistics \& Data Analysis},
	volume       = 178,
	pages        = 107596,
	doi          = {https://doi.org/10.1016/j.csda.2022.107596},
	issn         = {0167-9473},
	url          = {https://www.sciencedirect.com/science/article/pii/S0167947322001761}
}
```
