# SGVB PSD

A python package for estimating the power spectral density (PSD) 
of correlated multivariate detector noise using variational inference (VI).


## Installation

```bash
pip install sgvb_psd
```

## Usage

- [API Documentation](api.rst)
- [Simulated data Example](examples/simulation_example.ipynb)
- [ET Example](examples/ET_example.ipynb)

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
