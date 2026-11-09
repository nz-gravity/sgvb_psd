# Credits and papers

## SGVB papers

### Correlated detector noise

Jianan Liu, Avi Vajpeyi, Renate Meyer, Kamiel Janssens, Jeung Eun Lee,
Patricio Maturana-Russel, Nelson Christensen and Yixuan Liu (2024).
**Variational inference for correlated gravitational wave detector network noise.**
[arXiv:2409.13224](https://arxiv.org/abs/2409.13224) ·
[HTML](https://arxiv.org/html/2409.13224v1) ·
[Research code](https://github.com/nz-gravity/sgvb_psd_paper).

This paper introduces the application of blocked-Whittle SGVB spectral
estimation to correlated gravitational-wave detector noise, including ET.

### LISA noise

Jianan Liu, Avi Vajpeyi, Renate Meyer, Jeung Eun Lee and
Patricio Maturana-Russel (2026).
**Variational Bayesian Inference for the Spectral Structure of LISA Noise.**
[arXiv:2608.22245](https://arxiv.org/abs/2608.22245) ·
[DOI](https://doi.org/10.48550/arXiv.2608.22245).

The paper studies SGVB and HMC for multivariate LISA noise spectra using
an eigenbasis representation of the likelihood. The compact notebook here
uses the current joint modified-Cholesky implementation in XYZ2 and AET
channel coordinates; eigenbasis/factorized inference remains unavailable in
the current runtime.

## Related P-spline method

Avi Vajpeyi, Renate Meyer, Patricio Maturana-Russel and Jianan Liu (2026).
**Multivariate Bayesian P-spline estimation of spectral density matrices,
with application to LISA TDI noise.**
[arXiv:2607.04833](https://arxiv.org/abs/2607.04833) ·
[DOI](https://doi.org/10.48550/arXiv.2607.04833) ·
[LogPSplinePSD software](https://github.com/nz-gravity/LogPSplinePSD) ·
[Documentation](https://nz-gravity.github.io/LogPSplinePSD/).

LogPSplinePSD develops the related B-spline/Whittle approach with NUTS and
variational initialization. SGVB's cosine basis and variational approximation
are separate modelling and inference choices.

## Method and data acknowledgements

- The multivariate SGVB method builds on Zhixiong Hu and Raquel Prado,
  *Fast Bayesian inference on spectral analysis of multivariate stationary
  time series* (2023),
  [DOI:10.1016/j.csda.2022.107596](https://doi.org/10.1016/j.csda.2022.107596).
- The [LISA example](examples/lisa_example.ipynb) records the immutable
  [truncated noise-4a data source](https://github.com/nz-gravity/test_data/tree/1ed4cde58b9334340208fa17a294e8e7b736b2e9/lisa_noise/noise_4a_truncated/data)
  and links the [full dataset](https://zenodo.org/records/15698080).
  Its transfer-notch reference follows the equal-arm TDI noise-transfer
  structure described by [Nam et al., Table 2](https://arxiv.org/html/2211.02539v2).
- The [LVK example](examples/LVK_example.ipynb) uses public strain data from
  the [Gravitational Wave Open Science Center](https://gwosc.org/), with the
  exact GW150914 file URLs and quality checks recorded in the notebook.
- The [ET example](examples/ET_example.ipynb) uses the repository's small
  three-channel ET sample associated with the correlated-noise work above.

The examples use short data segments and small optimization budgets for
readable, inexpensive demonstrations. Use the linked papers and research
materials for the full analyses and cite the work relevant to your use.
