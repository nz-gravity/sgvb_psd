#############
API
#############


Here we document the `PSDEstimator`, the main class to be used from this package.


.. autoclass:: sgvb_psd.psd_estimator.PSDEstimator
    :members:
    :private-members:


Frequency coarse graining
=========================

.. autoclass:: sgvb_psd.inference.CoarseGrainConfig
    :members:

The partition is fixed before inference. Posterior spectra are evaluated on
the native Fourier grid, including any supplied reference matrix. See the
repository README for examples, counts and approximation limits.


Autoregressive references
=========================

.. autofunction:: sgvb_psd.ar_reference.fit_ar_reference

.. autoclass:: sgvb_psd.ar_reference.ARReference
    :members:
