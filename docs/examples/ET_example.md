---
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

[Open in Colab](https://colab.research.google.com/github/nz-gravity/sgvb_psd/blob/feature/jax-sgvb/docs/examples/ET_example.ipynb)

# Fast ET three-channel sample
Read a bounded slice of the existing 16 MB ET sample. Downsample with an
anti-aliasing filter from 2048 to 256 Hz, then fit the 10–100 Hz band with
independent vmapped Cholesky rows. This is a short demonstration, not the
full ET paper analysis. Downloads are cached; no generated outputs are saved.

```{code-cell} ipython3
# Run once in a fresh Colab runtime. This draft API is on the PR branch.
%pip install -q "jax[cpu]==0.6.2" "numpyro==0.19.0" "sgvb_psd[examples] @ git+https://github.com/nz-gravity/sgvb_psd.git@feature/jax-sgvb"

```

```{code-cell} ipython3
from pathlib import Path
from time import perf_counter
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sgvb_psd.psd_estimator import PSDEstimator
import jax

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                     "axes.labelsize": 10, "axes.titlesize": 11,
                     "legend.fontsize": 8, "figure.dpi": 110,
                     "text.usetex": False})
print("JAX", jax.__version__, "devices:", jax.devices())
# Deliberately small demonstration budgets, not a convergence certificate.
BUDGET = dict(N_theta=8, nchunks=2, ntrain_map=1000,
              n_elbo_maximisation_steps=300, N_samples=96, seed=7)

```

```{code-cell} ipython3
import h5py
from urllib.request import urlopen
DATA_PATH = Path("et_data.h5")
if not DATA_PATH.exists():
    url = "https://raw.githubusercontent.com/nz-gravity/sgvb_psd/main/docs/examples/et_data.h5"
    temporary = DATA_PATH.with_suffix(".part")
    with urlopen(url, timeout=60) as response, temporary.open("wb") as destination:
        import shutil
        shutil.copyfileobj(response, destination)
    temporary.replace(DATA_PATH)
with h5py.File(DATA_PATH, "r") as handle:
    raw = np.column_stack([handle[k][:32768] for k in ("X", "Y", "Z")])
fs = 256.
data = signal.resample_poly(raw, up=1, down=8, axis=0)
data -= data.mean(axis=0)
assert data.shape == (4096, 3) and np.all(np.isfinite(data))
print("ET sample after anti-aliased downsampling:", data.shape, fs)

```

```{code-cell} ipython3
start = perf_counter()
estimator = PSDEstimator(data, fs=fs, reference_psd=None,
                         parallel_rows=True, fmin_for_analysis=10., fmax_for_analysis=100., **BUDGET)
_ = estimator.run(lr=.003)  # Explicit rate: no costly hyperparameter search.
runtime = perf_counter() - start
assert np.all(np.isfinite(estimator.pointwise_ci))
assert np.all(np.isfinite(estimator.inference_runner.kdl_losses))
print(f"{len(data)} samples × {data.shape[1]} channels; fit {runtime:.2f} s")
if estimator.parallel_rows:
    print("Independent vmapped Cholesky-row dimensions:", estimator.inference_runner.row_dimensions)

```

```{code-cell} ipython3
def show_fit(estimator, data, labels, truth=None, reference=None):
    """Two-sided densities and magnitude-squared coherence, no hidden rescaling."""
    from matplotlib.ticker import NullFormatter
    p = data.shape[1]
    fig, axes = plt.subplots(1, p, figsize=(4*p, 3), squeeze=False)
    for i, ax in enumerate(axes[0]):
        fw, pw = signal.welch(data[:, i], fs=estimator.fs, window="boxcar",
                              nperseg=estimator.nt_per_chunk, noverlap=0,
                              detrend=False, return_onesided=False)
        keep = (fw >= estimator.freq[0]) & (fw <= estimator.freq[-1])
        ax.loglog(fw[keep], pw[keep], color="0.65", lw=.7, label="Blocked periodogram")
        q = estimator.pointwise_ci[:, :, i, i].real
        ax.fill_between(estimator.freq, q[0], q[2], alpha=.25)
        ax.loglog(estimator.freq, q[1], label="SGVB median")
        if truth is not None:
            ax.loglog(estimator.freq, truth(estimator.freq)[:, i, i].real, "k--", lw=1, label="Known simulation truth")
        if reference is not None:
            ax.loglog(estimator.freq, reference(estimator.freq)[:, i, i].real, ":", color="C2", lw=1, label="Fixed reference")
        ax.set(title=labels[i], xlabel="Frequency [Hz]", ylabel="Two-sided PSD [input units²/Hz]")
        ticks = np.geomspace(estimator.freq[0], estimator.freq[-1], 4)
        ax.set_xticks(ticks, labels=[f"{f:.2g}" for f in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", labelsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout(); plt.show()
    if p > 1:
        fig, ax = plt.subplots(figsize=(6, 3))
        s = estimator.psd_all
        for i in range(p):
            for j in range(i):
                coh = abs(s[:, :, i, j])**2/(s[:, :, i, i].real*s[:, :, j, j].real)
                q = np.quantile(coh, [.05, .5, .95], axis=0)
                line, = ax.plot(estimator.freq, q[1], label=f"{labels[i]}–{labels[j]}")
                ax.fill_between(estimator.freq, q[0], q[2], alpha=.15, color=line.get_color())
                if truth is not None:
                    known = truth(estimator.freq)
                    known_coh = abs(known[:, i, j])**2/(known[:, i, i].real*known[:, j, j].real)
                    ax.plot(estimator.freq, known_coh, "--", color=line.get_color(), label="Simulation truth")
        ax.set(xlabel="Frequency [Hz]", ylabel="Magnitude-squared coherence", ylim=(0, 1))
        ax.legend(); fig.tight_layout(); plt.show()

```

```{code-cell} ipython3
show_fit(estimator, data, ["ET X", "ET Y", "ET Z"])

```

## Interpretation and budget

The shaded bands are pointwise 90% variational intervals. Cheap settings
show the workflow; they do not establish convergence or calibrated coverage.
Increase the data duration, basis size and optimization budgets only for an
explicit analysis/validation plan. Compilation is included in the first-fit
time. JAX batches rows with `vmap`; actual CPU/GPU scheduling is device dependent.
Each row still includes regression on preceding channels, so cross-spectra
are retained. This is modified-Cholesky SGVB, not the removed eigenbasis mode or the
collapsed conditional-theta method in LogPSplinePSD.
