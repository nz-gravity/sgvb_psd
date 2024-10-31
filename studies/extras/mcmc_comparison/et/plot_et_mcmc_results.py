import h5py
import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import r

from sgvb_psd.postproc.plot_psd import (
    format_axes,
    plot_peridogram,
    plot_psd,
    plot_psdq,
)
from sgvb_psd.utils.periodogram import get_periodogram

RESULT = "et_data.h5"
DATA = dict(
    A="ET_caseA_noise.h5",
    B="ET_caseB_noise.h5",
    C="ET_caseB_noise.h5",  # Case C uses the same data as case B
)


class Case:
    def __init__(self, case):
        self.case = case
        self.result_fn = RESULT
        self.data_fn = DATA[case]
        self.psdq = self._load_estimated_psd_quantiles()
        self.true = self._load_true_psd()
        self.data = self._load_periodogram()
        print("Loaded case ", self)

    def _load_true_psd(self):
        with h5py.File(self.data_fn, "r") as f:
            freq = f["true_psd/freq"][:]
            idx = (freq > 5) & (freq < 128)
            psd = f["true_psd/psd"][:].T[idx]

            # PSD is currnelty just the XX, YY, ZZ components
            # make it a 3D array (nans for off-diagonal elements)
            psd = np.array([np.diag(psd[i]) for i in range(psd.shape[0])])
            return [psd, freq[idx]]

    def _load_estimated_psd_quantiles(self):
        with h5py.File(self.result_fn, "r") as f:
            return [f[f"case{self.case}_psd_quantiles"][:], f[f"freq"][:]]

    def _load_periodogram(self):
        with h5py.File(self.data_fn, "r") as f:
            freq = f["periodogram/freq"][:]
            idx = (freq > 5) & (freq < 128)
            return [f["periodogram/pdgrm"][:][idx], freq[idx]]

    def __repr__(self):
        return (
            f"Case {self.case} ["
            f"psdq: {self.psdq[0].shape}, "
            f"true: {self.true[0].shape}, "
            f"data: {self.data[0].shape}]"
        )

    def plot(self, axes=None, fname=None, **kwargs):
        axes = plot_psd(
            self.psdq,
            self.true,
            self.data,
            axes=axes,
            xlims=[5, 128],
            diag_ylims=[1e-52, 1e-46],
            off_ylims=[-3e-47, 3e-47],
            **kwargs,
        )
        if fname is not None:
            plt.savefig(fname)
            print(f"Saved {fname}")
        return axes


def load_vnpc_results(fpath="mcmc_results.RData"):
    # Import the necessary R packages
    vnpc_avg = rpackages.importr("vnpc.avg")
    beyond_whittle = rpackages.importr("beyondWhittle")

    r["load"](fpath)
    r_obj = r["mcmc_vnp_avg"]
    result_dict = {}

    # Iterate over the ListVector items (names and values)
    for name, value in zip(r_obj.names, r_obj):
        # Convert R objects to appropriate Python/Numpy structures
        if isinstance(value, ro.vectors.FloatMatrix):
            result_dict[name] = np.array(value)
        elif isinstance(value, ro.vectors.FloatArray):
            result_dict[name] = np.array(value)
        else:
            result_dict[name] = value  # Keep as it is for other data types

    coh = np.array(
        [
            result_dict["coherence.p05"],
            result_dict["coherence.median"],
            result_dict["coherence.p95"],
        ]
    )
    data = result_dict["data"]
    psd = np.array(
        [  # pointwise
            result_dict["psd.p05"],
            result_dict["psd.median"],
            result_dict["psd.p95"],
        ]
    )

    x, d, d, nfreq = psd.shape
    # reshape to x, nfreq. d, d
    psd = np.transpose(psd, (0, 3, 1, 2))
    # cut 0th and last nfreq
    psd = psd[:, 1:-1, :, :]

    psd = psd.astype(np.complex128)
    # make lower triangle elements imaginary by multiplying by 1j
    for i in range(d):
        for j in range(d):
            if i > j:
                psd[:, :, i, j] *= 1j

    # rescale PSDs (1e+22 is approx std for data)
    psd = psd / 1e22**2 / (1024.0 / np.pi)
    freq = np.linspace(5, 128, nfreq)

    return psd, freq


if __name__ == "__main__":
    mcmc_psd, mcmc_freq = load_vnpc_results("et_mcmc.RData")
    axes = Case("A").plot(color="C1")
    axes = plot_psdq(mcmc_psd, freqs=mcmc_freq, color="tab:red", axes=axes)
    plt.savefig("mcmc_et.png", dpi=500)
