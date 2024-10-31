import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import r

from sgvb_psd.postproc.plot_psd import plot_psdq
from sgvb_psd.psd_estimator import PSDEstimator
from sgvb_psd.utils.sim_varma import SimVARMA

# Import the necessary R packages
vnpc_avg = rpackages.importr("vnpc.avg")
beyond_whittle = rpackages.importr("beyondWhittle")


def generate_data(seed_number: int = 0) -> np.ndarray:
    r(f"set.seed({seed_number})")
    param_ar = r["rbind"](
        ro.FloatVector([0.1, 0, 0, 0]), ro.FloatVector([0, -0.3, 0, -0.5])
    )
    param_ma = r["matrix"](nrow=2, ncol=0)
    param_sigma = r["matrix"](ro.FloatVector([1, 0.9, 0.9, 1]), nrow=2, ncol=2)
    data = beyond_whittle.sim_varma(
        model=r.list(ar=param_ar, ma=param_ma, sigma=param_sigma), n=512, d=2
    )
    return np.array(data)


def load_vnpc_results(fpath="mcmc_results.RData"):
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
    return psd, coh, data


def plot():
    pass


if __name__ == "__main__":
    mcmc_psd, mcmc_coh, data = load_vnpc_results()
    psd_estimator = PSDEstimator(
        x=data,
        N_theta=40,
        nchunks=4,
        fs=2 * np.pi,
        N_samples=1000,
        max_hyperparm_eval=1000,
    )
    psd_estimator.run(lr=0.003)
    var2 = SimVARMA(
        n_samples=int((2**9) / 4),
        var_coeffs=np.array(
            [[[0.1, 0.0], [0.0, -0.3]], [[0.0, 0], [0.0, -0.5]]]
        ),
        vma_coeffs=np.array([[[0.0, 0.0], [0.0, 0.0]]]),
        sigma=np.array([[1.0, 0.9], [0.9, 1.0]]),
    )
    axes = plot_psdq(mcmc_psd, freqs=psd_estimator.freq, color="tab:red")
    axes = psd_estimator.plot(
        off_symlog=False,
        axes=axes,
        true_psd=[var2.psd, var2.freq],
        xlims=[0, np.pi],
    )
    plt.savefig("sgvb_mcmc_simulation_compare2.png", dpi=500)
