import h5py
import matplotlib.pyplot as plt
import numpy as np

from sgvb_psd.psd_estimator import PSDEstimator

DATA_FILES = dict(
    caseA="ET-CaseA-noise.h5",
    caseB="ET-CaseB-noise.h5",
    caseC="ET-CaseC-noise.h5",
)
CHANNELS = "XYZ"


class Data:
    def __init__(self, case):
        self.case = case

        fname = DATA_FILES[case]
        with h5py.File(fname, "r") as f:
            self.freq = f["freq"][:]
            self.channels = np.array([f[c][:] for c in CHANNELS])
            self.true_psd = f["true_psd"][:]
            self.true_freq = None  # load this jianan -- idk how 'true_freq' is stored in the file


def run_analysis(case):
    data = Data(case)
    psd_estimator = PSDEstimator(
        x=data.channels,
        N_theta=400,
        nchunks=125,
        ntrain_map=1000,
        N_samples=500,
        fs=2048,
        max_hyperparm_eval=1,
        fmax_for_analysis=128,
        degree_fluctuate=400,
    )
    # we previously optimized the learning rate, now hardcoding it
    psd_estimator.run(lr=0.0189)
    # save the PSD quantiles
    with h5py.File(f"ET-{case}-SGVB-PSD.h5", "w") as f:
        f["freq"] = psd_estimator.freq
        f["psd_quantiles"] = psd_estimator.psd_quantiles
    psd_estimator.plot(
        true_psd=[data.true_psd, data.true_freq],
        xlims=[5, 128],
    )
    plt.savefig(f"ET-{case}-SGVB-PSD.png")


if __name__ == "__main__":
    for case in DATA_FILES:
        run_analysis(case)
