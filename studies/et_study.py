import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

from sgvb_psd.psd_estimator import PSDEstimator

DATA_FILES = dict(
    caseA="ET-CaseA-noise.h5",
    caseB="ET-CaseB-noise.h5",
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


def run_analysis(case, label=""):
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
    with h5py.File(f"ET-{case}{label}-SGVB-PSD.h5", "w") as f:
        f["freq"] = psd_estimator.freq
        f["psd_pointwise_ci"] = psd_estimator.pointwise_ci
        f["psd_uniform_ci"] = psd_estimator.uniform_ci
    psd_estimator.plot(
        true_psd=[data.true_psd, data.true_freq],
        xlims=[5, 128],
    )
    plt.savefig(f"ET-{case}{label}-SGVB-PSD.png")


def combine_results():
    # load each PSD-quantile file
    psd_pointwise = {}
    psd_uniform = {}
    for case in DATA_FILES:
        with h5py.File(f"ET-{case}-SGVB-PSD.h5", "r") as f:
            psd_pointwise[case] = f["psd_pointwise_ci"][:]
            psd_uniform[case] = f["psd_uniform_ci"][:]
            freq = f["freq"][:]

    fpath = "ET-Case-ABC-SGVB-PSD.h5"
    print(f"Creating file: {fpath}")
    with h5py.File(fpath, "w") as f:
        for case in DATA_FILES:
            f.create_dataset(
                f"{case}/psd_pointwise_ci", data=psd_pointwise[case]
            )
            f.create_dataset(f"{case}/psd_uniform_ci", data=psd_uniform[case])
        f["freq"] = freq
    print(f"Data saved to: {fpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Run PSD analysis for a specified case."
    )
    parser.add_argument(
        "--case",
        choices=DATA_FILES.keys(),
        required=True,
        help="Specify the case to analyze (caseA or caseB).",
    )
    parser.add_argument(
        "--label", type=str, required=True, help="Extra label for result."
    )

    args = parser.parse_args()

    run_analysis(args.case, args.output)
    # combine_results()  # Uncomment this if you want to combine results after analysis


if __name__ == "__main__":
    main()
