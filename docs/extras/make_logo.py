from sgvb_psd.utils.sim_varma import SimVARMA
import numpy as np
from sgvb_psd.utils.tf_utils import set_seed
from sgvb_psd.psd_estimator import PSDEstimator
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

sigma = np.array([[1.0, 0.9], [0.9, 1.0]])
varCoef = np.array([[[0.5, 0.0], [0.0, -0.3]], [[0.0, 0.0], [0.0, -0.5]]])
vmaCoef = np.array([[[1.0, 0.0], [0.0, 1.0]]])
n = 256
set_seed(0)
var = SimVARMA(
    n_samples=n, var_coeffs=varCoef, vma_coeffs=vmaCoef, sigma=sigma
)

estimator = PSDEstimator(
    x=var.data, fs=2 * np.pi, N_theta=35, N_samples=50, ntrain_map=50
)
estimator.run(lr=0.01)


TXT = "SGVB"


axs = estimator.plot(
    plot_periodogram=False,
    tick_ln=0,
    xlims=[0, np.pi],
    off_symlog=False,
    add_axes_labels=False,
    add_channel_labels=False,
    alpha=0,
)
for i, ax in enumerate(axs.flatten()):
    ax.set_xticks([])
    ax.set_yticks([])

    # turn off all spines (turn off axes frame)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.annotate(
        TXT[i],
        xy=(0.5, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=100,
        color="C0",
        fontweight="bold",
    )
plt.savefig("../_static/logo.png")
