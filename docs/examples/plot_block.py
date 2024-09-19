import matplotlib.pyplot as plt
import numpy as np

N = 10000
N_CHANNELS = 2
n_blocks = 6
n_per_block = N // n_blocks

t = np.linspace(0, 100, N)
x = np.random.random((N_CHANNELS, n_blocks, n_per_block))

fig, axs = plt.subplots(N_CHANNELS, 1, figsize=(3, 2))


for i in range(N_CHANNELS):
    for j in range(n_blocks):
        _t = t[j * n_per_block : (j + 1) * n_per_block]
        axs[i].plot(_t, np.sin(25 * _t) + x[i, j], alpha=1)
    axs[i].annotate(
        f"Channel {i+1}",
        xy=(0.1, 0.7),
        xycoords="axes fraction",
        # fontcolor white
        color="black",
    )
    axs[i].set_yticks([])
    axs[i].set_xticks([])

fig.legend(
    handles=[
        plt.Line2D([0], [0], color=f"C{i}", lw=3, label=f"Block {i+1}")
        for i in range(n_blocks)
    ],
    loc="outside center right",
    frameon=False,
)
axs[-1].set_xlabel("Time [s]")

plt.subplots_adjust(hspace=0, right=0.5)
plt.savefig("./out_plots/plot_block.png")
