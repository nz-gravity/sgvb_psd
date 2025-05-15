import matplotlib.pyplot as plt
import numpy as np



def plot_timings(fname: str = 'timings.txt') -> None:
    data = np.loadtxt(fname)
    log2n, median_time, std_time = data.T
    n = 2 ** log2n

    plt_fname = fname.replace('.txt', '.png')
    fig, ax = plt.subplots()
    ax.fill_between(
        n,
        median_time - std_time,
        median_time + std_time,
        alpha=0.2,
        color="tab:blue",
    )
    ax.loglog(n, median_time, color="tab:blue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("Time (s)")
    plt.savefig(plt_fname, bbox_inches="tight")

plot_timings()