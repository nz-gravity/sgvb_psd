import matplotlib.pyplot as plt


def plot_losses(map_losses, kdl_losses, map_timing, kdl_timing):
    """
    Plot the posterior MAP values and KDL losses.

    Args:
        map_losses (list): List of posterior losses.
        kdl_losses (list): List of KDL losses.
    """
    fig, axes = plt.subplots(2,1, sharex=True)
    axes[0].plot(map_losses)
    axes[0].set_ylabel(r"$p(\theta | x)$")
    axes[1].plot(kdl_losses)
    axes[1].set_ylabel(r"$D_{KL}$")

    for ax in axes:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), )


    axes[1].set_xlabel("Iteration")
    axes[0].text(0.05, 0.8, f"Phase 1: {map_timing:.2f}s", transform=axes[0].transAxes, fontsize=10)
    axes[1].text(0.05, 0.8, f"Phase 2: {kdl_timing:.2f}s", transform=axes[1].transAxes, fontsize=10)
    plt.subplots_adjust(hspace=0)
    return axes

