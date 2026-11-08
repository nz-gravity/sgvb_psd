from pathlib import Path

import matplotlib


def set_rcparms():
    # Absolute path to the top level of the repository
    rc_file = Path(__file__).resolve().absolute().parents[0] / "matplotlibrc"
    matplotlib.rc_file(rc_file)


set_rcparms()

from .plot_coherence import plot_coherence
from .plot_psd import format_axes, plot_peridogram, plot_psdq, plot_single_psd
from .psd_analyzer import PSDAnalyzer
