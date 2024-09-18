from pathlib import Path

import matplotlib

# Absolute path to the top level of the repository
rc_file = Path(__file__).resolve().absolute().parents[0] / "matplotlibrc"
matplotlib.rc_file(rc_file)

from .plot_psd import format_axes, plot_peridogram, plot_psdq, plot_single_psd
from .plot_coherence import plot_coherence
from .psd_analyzer import PSDAnalyzer
