from pathlib import Path
import matplotlib

# Absolute path to the top level of the repository
rc_file = Path(__file__).resolve().absolute().parents[0] / "matplotlibrc"
matplotlib.rc_file(rc_file)

from .psd_analyzer import PSDAnalyzer
from .plot_psd import plot_psdq, plot_peridogram, plot_single_psd, format_axes