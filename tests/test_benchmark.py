from click.testing import CliRunner
from sgvb_psd.utils.benchmark.cli import cli
import os

def test_benchmark(plot_dir):
    runner = CliRunner()
    result = runner.invoke(cli, ['-s', '4', '-e', '5', '-f', f'{plot_dir}/timings.txt'])
    assert result.exit_code == 0
    assert os.path.exists(f'{plot_dir}/timings.txt')
