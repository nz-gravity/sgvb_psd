import click


@click.command("sgvb_benchmark")
@click.option(
    '-s',
    '--minlog2n',
    default=3,
    help='Minimum log2(n) for the benchmark.',
)
@click.option(
    '-e',
    '--maxlog2n',
    default=4,
    help='Maximum log2(n) for the benchmark.',
)
@click.option(
    '-f',
    '--fname',
    default='timings.txt',
    help='Filename for the benchmark results.',
)
@click.option(
    '-n',
    '--nrep',
    default=5,
    help='Number of repetitions for the benchmark.',
)
def cli(minlog2n, maxlog2n, fname, nrep):
    """
    Command line interface for benchmarking the PSD estimator.
    """
    from .benchmark import benchmark

    benchmark(minLog2n=minlog2n, maxLog2n=maxlog2n, fname=fname, nrep=nrep)
