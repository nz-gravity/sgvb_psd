import logging
import os

import emcee
import matplotlib.pyplot as plt
import numpy as np
from sgvb_psd.postproc import plot_psdq
from sgvb_psd.psd_estimator import PSDEstimator
from sgvb_psd.utils.sim_varma import SimVARMA
from tqdm.auto import trange

# Set logging level
logging.getLogger("SGVB-PSD").setLevel('ERROR')

# Constants
N_EXPERIMENTS = 10
FS = 100
OUTDIR = 'out_gibbs'
os.makedirs(OUTDIR, exist_ok=True)

# Simulation and VI configurations
SIM_KWGS = dict(
    sigma=np.array([[1.0, 0.9], [0.9, 1.0]]),
    var_coeffs=np.array([[[0.5, 0.0], [0.0, -0.3]], [[0.0, 0.0], [0.0, -0.5]]]),
    vma_coeffs=np.array([[[1.0, 0.0], [0.0, 1.0]]]),
    n_samples=1024,
)

VI_KWGS = dict(
    N_theta=30,
    nchunks=1,
    ntrain_map=1000,
    max_hyperparm_eval=1,
    fs=FS,
    seed=0,
    N_samples=1,
)


# Signal model function
def signal_model(t: np.ndarray, f0: float, a: float = 5.0):
    y = a * np.sin(2 * np.pi * f0 * t)
    return np.array([y, y]).T


# Plot simulation PSD
def plot_simulation_psd(optim: PSDEstimator, sim: SimVARMA):
    axs = optim.plot(
        true_psd=[sim.psd, sim.freq],
        off_symlog=False,
        xlims=[0, FS / 2.0],
        quantiles='pointwise',
    )
    plot_psdq(
        optim.uniform_ci,
        freqs=optim.freq,
        axs=axs,
        color='red',
        ls='--',
    )
    return axs




# Utility functions
def get_univar_psd(psd_matrix):
    return np.real(np.array([psd_matrix[:, 0, 0], psd_matrix[:, 1, 1]])).T


def log_prob(theta, xf, psd):
    f0 = theta[0]
    if not 0 < f0 < FS / 2:
        return -np.inf

    signal = signal_model(t, f0)
    signal_fft = np.fft.rfft(signal, axis=0)
    residual = xf - signal_fft
    residual = residual[1:-1]
    return -0.5 * np.sum(residual ** 2 / psd)



# Run MCMC
def run_mcmc(data, psd, x0=None):
    fft_data = np.fft.rfft(data, axis=0)
    univar_psd = get_univar_psd(psd)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(fft_data, univar_psd))
    if x0 is None:
        x0 = np.random.normal(true_f0, 1, size=(nwalkers, ndim))

    assert x0.shape == (nwalkers, ndim), x0.shape
    sampler.run_mcmc(x0, nsteps)
    samples = sampler.get_chain(discard=int(nsteps * 0.5), flat=True)
    x0 = sampler.get_chain()[-1, :, :]
    return x0, samples


# Run SGVB
def run_sgvb(data, x0, psd_x0=None):
    x0_mean = np.mean(x0)
    signal = signal_model(t, x0_mean)
    optim = PSDEstimator(**VI_KWGS, x=data - signal, init_params=psd_x0)
    optim.run(lr=0.003)
    psd_x0, psd = optim.sample_posterior(1)
    psd = psd[0][0]
    return psd_x0, psd


# Plot 1D PSD
def plot_1d(psd, x0, data, idx=None):
    plt.close('all')
    psd = get_univar_psd(psd).T[0]
    x0_mean = np.mean(x0)
    signal = signal_model(t, x0_mean)
    n = len(data)
    fft_signal = np.fft.rfft(signal.T[0], axis=0)[1:-1]
    fft_data = np.fft.rfft(data.T[0], axis=0)[1:-1]
    freq = np.fft.rfftfreq(n, d=1 / FS)[1:-1]

    plt.plot(freq, np.abs(fft_data), color='gray', label='data')
    plt.loglog(freq, psd, color='red', label='psd')
    plt.plot(freq, np.abs(fft_signal), color='blue', ls='--', label='signal')
    plt.legend()
    plt.xlabel('Frequency (Hz)')
    if idx is not None:
        plt.title(f'Iteration {idx}')
    plt.tight_layout()
    plt.show()


def plot_gibbs_results(fname):
    dataset = np.load(fname)

    x0s = dataset['x0s']
    psds = dataset['psds']
    data = dataset['data']

    n_psds = psds.shape[0]

    # lets grab last 10% of samples
    n_psds = int(n_psds * 0.1)

    # just grab n_psd samples for x0s
    x0s = x0s[:n_psds]

    signals = np.array([signal_model(t, x0) for x0 in x0s])[..., 0]
    signal_fft = np.abs(np.fft.rfft(signals, axis=1))[..., 1:-1]
    data_fft = np.abs(np.fft.rfft(data, axis=0))[1:-1]
    freq = np.fft.rfftfreq(len(data), d=1 / FS)[1:-1]

    signal_fft_ci = np.percentile(signal_fft, [5.0, 95.0], axis=0)
    psd_ci = np.percentile(np.real(psds), [5.0, 95.0], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    ax = axes[0]
    all_x0s = dataset['x0s']
    # keep only last 10% of samples
    all_x0s = all_x0s[int(all_x0s.shape[0] * 0.9):]
    ax.hist(all_x0s, color='gray', alpha=0.5)
    ax.axvline(true_f0, color='red', ls='--')
    ax.set_xlabel('f0 samples')

    ax = axes[1]
    ax.loglog(freq, data_fft, color='gray', label='data')
    ax.fill_between(freq, psd_ci[0], psd_ci[1], color='red', alpha=0.3, label='psd')
    ax.fill_between(freq, signal_fft_ci[0], signal_fft_ci[1], color='blue', alpha=0.3, label='signal')
    ax.legend()
    ax.set_xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(fname.replace('.npz', '.png'))


# Initialize simulation
sim = SimVARMA(**SIM_KWGS, seed=0)
dt = 1 / FS
t = np.linspace(0, (sim.n_samples - 1) * dt, sim.n_samples)
true_f0 = 5.0
true_signal = signal_model(t, true_f0)
sim.data += true_signal

# MCMC parameters
nwalkers = 16
ndim = 1
nsteps = 1000




def main():
    # Gibbs sampling
    gibbs = 500
    x0 = np.random.normal(true_f0, 1, size=(nwalkers, ndim))
    x0s = []
    psds = []
    psd_x0 = None

    for i in trange(gibbs, desc='Gibbs sampling'):
        x0, samples = run_mcmc(sim.data, sim.psd, x0)
        psd_x0, psd = run_sgvb(sim.data, x0, psd_x0)
        x0s.append(samples)
        psds.append(psd[:, 0, 0])
        if i % 10 == 0 and i > 0:
            plot_1d(psd, x0, sim.data, i)

    x0s = np.array(x0s).ravel()
    psds = np.array(psds)
    np.savez(f'{OUTDIR}/gibbs.npz', x0s=x0s, psds=psds, data=sim.data.T[0])

    plot_gibbs_results(f'{OUTDIR}/gibbs.npz')


if __name__ == '__main__':
    main()
