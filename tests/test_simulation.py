import matplotlib.pyplot as plt
import numpy as np

from sgvb_psd.optimal_psd_estimator import OptimalPSDEstimator
from sgvb_psd.utils import sim_varma
from sgvb_psd.utils.periodogram import get_periodogram
from sgvb_psd.postproc.plot_psd import plot_peridogram

def test_sim_varma():
    ar = np.array([
        [[0.5, 0], [0, -0.3]],
        [[0, 0], [0, -0.5]]
    ])
    Sigma = np.array([[1, 0.9], [0.9, 1]])

    ar = np.array([[0.5, 0, 0, 0], [0, -0.3, 0, -0.5]])
    n = 256
    d = 2

    # Simulate the VAR(2) process
    x = sim_varma(model='ar', coeffs=ar, n=n, d=d, sigma=Sigma)

    # Plot the time series
    plt.figure(figsize=(10, 6))
    plt.plot(x)
    plt.title('Simulated VAR(2) Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend([f'Series {i + 1}' for i in range(d)])
    plt.grid(True)
    plt.show()

    optim = OptimalPSDEstimator(
        N_theta=30, nchunks=1, duration=1, ntrain_map=100, x=x, max_hyperparm_eval=1

    )
    optim.run()
    optim.plot()
    plt.show()



def test_pdgmr():
    Sigma = np.array([[1, 0.9], [0.9, 1]])

    ar = np.array([[0.5, 0, 0, 0], [0, -0.3, 0, -0.5]])
    n = 256
    d = 2

    # Simulate the VAR(2) process
    x = sim_varma(model='ar', coeffs=ar, n=n, d=d, sigma=Sigma)
    f, pdgrm = get_periodogram(x, fs=2*np.pi)
    assert pdgrm.shape == (129, 2, 2)

    axes = plot_peridogram(x, fs=2*np.pi)
    plt.gcf()
    plt.savefig('test.png')
