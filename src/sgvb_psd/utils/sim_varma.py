import numpy as np
import matplotlib.pyplot as plt


class SimVARMA:
    def __init__(self, n, varCoef, vmaCoef, sigma=np.array([1.])):
        self.n = n
        self.varCoef = varCoef
        self.vmaCoef = vmaCoef
        self.sigma = sigma
        self.data = self.simulate()
        self.periodogram = self.__compute_periodogram()
        self.psd = self.__compute_psd()

    def simulate(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        dim = self.vmaCoef.shape[1]
        lag_ma = self.vmaCoef.shape[0]
        lag_ar = self.varCoef.shape[0]

        if self.sigma.shape[0] == 1:
            Sigma = np.identity(dim) * self.sigma
        else:
            Sigma = self.sigma

        x_init = np.array(np.zeros(shape=[lag_ar + 1, dim]))
        x = np.empty((self.n + 101, dim))
        x[:] = np.NaN
        x[:lag_ar + 1, ] = x_init
        epsilon = np.random.multivariate_normal(np.repeat(0., dim), Sigma, size=[lag_ma, ])

        for i in np.arange(lag_ar + 1, x.shape[0]):
            epsilon = np.concatenate(
                [np.random.multivariate_normal(np.repeat(0., dim), Sigma, size=[1, ]), epsilon[:-1]])
            x[i,] = np.sum(np.matmul(self.varCoef, x[i - 1:i - lag_ar - 1:-1][..., np.newaxis]), axis=(0, -1)) + \
                    np.sum(np.matmul(self.vmaCoef, epsilon[..., np.newaxis]), axis=(0, -1))

        return x[101:]

    def __compute_periodogram(self):
        # Implement periodogram computation here
        pass

    def __compute_psd(self):
        # Implement PSD computation here (the true PSD)
        pass

    def plot(self, plot_periodogram=True, plot_psd=True, axes=None, pgrm_kwargs={}, psd_kwargs={}):
        # use plot_psd code
        pass



def sim_varma(n, varCoef, vmaCoef, sigma=np.array([1.])):
    
    dim = vmaCoef.shape[1]
    lag_ma = vmaCoef.shape[0]
    lag_ar = varCoef.shape[0]
    
    if sigma.shape[0] == 1:
        Sigma = np.identity(dim) * sigma
    else:
        Sigma = sigma
    
    x_init = np.array(np.zeros(shape = [lag_ar+1, dim]))
    x = np.empty((n+101, dim))
    x[:] = np.NaN
    x[:lag_ar+1, ] = x_init
    epsilon = np.random.multivariate_normal(np.repeat(0., dim), Sigma, size=[lag_ma,])
    for i in np.arange(lag_ar+1, x.shape[0]):
        epsilon = np.concatenate([np.random.multivariate_normal(np.repeat(0., dim), Sigma, size=[1,]), epsilon[:-1]])
        x[i,] = np.sum(np.matmul(varCoef, x[i-1:i-lag_ar-1:-1][...,np.newaxis]), axis=(0, -1)) + \
                np.sum(np.matmul(vmaCoef, epsilon[...,np.newaxis]), axis=(0, -1)) 
    x = x[101: ]        
    return x















