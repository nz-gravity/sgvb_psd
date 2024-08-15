
"""
Simulate true VMA(1) time series. 

"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import tensorflow as tf


class VarmaSim:
    def __init__(self, n=256):
        self.n = n
    
    def simData(self, vmaCoef, sigma=np.array([1.])):
        n = self.n
        dim = vmaCoef.shape[1] #the number of column of vmaCoef = 2
        lag_ma = vmaCoef.shape[0] #the number of row of vmaCoef = 3
        
        if sigma.shape[0] == 1:
            Sigma = np.identity(dim) * sigma
        else:
            Sigma = sigma
        
        x = np.empty((n+101, dim)) #1024+101 rows, 2 columns nearly zero matrix
        x[:] = np.NaN
        epsilon = np.random.multivariate_normal(np.repeat(0., dim), Sigma, size=[lag_ma,]) #result is a 2x2 martix
        for i in np.arange(0, x.shape[0]): #from lag_ar+1 i.e.3 to(# of x)-1 i.e.1125-1
            epsilon = np.concatenate([np.random.multivariate_normal(np.repeat(0., dim), Sigma, size=[1,]), epsilon[:-1]]) #itration for wt, wt-1, wt-2
            x[i,] = np.sum(np.matmul(vmaCoef, epsilon[...,np.newaxis]), axis=(0, -1)) 
        x = x[101: ]        
        return x

    def calculateSpecMatrix(self, f, vmaCoef, sigma=np.array([1.]), inverse = False):
        specTrue = np.apply_along_axis(lambda f: self.calculateSpecMatrixHelper(f, vmaCoef, sigma, inverse=inverse), axis=1, arr = f.reshape(-1,1))
        return specTrue
        
    def calculateSpecMatrixHelper(self, f, vmaCoef, sigma=np.array([1.]), inverse = False):
        # f:     a single frequency
        # Phi:   AR coefficient array. M-by-p-by-p. i-th row imply lag (M-i), p is dim of time series.
        dim = vmaCoef.shape[1]
        if sigma.shape[0] == 1:
            Sigma = np.identity(dim) * sigma
        else:
            Sigma = sigma
        
        
        k_ma = np.arange(vmaCoef.shape[0])
        A_f_re_ma = np.sum(vmaCoef * np.cos(np.pi*2*k_ma*f)[:, np.newaxis, np.newaxis], axis = 0)
        A_f_im_ma = - np.sum(vmaCoef * np.sin(np.pi*2*k_ma*f)[:, np.newaxis, np.newaxis], axis = 0)
        A_f_ma = A_f_re_ma + 1j * A_f_im_ma
        A_bar_f_ma = A_f_ma #+ np.identity(dim) already included in the prev A_f_re,im steps
        H_f_ma = A_bar_f_ma

        if inverse == False:
            Spec_mat = H_f_ma @ Sigma @ H_f_ma.conj().T
            return Spec_mat
        else:
            Spec_inv = A_bar_f_ma.conj().T @ np.linalg.inv(Sigma) @ A_bar_f_ma   
            return Spec_inv


if __name__ == '__main__':
    
    n = 256
    sigma = np.array([[1., 0.5], [0.5, 1.]])  
    vmaCoef = np.array([[[1.,0.],[0.,1.]], [[-0.75, 0.5], [0.5, 0.75]]])
    
    Simulation = VarmaSim(n=n)
    x = Simulation.simData(vmaCoef, sigma=sigma)
    freq =  np.arange(1,np.floor_divide(n, 2)+1, 1) / (n)
    #np.floor_divide(500*2, 2)=500.
    specTrue = Simulation.calculateSpecMatrix(freq, vmaCoef, sigma)
    
    fig, ax = plt.subplots(1,4, figsize = (11, 5))
    for i in range(2):
        f, Pxx_den0 = signal.periodogram(x[:,i], fs=1)
        #create spectral density for each column of x, f is Fourier frequencies.
        f = f[1:]
        Pxx_den0 = Pxx_den0[1:] / 2
        ax[i].plot(f, np.log(Pxx_den0), marker = '.', markersize=2, linestyle = '-')
        ax[i].plot(freq, np.log(np.real(specTrue[:,i,i])), linewidth=2, color = 'red', linestyle="-.", label = 'Truth')
        ax[i].set_xlim([0, 0.5])
        ax[i].grid(True)
    ax[2].plot(freq, np.absolute(specTrue[:,0,1])**2 / (np.real(specTrue[:,0,0] * np.real(specTrue[:,1,1]))), linewidth=2, color = 'red', linestyle="-.", label = 'Truth')
    ax[2].set_xlim([0,0.5])
    ax[2].set_ylim([0., 1.])
    ax[2].grid(True)
    ax[3].plot(freq, -np.imag(specTrue[:,0,1]), linewidth=2, color = 'red', linestyle="-.", label = 'Truth')
    ax[3].grid(True)
    plt.tight_layout()
    plt.show()
    
    
    
    
    