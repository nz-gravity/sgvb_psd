
"""
Simulate true VAR(2) time series. 

"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import tensorflow as tf


class VarmaSim:
    def __init__(self, n):
        self.n = n
    
    def simData(self, varCoef, vmaCoef, sigma=np.array([1.])):
        n = self.n
        dim = vmaCoef.shape[1] #the number of column of vmaCoef = 2
        lag_ma = vmaCoef.shape[0] #the number of row of vmaCoef = 3
        lag_ar = varCoef.shape[0] #the number of row of varCoef = 2
        
        if sigma.shape[0] == 1:
            Sigma = np.identity(dim) * sigma
        else:
            Sigma = sigma
        
        x_init = np.array(np.zeros(shape = [lag_ar+1, dim])) #zero matrix 3 rows if lag_ar+1=2+1, 2 colmuns if dim=2
        x = np.empty((n+101, dim)) #1024+101 rows, 2 columns nearly zero matrix
        x[:] = np.NaN
        x[:lag_ar+1, ] = x_init
        #epsilon = np.random.multivariate_normal(np.repeat(0., dim), Sigma, size=[lag_ma,]) #result is a 3x2 martix
        for i in np.arange(lag_ar+1, x.shape[0]): #from lag_ar+1 i.e.3 to(# of x)-1 i.e.1125-1
            epsilon = np.random.multivariate_normal(np.repeat(0., dim), Sigma)
            x[i,] = np.sum(np.matmul(varCoef, x[i-1:i-lag_ar-1:-1][...,np.newaxis]), axis=(0, -1)) + \
                    np.matmul(vmaCoef, epsilon) 
        x = x[101: ]        
        return x

    def calculateSpecMatrix(self, f, varCoef, vmaCoef, sigma=np.array([1.]), inverse = False):
        specTrue = np.apply_along_axis(lambda f: self.calculateSpecMatrixHelper(f, varCoef, vmaCoef, sigma, inverse=inverse), axis=1, arr = f.reshape(-1,1))
        return specTrue
        
    def calculateSpecMatrixHelper(self, f, varCoef, vmaCoef, sigma=np.array([1.]), inverse = False):
        # f:     a single frequency
        # Phi:   AR coefficient array. M-by-p-by-p. i-th row imply lag (M-i), p is dim of time series.
        dim = vmaCoef.shape[1]
        if sigma.shape[0] == 1:
            Sigma = np.identity(dim) * sigma
        else:
            Sigma = sigma
        
        k_ar = np.arange(1, varCoef.shape[0]+1, 1)
        A_f_re_ar = np.sum(varCoef * np.cos(np.pi*2*k_ar*f)[:, np.newaxis, np.newaxis], axis = 0)
         
        A_f_im_ar = - np.sum(varCoef * np.sin(np.pi*2*k_ar*f)[:, np.newaxis, np.newaxis], axis = 0)
        A_f_ar = A_f_re_ar + 1j * A_f_im_ar
        A_bar_f_ar = np.identity(dim) - A_f_ar
        H_f_ar = np.linalg.inv(A_bar_f_ar) #create inverse matrices for A_bar_f_ar
        

        if inverse == False:
            Spec_mat = H_f_ar  @ Sigma  @ H_f_ar.conj().T
            return Spec_mat
  

if __name__ == '__main__':
    
    n = 1024
    sigma = np.array([[1., 0.9], [0.9, 1.]])  
    varCoef = np.array([[[0.5, 0.], [0., -0.3]], [[0., 0.], [0., -0.5]]])
    vmaCoef = np.array([[[1.,0.],[0.,1.]]])
    
    Simulation = VarmaSim(n=n)
    x = Simulation.simData(varCoef, vmaCoef, sigma=sigma)
    freq =  np.arange(1,np.floor_divide(n, 2)+1, 1) / (n)
    #np.floor_divide(500*2, 2)=500.
    specTrue = Simulation.calculateSpecMatrix(freq, varCoef, vmaCoef, sigma)
    
    fig, ax = plt.subplots(1,5, figsize = (11, 5))
    for i in range(2):
        f, Pxx_den0 = signal.periodogram(x[:,i], fs=1)
        #create periodgram for each column of x, f is Fourier frequencies.
        f = f[1:]
        Pxx_den0 = Pxx_den0[1:] / 2
        ax[i].plot(f, np.log(Pxx_den0), marker = '.', markersize=2, linestyle = 'None')
        ax[i].plot(freq, np.log(np.real(specTrue[:,i,i])), linewidth=2, color = 'red', linestyle="-.", label = 'Truth')
        ax[i].set_xlim([0, 0.5])
        ax[i].grid(True)
    ax[2].plot(freq, np.absolute(specTrue[:,0,1])**2 / (np.real(specTrue[:,0,0]) * np.real(specTrue[:,1,1])), linewidth=2, color = 'red', linestyle="-.", label = 'Truth')
    ax[2].set_xlim([0,0.5])
    ax[2].set_ylim([0., 1.])
    ax[2].grid(True)
    ax[3].plot(freq, np.real(specTrue[:,0,1]), linewidth=2, color = 'red', linestyle="-.", label = 'Truth')
    ax[3].grid(True)
    ax[4].plot(freq, np.imag(specTrue[:,0,1]), linewidth=2, color = 'red', linestyle="-.", label = 'Truth')
    ax[4].grid(True)
    plt.tight_layout()
    plt.show()
        

    



    
    
    
    
    






