'create true psd for simulation study'
import numpy as np

class VarmaSim:
    def __init__(self, n=1024):
        self.n = n
    
    def calculateSpecMatrix(self, f, varCoef, vmaCoef, sigma=np.array([1.]), inverse=False):
        specTrue = np.apply_along_axis(
            lambda f: self.calculateSpecMatrixHelper(f, varCoef, vmaCoef, sigma, inverse=inverse), 
            axis=1, 
            arr=f.reshape(-1, 1)
        )
        return specTrue
        
    def calculateSpecMatrixHelper(self, f, varCoef, vmaCoef, sigma=np.array([1.]), inverse=False):
        # f:     a single frequency
        # Phi:   AR coefficient array. M-by-p-by-p. i-th row implies lag (M-i), p is dim of time series.
        dim = vmaCoef.shape[1]
        if sigma.shape[0] == 1:
            Sigma = np.identity(dim) * sigma
        else:
            Sigma = sigma
        
        k_ar = np.arange(1, varCoef.shape[0] + 1, 1)
        A_f_re_ar = np.sum(varCoef * np.cos(np.pi * 2 * k_ar * f)[:, np.newaxis, np.newaxis], axis=0)
        A_f_im_ar = -np.sum(varCoef * np.sin(np.pi * 2 * k_ar * f)[:, np.newaxis, np.newaxis], axis=0)
        A_f_ar = A_f_re_ar + 1j * A_f_im_ar
        A_bar_f_ar = np.identity(dim) - A_f_ar
        H_f_ar = np.linalg.inv(A_bar_f_ar)
        
        k_ma = np.arange(vmaCoef.shape[0])
        A_f_re_ma = np.sum(vmaCoef * np.cos(np.pi * 2 * k_ma * f)[:, np.newaxis, np.newaxis], axis=0)
        A_f_im_ma = -np.sum(vmaCoef * np.sin(np.pi * 2 * k_ma * f)[:, np.newaxis, np.newaxis], axis=0)
        A_f_ma = A_f_re_ma + 1j * A_f_im_ma
        A_bar_f_ma = A_f_ma
        H_f_ma = A_bar_f_ma

        if inverse == False:
            Spec_mat = H_f_ar @ H_f_ma @ Sigma @ H_f_ma.conj().T @ H_f_ar.conj().T
            return Spec_mat
        else:
            Spec_inv = A_bar_f_ar.conj().T @ A_bar_f_ma.conj().T @ np.linalg.inv(Sigma) @ A_bar_f_ma @ A_bar_f_ar    
            return Spec_inv
