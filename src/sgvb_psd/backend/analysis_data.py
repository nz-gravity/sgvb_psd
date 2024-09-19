import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..logging import logger

tfd = tfp.distributions
tfb = tfp.bijectors


class AnalysisData:  # Parent used to create BayesianModel object
    def __init__(self, x, nchunks=128, fmax_for_analysis=128, fs=2048):
        # x:      N-by-p, multivariate timeseries with N samples and p dimensions
        # ts:     time series x
        # y_ft:   fourier transformed time series
        # freq:   frequencies w/ y_ft
        # p_dim:  dimension of ts
        # Xmat:   basis matrix
        # Zar:    arry of design matrix Z_k for every freq k
        self.ts = x
        if x.shape[1] < 2:
            raise Exception("Time series should be at least 2 dimensional.")

        self.y_ft = []  # inital
        self.freq = []
        self.p_dim = []
        # self.Xmat = []
        self.Zar = []
        self.nchunks = nchunks

        self.fs = fs
        self.fmax_for_analysis = fmax_for_analysis

    # scaled fft and get the elements of freq = 1:[Nquist]
    # discarding the rest of freqs
    def sc_fft(self):
        # x is a n-by-p matrix
        # unscaled fft
        x = self.ts

        num_segments = self.nchunks

        fmax_for_analysis = self.fmax_for_analysis
        len_chunk = x.shape[0] // num_segments
        x = np.array(
            np.split(x[0 : len_chunk * num_segments, :], num_segments)
        )

        y = []
        for i in range(num_segments):
            y_fft = np.apply_along_axis(np.fft.fft, 0, x[i])
            y.append(y_fft)
        y = np.array(y)

        # scale it
        n = x.shape[1]
        y = y / np.sqrt(n)  # np.sqrt(n)
        # discard 0 freq

        Ts = 1
        fq_y = np.fft.fftfreq(np.size(x, axis=1), Ts)
        ftrue_y = np.fft.fftfreq(n, d=1 / self.fs)

        if np.mod(n, 2) == 0:  # n is even
            idx = int(n / 2)
        else:  # n is odd
            idx = int((n - 1) / 2)

        y = y[:, 0:idx, :]
        fq_y = fq_y[0:idx]
        ftrue_y = ftrue_y[0:idx]

        if fmax_for_analysis is None:
            fmax_for_analysis = ftrue_y[-1]
        fmax_idx = np.searchsorted(ftrue_y, fmax_for_analysis)
        y = y[:, 0:fmax_idx, :]
        fq_y = fq_y[0:fmax_idx]
        p_dim = x.shape[2]

        self.y_ft = y
        self.freq = fq_y
        self.p_dim = p_dim
        self.num_obs = fq_y.shape[0]

        return dict(y=y, fq_y=fq_y, p_dim=p_dim)

    # Demmler-Reinsch basis for linear smoothing splines (Eubank,1999)
    def DR_basis(self, N=10):
        # nu: vector of frequences
        # N:  amount of basis used
        # return a len(nu)-by-N matrix
        nu = self.freq
        basis = np.array(
            [
                np.sqrt(2) * np.cos(x * np.pi * nu * 2)
                for x in np.arange(1, N + 1)
            ]
        ).T
        return basis

    #  DR_basis(y_ft$fq_y, N=10)

    # cbinded X matrix
    def Xmtrix(self, N_delta=15, N_theta=15):
        nu = self.freq
        X_delta = np.concatenate(
            [
                np.column_stack([np.repeat(1, nu.shape[0]), nu]),
                self.DR_basis(N=N_delta),
            ],
            axis=1,
        )
        X_theta = np.concatenate(
            [
                np.column_stack([np.repeat(1, nu.shape[0]), nu]),
                self.DR_basis(N=N_theta),
            ],
            axis=1,
        )
        try:
            if self.Xmat_delta is not None:
                Xmat_delta = tf.convert_to_tensor(X_delta, dtype=tf.float32)
                Xmat_theta = tf.convert_to_tensor(X_theta, dtype=tf.float32)
                return Xmat_delta, Xmat_theta
        except:  # NPE
            self.Xmat_delta = tf.convert_to_tensor(
                X_delta, dtype=tf.float32
            )  # basis matrix
            self.Xmat_theta = tf.convert_to_tensor(X_theta, dtype=tf.float32)
            self.N_delta = N_delta  # N
            self.N_theta = N_theta
            return self.Xmat_delta, self.Xmat_theta

    def set_y_work(self):
        self.y_work = self.y_ft
        return self.y_work

    def dmtrix_k(self, y_k):

        n, p_work = y_k.shape
        Z_k = np.zeros(
            [n, p_work, int(p_work * (p_work - 1) / 2)], dtype=complex
        )

        for j in range(n):
            count = 0
            for i in np.arange(1, p_work):
                Z_k[j, i, count : count + i] = y_k[j, :i]  # .flatten()
                count += i
        return Z_k

    def Zmtrix(self):  # dense Z matrix
        y_work = self.set_y_work()
        c, n, p = y_work.shape
        if p > 1:
            if c == 1:
                y_ls = np.squeeze(y_work, axis=0)
                Z_ = self.dmtrix_k(y_ls)
            else:
                y_ls = np.squeeze(np.split(y_work, c))
                Z_ = np.array([self.dmtrix_k(x) for x in y_ls])
        else:
            Z_ = 0
        self.Zar_re = np.real(
            Z_
        )  # add new variables to self, if Zar not defined in init at the beginning
        self.Zar_im = np.imag(Z_)
        return self.Zar_re, self.Zar_im

    def __repr__(self):
        x = self.ts.shape
        y = self.y_work.shape
        xmat_delta = self.Xmat_delta.shape
        xmat_theta = self.Xmat_theta.shape
        return f"SpecPrep(x(t)={x}, y(f)={y}, Xmat_delta={xmat_delta}, Xmat_theta={xmat_theta})"
