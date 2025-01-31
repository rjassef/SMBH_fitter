import numpy as np
import emcee
from multiprocessing import get_context
import matplotlib.pyplot as plt
import corner

class MCMC(object):

    def __init__(self, x_init, x_min, x_max, modelv, log_prob, sigma_obs, sigma_obs_err, ifix=None, nwalkers=32, spread_factor=1e-5):

        self.x_init = x_init
        self.x_min = x_min
        self.x_max = x_max
        self.nwalkers = nwalkers
        self.ndim = len(x_init)
        self.modelv = modelv
        self.log_prob = log_prob
        self.sigma_obs = sigma_obs
        self.sigma_obs_err = sigma_obs_err

        self.labels = ["n", "log Mbulge", "Re bulge", "log Mbh"]

        self.ifix = ifix
        if ifix is None:
            self.ifix = np.zeros(len(x_init), np.bool)

        self.x0 = x_init * (1 + spread_factor*np.random.randn(nwalkers, self.ndim))

        #self.runMCMC()

        return
    
    def runMCMC(self, nproc=8,  nburn=500, nsamp=2000):

        with get_context("fork").Pool(nproc) as pool:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, args=[self.modelv, self.sigma_obs, self.sigma_obs_err, self.x_min, self.x_max, self.x_init, self.ifix],pool=pool)

            state = self.sampler.run_mcmc(self.x0, nburn, progress=True)
            self.sampler.reset()
            self.sampler.run_mcmc(state, nsamp, progress=True)

        self.flat_samples = self.sampler.get_chain(flat=True)
        self.flat_samples_nolog = np.copy(self.flat_samples)
        self.flat_samples_nolog[:,1] = 10.**self.flat_samples[:,1]
        self.flat_samples_nolog[:,3] = 10.**self.flat_samples[:,3]

    def save_flat_samples(self, fname):
        np.savetxt(fname, self.flat_samples)
        return 
    
    def load_flat_samples(self, fname):
        self.flat_samples = np.loadtxt(fname)
        self.flat_samples_nolog = np.copy(self.flat_samples)
        self.flat_samples_nolog[:,1] = 10.**self.flat_samples[:,1]
        self.flat_samples_nolog[:,3] = 10.**self.flat_samples[:,3]   
        return    

    def plotConvergence(self):

        n_plots = np.sum(~self.ifix)
        fig, axes = plt.subplots(n_plots, figsize=(10, np.ceil(7*n_plots/self.ndim)), sharex=True)
        samples = self.sampler.get_chain()
        i = 0
        for ii in range(self.ndim):
            if self.ifix[ii]:
                continue
            ax = axes[i]
            ax.plot(samples[:, :, ii], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.labels[ii])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            i += 1

        axes[-1].set_xlabel("step number")
        plt.show()

        return
    
    def corner_plot(self, keep_log=True):

        labels_use = []
        for i in range(len(self.labels)):
            if ~self.ifix[i]:
                labels_use.append(self.labels[i])
        
        flat_samples_use = self.flat_samples
        if not keep_log:
            flat_samples_use = self.flat_samples_nolog

        fig = corner.corner(
            flat_samples_use[:,~self.ifix], labels=labels_use
        );
        return 
    
    def best_fit(self):
        best_fit = np.median(self.flat_samples[:,~self.ifix], axis=0)
        x_use = np.copy(self.x_init)
        x_use[~self.ifix] = best_fit
        lp_bestfit = self.log_prob(x_use, self.modelv, self.sigma_obs, self.sigma_obs_err, self.x_min, self.x_max, self.x_init, self.ifix)
        return x_use, lp_bestfit
    
    def plot_bestfit(self):

        fig, ax = plt.subplots(1, figsize=(10,8))

        ne, log_Mbulge, re_bulge, log_Mbh = self.best_fit()[0]
        self.modelv.set_params(ne, 10.**log_Mbulge, re_bulge, 10.**log_Mbh)

        r_mean = 0.5*(self.modelv.r_ins+self.modelv.r_outs)
        ax.errorbar(r_mean, self.sigma_obs, yerr=self.sigma_obs_err, fmt='bo', label='Observed')
        ax.plot(r_mean, self.modelv.model_sigmas(), 'ro', label='Best-fit Model')
        ax.legend()
        ax.set_xlabel("Distance from Center (kpc)")
        ax.set_ylabel("Velocity Dispersion (km/s)")
        plt.show()

        return

