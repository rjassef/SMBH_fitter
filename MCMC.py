import numpy as np
import emcee
from multiprocessing import get_context
import matplotlib.pyplot as plt
import corner

class MCMC(object):

    def __init__(self, x_init, x_min, x_max, modelv, log_prob, sigma_obs, sigma_obs_err, ifix=None, nwalkers=32, spread_factor=1e-5, flatchain_folder="flatchains"):

        self.x_init = x_init
        self.x_min = x_min
        self.x_max = x_max
        self.nwalkers = nwalkers
        self.ndim = len(x_init[~ifix])
        self.modelv = modelv
        self.log_prob = log_prob
        self.sigma_obs = sigma_obs
        self.sigma_obs_err = sigma_obs_err
        self.flatchain_folder = flatchain_folder

        self.labels = [
            "Sersic Index n", 
            r"$\log M_{\rm Host}/M_{\odot}$", 
            r"$R_{\rm eff}/{\rm kpc}$", 
            r"$\log M_{\rm BH}/M_{\odot}$"
        ]

        self.ifix = ifix
        if ifix is None:
            self.ifix = np.zeros(len(x_init), np.bool)

        self.x0 = x_init[~ifix] * (1 + spread_factor*np.random.randn(nwalkers, self.ndim))

        #self.runMCMC()

        return
    
    def runMCMC(self, nproc=8,  nburn=1000, nsamp=5000):

        with get_context("fork").Pool(nproc) as pool:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, args=[self.modelv, self.sigma_obs, self.sigma_obs_err, self.x_min, self.x_max, self.x_init, self.ifix],pool=pool)

            state = self.sampler.run_mcmc(self.x0, nburn, progress=True)
            self.sampler.reset()
            self.sampler.run_mcmc(state, nsamp, progress=True)

        self.flat_samples = self.sampler.get_chain(flat=True)

    def save_flat_samples(self, fname):
        np.savetxt(self.flatchain_folder+"/"+fname, self.flat_samples)
        return 
    
    def load_flat_samples(self, fname):
        self.flat_samples = np.loadtxt(self.flatchain_folder+"/"+fname) 
        return    

    def plotConvergence(self):

        fig, axes = plt.subplots(self.ndim, figsize=(10, np.ceil(7*self.ndim/len(self.x_init))), sharex=True)
        samples = self.sampler.get_chain()
        i = 0
        for ii in range(len(self.x_init)):
            if self.ifix[ii]:
                continue
            if self.ndim>1:
                ax = axes[i]
            else:
                ax = axes
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.labels[ii])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            i += 1

        if self.ndim>1:
            axes[-1].set_xlabel("step number")
        else:
            axes.set_xlabel("step number")
        plt.show()

        return
    
    def corner_plot(self):

        labels_use = []
        for i in range(len(self.labels)):
            if ~self.ifix[i]:
                labels_use.append(self.labels[i])
        
        flat_samples_use = self.flat_samples

        fig = corner.corner(
            flat_samples_use, labels=labels_use, label_kwargs={"fontsize": 12}, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}
        );
        return 
    
    def best_fit(self):

        #Get the percentiles for the parameters that vary. 
        pps = np.percentile(self.flat_samples, [16, 50, 84], axis=0)

        #Best fit is the median, and errors are 1sigma.
        best_fit = np.copy(self.x_init)
        best_fit[~self.ifix] = pps[1]

        err_low = np.zeros(best_fit.shape)
        err_hig = np.zeros(best_fit.shape)
        err_low[~self.ifix] = best_fit[~self.ifix]-pps[0]
        err_hig[~self.ifix] = pps[2]-best_fit[~self.ifix]

        lp_bestfit = self.log_prob(pps[1], self.modelv, self.sigma_obs, self.sigma_obs_err, self.x_min, self.x_max, self.x_init, self.ifix)

        return best_fit, err_low, err_hig, lp_bestfit

        # best_fit = np.median(self.flat_samples[:,~self.ifix], axis=0)
        # x_use = np.copy(self.x_init)
        # x_use[~self.ifix] = best_fit
        # lp_bestfit = self.log_prob(x_use, self.modelv, self.sigma_obs, self.sigma_obs_err, self.x_min, self.x_max, self.x_init, self.ifix)
        # return x_use, lp_bestfit
    
    def plot_bestfit(self):

        fig, ax = plt.subplots(1, figsize=(5,4))

        ne, log_Mbulge, re_bulge, log_Mbh = self.best_fit()[0]
        self.modelv.set_params(ne, 10.**log_Mbulge, re_bulge, 10.**log_Mbh)

        chi2 = self.best_fit()[-1] * -2
        chi2_nu = chi2/(len(self.sigma_obs)-(4-np.sum(self.ifix)))

        ax.set_title(r"Goodness of fit:   $\chi^2$ = {:.2f},   $\chi^2/{{\rm dof}}$ = {:.2f}".format(chi2, chi2_nu), transform=ax.transAxes)

        r_mean = 0.5*(self.modelv.data.r_ins+self.modelv.data.r_outs)
        ax.errorbar(r_mean, self.sigma_obs, yerr=self.sigma_obs_err, marker='o', color='xkcd:gray', linestyle='none', label='Observed')
        ax.plot(r_mean, self.modelv.model_sigmas(), 'mo', label='Best-fit Model', markerfacecolor="none")
        ax.legend()
        ax.set_xlabel("Distance from Center (kpc)")
        ax.set_ylabel("Velocity Dispersion (km/s)")
        plt.show()

        return

