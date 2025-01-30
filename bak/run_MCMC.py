import numpy as np
from fFunc import Ffunc
from gFunc import Gfunc
from scipy.optimize import curve_fit
from modelVelocity import ModelVelocity
import emcee

from multiprocessing import get_context
import os
os.environ["OMP_NUM_THREADS"] = "1"

def gauss(x, *p):
    A, sigma = p
    return A*np.exp(-x**2/(2.*sigma**2))

def log_prob(x, g_func, f_funcs, r_ins, r_outs, sigma_Bs, sigma_obs, sigma_obs_err):

    n, log_Mbulge, re_bulge, log_Mbh = x
    Mbulge = 10**log_Mbulge
    Mbh = 10**log_Mbh

    if re_bulge < 0.3 or re_bulge>3. or n>8 or n<0.5 or log_Mbulge<8 or log_Mbulge>12 or log_Mbh>11: 
        return -np.inf

    sigma_model = np.zeros(len(r_ins))
    for k in range(len(r_ins)):
        modelv = ModelVelocity(n, Mbulge, re_bulge, Mbh, sigma_Bs[k], r_ins[k], r_outs[k], g_func, f_func=f_funcs[k])

        vs = np.arange(0, 2000, 100)
        Ivs = np.zeros(vs.shape)
        for i, v in enumerate(vs):
            #Ivs[i] = modelv.Iv(v)
            Ivs[i] = modelv.Iv_romb(v)
        p0 = [0.1, modelv.sigma(0.5*(r_ins[k]+r_outs[k]))]
        coeffs, _ = curve_fit(gauss, vs, Ivs, p0=p0)
        sigma_model[k] = coeffs[1]
        
    chi2 = np.sum(((sigma_obs-sigma_model)/sigma_obs_err)**2)
    return -chi2/2.0


#These are the ring parameters.
sigma_Bs = [0.2908075, 0.2908075   , 0.424775  , 0.424775  , 0.424775  , 0.424775 ]
r_ins =    [0.       , 0.2908075   , 0.61265625, 0.816875  , 1.02109375, 1.2253125]
r_outs =   [0.2908075, 0.4846791884, 0.816875  , 1.02109375, 1.2253125 , 1.429531250]

#These are the measurements in each ring. 
sigma_obs     = np.array([300,245,191,176,202,191])
sigma_obs_err = np.array([3,21,20,18,18,20])

#Initialize the g and f functions. 
f_funcs = [None]*len(r_outs)
for i in range(len(r_ins)):
    f_funcs[i] = Ffunc(sigma_Bs[i], r_ins[i], r_outs[i])
g_func = Gfunc()

#Initial guesses. Order is n, 
nwalkers = 32
ndim = 4
#p0 = np.array([1., 10.5, 1.0, 9.5])
p0 = np.array([4.0, 9., 0.5, 10.])
p0 = p0 * (1+1e-5*np.random.randn(nwalkers, ndim))

#nburn = 500
#nsamp = 2000
#nburn = 200
#nsamp = 500
nburn = 1000
nsamp = 10000
with get_context("fork").Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[g_func, f_funcs, r_ins, r_outs, sigma_Bs, sigma_obs, sigma_obs_err],pool=pool)
    #sampler.run_mcmc(p0, 500, progress=True)

    state = sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, nsamp, progress=True)