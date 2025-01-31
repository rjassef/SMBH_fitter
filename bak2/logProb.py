import numpy as np 
from modelVelocity import ModelVelocity

def log_prob(x, g_func, f_funcs, r_ins, r_outs, sigma_Bs, sigma_obs, sigma_obs_err, x_min, x_max, x_init, ifix):

    x_use = np.array(x)
    x_use[ifix] = x_init[ifix]

    if np.sum(x_use[~ifix]<x_min[~ifix])>0 or np.sum(x_use[~ifix]>x_max[~ifix])>0:
       return -np.inf

    n, log_Mbulge, re_bulge, log_Mbh = x_use
    Mbulge = 10**log_Mbulge
    Mbh = 10**log_Mbh

    sigma_model = np.zeros(len(r_ins))
    for k in range(len(r_ins)):
        modelv = ModelVelocity(n, Mbulge, re_bulge, Mbh, sigma_Bs[k], r_ins[k], r_outs[k], g_func, f_func=f_funcs[k])
        sigma_model[k] = modelv.model_sigma()
    
    chi2 = np.sum(((sigma_obs-sigma_model)/(sigma_obs_err))**2)
    return -chi2/2.

def log_probv2(x, modelv, sigma_obs, sigma_obs_err, x_min, x_max, x_init, ifix):

    x_use = np.array(x)
    x_use[ifix] = x_init[ifix]

    if np.sum(x_use[~ifix]<x_min[~ifix])>0 or np.sum(x_use[~ifix]>x_max[~ifix])>0:
       return -np.inf

    n, log_Mbulge, re_bulge, log_Mbh = x_use
    Mbulge = 10**log_Mbulge
    Mbh = 10**log_Mbh
    modelv.set_params(n, Mbulge, re_bulge, Mbh)

    chi2 = np.sum(((sigma_obs-modelv.model_sigmas())/(sigma_obs_err))**2)
    return -chi2/2.


