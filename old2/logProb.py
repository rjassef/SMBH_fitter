import numpy as np 
from modelVelocity import ModelVelocity

def log_prob(x, modelv, sigma_obs, sigma_obs_err, x_min, x_max, x_init, ifix):

    x_use = np.copy(x_init)
    x_use[~ifix] = x

    if np.sum(x<x_min[~ifix])>0 or np.sum(x>x_max[~ifix])>0:
       return -np.inf

    n, log_Mbulge, re_bulge, log_Mbh = x_use
    Mbulge = 10**log_Mbulge
    Mbh = 10**log_Mbh
    modelv.set_params(n, Mbulge, re_bulge, Mbh)

    chi2 = np.sum(((sigma_obs-modelv.model_sigmas())/(sigma_obs_err))**2)
    return -chi2/2.


