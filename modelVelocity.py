import numpy as np
from fFunc import Ffunc
from gFunc import Gfunc
from astropy.constants import G, M_sun
from scipy.integrate import quad, romb
import astropy.units as u

#All distances are in kpc, all masses in Msun units. 
class ModelVelocity(object):

    def __init__(self, n, Mbulge, re_bulge, Mbh, sigma_B, r_in, r_out, g_func=None, f_func=None):

        #Save the input parameters.
        self.n = n
        self.Mbulge = Mbulge
        self.re_bulge = re_bulge
        self.Mbh = Mbh
        self.sigma_B = sigma_B
        self.r_in = r_in
        self.r_out = r_out
        
        #This is the re from the light profile. It is a Gaussian with a FWHM of 2.35kpc from Tanio's paper. 
        self.b_gaussian = g_func.b_n(0.5)
        self.n_light = 0.5
        self.FWHM_light = 2.35
        self.re_light = self.FWHM_light/2. * (self.b_gaussian/np.log(2.))**0.5

        #Start the the f and g function objects.
        if f_func is None:
            self.f_func = Ffunc(sigma_B, r_in, r_out)
        else:
            self.f_func = f_func

        if g_func is None:
            self.g_func = Gfunc()
        else:
            self.g_func = g_func

        #Get the convenience constant Ie
        g_func_rnorm_max = g_func.rnorm_max
        self.Ie_bulge = Mbulge/g_func.g_interp((n,g_func_rnorm_max))

        #Constant with the right units for getting sigma. 
        self.K = (((2./3.) * G * M_sun/(1.*u.kpc))**0.5).to(u.km/u.s).value

        return
    
    def sigma(self, r):

        rnorm = r/self.re_bulge
        host_mass = self.Ie_bulge * self.g_func.g_interp((self.n, rnorm))
        # try:
        #     host_mass = np.zeros(len(r))
        #     for j in range(len(r)):
        #         host_mass[j] = self.Ie_bulge * self.g_func.g_exact(self.n, rnorm[j])
        # except TypeError:
        #     host_mass = self.Ie_bulge * self.g_func.g_exact(self.n, rnorm)
        #print(self.K, self.Mbh, host_mass)
        return self.K * ((self.Mbh+host_mass)/(r+1e-32))**0.5

    def Iv(self, v):
    
        func = lambda r: np.exp(-self.b_gaussian*(r/self.re_light)**2-0.5*(v/self.sigma(r))**2) * self.f_func.f_interp(r)*r

        #return quad(func, 0., 5., epsrel=1e-3)[0]
        return quad(func, 0., 10., epsrel=1e-3)[0]
    
    def Iv_romb(self, v):
    
        func = lambda r: np.exp(-self.b_gaussian*(r/self.re_light)**2-0.5*(v/self.sigma(r))**2) * self.f_func.f_interp(r)*r

        #nsamps = 2**6 + 1
        nsamps = 2**7 + 1
        dr = 10./(nsamps-1)
        rs = np.arange(0., 10.+0.1*dr, dr)
        return romb(func(rs), dx=dr)
    
    def Iv_nosmearing(self, v):
    
        func = lambda r: np.exp(-self.b_gaussian*(r/self.re_light)**2-0.5*(v/self.sigma(r))**2) * r

        return quad(func, self.r_in, self.r_out)[0]
