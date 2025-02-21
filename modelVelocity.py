import numpy as np
from fFunc import Ffunc
from gFunc import Gfunc
from astropy.constants import G, M_sun
from scipy.integrate import quad, romb
import astropy.units as u
from scipy.optimize import curve_fit

#All distances are in kpc, all masses in Msun units. 
class ModelVelocity(object):

    def __init__(self, data):

        #Save the input parameters.
        self.data = data
        
        #Start the the f and g function objects.
        self.f_funcs = [None]*len(data.r_ins)
        for i in range(len(data.r_ins)):
            self.f_funcs[i] = Ffunc(data.sigma_Bs[i], data.r_ins[i], data.r_outs[i])
        self.g_func = Gfunc()

        #This is the re from the light profile. It is a Gaussian with a FWHM of 2.35kpc from Tanio's paper. 
        self.b_gaussian = self.g_func.b_n(0.5)
        self.n_light = 0.5
        self.FWHM_light = 2.35
        self.re_light = self.FWHM_light/2. * (self.b_gaussian/np.log(2.))**0.5

        #Constant with the right units for getting sigma. 
        self.K = (((2./3.) * G * M_sun/(1.*u.kpc))**0.5).to(u.km/u.s).value

        return
    
    def set_params(self, n, Mbulge, re_bulge, Mbh):

        self.n = n
        self.Mbulge = Mbulge
        self.re_bulge = re_bulge
        self.Mbh = Mbh
    
        #Get the convenience constant Ie
        g_func_rnorm_max = self.g_func.rnorm_max
        self.Ie_bulge = Mbulge/self.g_func.g_interp((n,g_func_rnorm_max))

        return

    def sigma(self, r):

        rnorm = r/self.re_bulge
        host_mass = self.Ie_bulge * self.g_func.g_interp((self.n, rnorm))
        return self.K * ((self.Mbh+host_mass)/(r+1e-32))**0.5

    def Iv(self, v):
    
        func = lambda r: np.exp(-self.b_gaussian*(r/self.re_light)**2-0.5*(v/self.sigma(r))**2) * self.f_func.f_interp(r)*r

        #return quad(func, 0., 5., epsrel=1e-3)[0]
        return quad(func, 0., 10., epsrel=1e-3)[0]
    
    def Iv_romb(self, v, k):
    
        func = lambda r: np.exp(-self.b_gaussian*(r/self.re_light)**2-0.5*(v/self.sigma(r))**2) * self.f_funcs[k].f_interp(r)*r

        #nsamps = 2**6 + 1
        nsamps = 2**7 + 1
        dr = 10./(nsamps-1)
        rs = np.arange(0., 10.+0.1*dr, dr)
        return romb(func(rs), dx=dr)
    
    def Iv_nosmearing(self, v):
    
        func = lambda r: np.exp(-self.b_gaussian*(r/self.re_light)**2-0.5*(v/self.sigma(r))**2) * r

        return quad(func, self.r_in, self.r_out)[0]
    
    def _gauss(self, x, *p):
        A, sigma = p
        return A*np.exp(-x**2/(2.*sigma**2))

    def model_sigma(self, k, vmin=0., vmax=2000., dv=100.):

        vs = np.arange(vmin, vmax+0.1*dv, dv)
        Ivs = np.zeros(vs.shape)
        for i, v in enumerate(vs):
            Ivs[i] = self.Iv_romb(v, k)

        r_mean = 0.5*(self.data.r_ins[k]+self.data.r_outs[k])
        initial_guess = [0.1, self.sigma(r_mean)]
        coeffs, _ = curve_fit(self._gauss, vs, Ivs, p0=initial_guess, check_finite=False)
        return coeffs[1]
    
    def model_sigmas(self, vmin=0, vmax=2000., dv=100.):

        sigma_model = np.zeros(len(self.data.r_ins))
        for k in range(len(self.data.r_ins)):
            sigma_model[k] = self.model_sigma(k, vmin, vmax, dv)

        return sigma_model
