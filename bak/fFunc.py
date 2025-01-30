import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import erf

class Ffunc(object):

    def __init__(self, sigma_B, r_in, r_out):

        #Save the input values. 
        self.sigma_B = sigma_B
        self.r_in = r_in
        self.r_out = r_out

        #For a range of value of r, get the integrals and build the interpolation table. 
        self.rs = np.arange(0., 10.01, 0.1)
        self.f_precalc = np.zeros(self.rs.shape)
        for i, r in enumerate(self.rs):
            self.f_precalc[i] = self.f_exact(r)

        self.f_interp = CubicSpline(self.rs,self.f_precalc)#, extrapolate=True)

        return

    def f_exact(self, r):

        x_in = (self.r_in-r)/(2**0.5 * self.sigma_B)
        x_out = (self.r_out-r)/(2**0.5*self.sigma_B)

        f1 = self.sigma_B**2 * (np.exp(-x_in**2) - np.exp(-x_out**2))
        f2 = self.sigma_B * r * (np.pi/2.)**0.5 * (erf(x_out)-erf(x_in))

        return f1+f2

