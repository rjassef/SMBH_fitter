import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import i0

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

        x_in = self.r_in/self.sigma_B
        x_out = self.r_out/self.sigma_B
        y = r/self.sigma_B

        func = lambda x: np.exp(-0.5*(y**2+x**2)) * i0(y*x) * x

        return self.sigma_B**2 * quad(func, x_in, x_out)[0]
    

