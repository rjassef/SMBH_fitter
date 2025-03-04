import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator

class Gfunc(object):

    def __init__(self):

        #The integral we have to do is in units of r/r_e. We'll scan from 0. to 5.
        #In n we'll go from n=0.5 to n=8. in units of 0.1.
        ns = np.arange(0.5, 10.01, 0.1)
        rnorms = np.arange(0., 50.01, 0.05)
        self.g_values = np.zeros((len(ns), len(rnorms)))
        for i, n in enumerate(ns):
            for j, rnorm in enumerate(rnorms):
                self.g_values[i,j] = self.g_exact(n, rnorm)

        
        self.g_interp = RegularGridInterpolator((ns,rnorms), self.g_values)
        self.rnorm_max = np.max(rnorms)

        return

    def b_n(self, n):
        return 2.*n - 1./3. + 4./(405.*n) + 46./(25515.*n**2) + 131./(1148175.*n**3) - 2194697./(30690717750.*n**4)

    def g_exact(self, n, rnorm):
        #Calculate b_n
        b = self.b_n(n)

        func = lambda x : np.exp(-b*x**(1./n)) * x

        return quad(func, 0., rnorm)[0]