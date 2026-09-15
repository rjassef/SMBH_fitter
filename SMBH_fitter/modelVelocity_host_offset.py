import numpy as np
from scipy.integrate import romb

from .modelVelocity import ModelVelocity
from .offset_model_theta_integral import ThetaIntegral


class ModelVelocity_host_offset(ModelVelocity):

    def __init__(self, data):

        super().__init__(data)

        self.theta_obj = ThetaIntegral()

        self.nsamps = 2**6 + 1
        self.dr = 10./(self.nsamps-1)
        self.rs = np.arange(0., 10.+0.1*self.dr, self.dr)
        self.gauss_light = np.exp(-self.b_gaussian*(self.rs/self.re_light)**2)

        return

    def set_params(self, x):
        # x = [n, log_Mbulge, re_bulge] no Mbh
        self.n = x[0]
        self.log_Mbulge = x[1]
        self.re_bulge = x[2]
        self.Mbh = 0

        g_func_rnorm_max = self.g_func.rnorm_max
        self.Ie_bulge = (10.**(self.log_Mbulge)) / self.g_func.g_interp((self.n, g_func_rnorm_max))

        return

    def Iv_romb(self, v, k):

        theta_term = self.theta_obj.theta_interp((v*np.ones(len(self.rs)), self.rs, self.log_Mbulge*np.ones(len(self.rs)), self.re_bulge*np.ones(len(self.rs)), self.n*np.ones(len(self.rs))))
        term = self.gauss_light * theta_term * self.f_funcs[k].f_interp(self.rs) * self.rs

        return romb(term, dx=self.dr)
