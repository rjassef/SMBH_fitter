# theta_integral.py
import numpy as np
from scipy.integrate import romb
from scipy.interpolate import RegularGridInterpolator
from .gFunc import Gfunc


class ThetaIntegral(object):

    def __init__(self):
        self.r_offset = 0.98

        # unit constant, pc (km/s)^2 Msun^-1
        self.G = 0.004301
        self.A0 = (2.0 / 3.0) * self.G / 1000.0

        self.vel_points = np.arange(0., 1500, 100)
        self.r_points = np.linspace(1.e-3, 4.99, 15)
        self.M_points = np.arange(9., 12, 0.2)
        self.re_points = np.linspace(0.5, 8, 15)
        self.n_points = np.linspace(0.5, 8.5, 15)

        # init Gfunc for enclosed mass calculation (Sérsic)
        self.g_func = Gfunc()

        self._build_interpolator()
        return

    def _calc_Menc(self, Mlin, re, n, r):

        rnorm = r / re
        g_max = self.g_func.g_interp((n, self.g_func.rnorm_max))
        Ie = Mlin / g_max
        return Ie * self.g_func.g_interp((n, rnorm))

    def theta_integrand(self, vel, r, M_mass, re, n):
        """
        M_mass: log10(Mbulge)
        """
        A = vel**2 / (2.0 * self.A0)

        def integrand_func(x):
            r_val = np.sqrt(r**2 + self.r_offset**2 - 2 * r * self.r_offset * np.cos(x))
            Menc = self._calc_Menc(10**M_mass, re, n, r_val)
            return np.exp(-A * r_val / Menc)

        theta_points = np.linspace(0, 2 * np.pi, 2**7 + 1)
        y_values = integrand_func(theta_points)
        dx_theta = theta_points[1] - theta_points[0]
        return romb(y_values, dx=dx_theta)

    def _build_interpolator(self):
        """Construct 5D lookup table and RegularGridInterpolator"""
        results = np.zeros((
            len(self.vel_points),
            len(self.r_points),
            len(self.M_points),
            len(self.re_points),
            len(self.n_points)
        ))
        total_iter = np.prod(results.shape)
        current = 0

        for i, vel_val in enumerate(self.vel_points):
            for j, r_val in enumerate(self.r_points):
                for k, M_val in enumerate(self.M_points):
                    for l, re_val in enumerate(self.re_points):
                        for m, n_val in enumerate(self.n_points):
                            current += 1
                            try:
                                res = self.theta_integrand(vel_val, r_val, M_val, re_val, n_val)
                                results[i, j, k, l, m] = res
                            except Exception as e:
                                print(f"Calc failed: vel={vel_val},r={r_val},logM={M_val},re={re_val},n={n_val}, err={e}")
                                results[i, j, k, l, m] = 0.0

        self.theta_interp = RegularGridInterpolator(
            (self.vel_points, self.r_points, self.M_points, self.re_points, self.n_points),
            results,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        return

    def theta_interp_eval(self, vel, r, M_mass, re, n):
        pt = np.array([[vel, r, M_mass, re, n]])
        val = self.theta_interp(pt)[0]
        return val
