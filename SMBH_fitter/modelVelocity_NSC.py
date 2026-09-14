import numpy as np
from .modelVelocity import ModelVelocity

#Here we need to overload the ModelVelocity with the new host model. Hence, we also need to process the parameters differently.
class ModelVelocity_NSC(ModelVelocity):

    def __init__(self, data, RNSC_max=None):

        if RNSC_max is None:
            RNSC_max = 100. #0.01 #10pc in units of kpc.
        self.RNSC_max = RNSC_max

        super().__init__(data)

    def set_params(self, x):

        self.Mbh = 10.**x[0]

        self.n = x[1]
        self.Mbulge = 10.**x[2]
        self.re_bulge = x[3]

        self.Rc = x[4]
        self.Mnsc_tot = 10.**x[5]
    
        #Get the convenience constant Ie
        g_func_rnorm_max = self.g_func.rnorm_max
        self.Ie_bulge = self.Mbulge/self.g_func.g_interp((self.n,g_func_rnorm_max))

        return

    #The host mass will be the sum the of the actual host plus that of the Nuclear Star Cluster (NSC). For the NSC we assume a mass density that drops like r^-2 until a characteristic radius R_c, and then falls like r^-3 out to Rmax. The NSC model is taken from equation (2) of Juodžbalis et al. (2025). See the supplementary sections of the paper for details. 
    def Mhost_extended(self, r):
        rnorm = r/self.re_bulge
        return self.Ie_bulge*self.g_func.g_interp((self.n, rnorm))

    def Mnsc(self, r):
        A = self.Mnsc_tot/(4*np.pi*self.Rc*(1+np.log(self.RNSC_max/self.Rc)))

        return np.where(r<self.Rc, 
                        4*np.pi*A*r,
                        4*np.pi*A*self.Rc*(1+np.log((r+1e-32)/self.Rc))
                        )

    def Mhost(self, r):
        return self.Mhost_extended(r) + self.Mnsc(r)
