import numpy as np
from .modelVelocity import ModelVelocity

#Here we need to overload the ModelVelocity with the new host model. Hence, we also need to process the parameters differently.
class ModelVelocity_2S(ModelVelocity):

    def __init__(self, data):
        super().__init__(data)

    def set_params(self, x):

        self.Mbh = 10.**x[0]

        self.n = x[1::3]
        self.Mbulge = 10.**x[2::3]
        self.re_bulge = x[3::3]
    
        #Get the convenience constant Ie
        g_func_rnorm_max = self.g_func.rnorm_max
        self.Ie_bulge = np.zeros(len(self.Mbulge))
        for i, Mbulge in enumerate(self.Mbulge):
            self.Ie_bulge[i] = Mbulge/self.g_func.g_interp((self.n[i],g_func_rnorm_max))

        return

    def Mhost(self, r):
        Mh = 0
        g_func_rnorm_max = self.g_func.rnorm_max
        for i, Ie in enumerate(self.Ie_bulge):
            rnorm = r/self.re_bulge[i]
            #The following condition is needed because in one of the cases we will assume that the Re could be very small and the normalized radius could exceed 50Re for the outer rings. In those cases, we assume the mass is just converged. 
            rnorm = np.where(rnorm > g_func_rnorm_max, g_func_rnorm_max, rnorm)
            Mh += Ie * self.g_func.g_interp((self.n[i], rnorm))
        return Mh
