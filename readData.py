import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

class ReadData(object):

    def __init__(self, use_average_beam_axes=False, use_minor_axis=False):

        #Define the cosmology. 
        cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

        #Read the measurements. 
        data = np.loadtxt("data.dat", skiprows=1, usecols=[1,2,3,4,5])
        self.r_ins_ang = data[:,0]
        self.r_outs_ang = data[:,1]
        self.FWHM_Bs_ang = data[:,2]
        self.sigma_obs = data[:,3]
        self.sigma_obs_err = data[:,4]

        #If this option is on, then we use the GEOMETRIC mean of the major and minor axis FWHM of the beam. Otherwise, we use the major axis FWHM, which is what we have in the data.dat file. 
        if use_average_beam_axes:
            self.FWHM_Bs_ang = np.where(self.FWHM_Bs_ang==0.089, (0.089*0.053)**0.5, (0.13*0.09)**0.5)
        #If this option is on, we test using the minor axis instead for the beam convolution. 
        elif use_minor_axis:
            self.FWHM_Bs_ang = np.where(self.FWHM_Bs_ang==0.089, 0.053, 0.09)

        #Convert FWHM to sigma. 
        self.sigma_Bs_ang = self.FWHM_Bs_ang / (2. * (2.*np.log(2.))**0.5 )

        #Now, convert to physical scales, assuming a redshift of 4.6019
        z = 4.6019
        DA = cosmo.angular_diameter_distance(z)
        self.scale_factor = ((1.*u.arcsec).to(u.rad) * DA.to(u.kpc)).value
        self.r_ins = self.r_ins_ang * self.scale_factor
        self.r_outs = self.r_outs_ang * self.scale_factor
        self.sigma_Bs = self.sigma_Bs_ang * self.scale_factor

        return
