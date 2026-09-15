# SMBH_fitter

This code models the observed velocity dispersion radial profile of W2245-0526 as presented by Liao et al. (submitted). All fits carried out with it are presented in [this notebook](Fit_Observed_Dispersion_Profile.ipynb) and discussed in Liao et al. (submitted).

The model velocity dispersion as a function of the distance to the center of the object, $r$, is calculated by combining the gravitational effects of a central super-massive black hole and a host galaxy. Additionally, we consider the smoothing effect of ALMA beam on the model, as well as the size of the integration regions (concentric elliptical rings), before comparing the data with the model. We refer the reader to section 4 of Liao et al. (submitted) for further details. 

Specifically, what we aim to do is to estimate from a set of model parameters the observed velocity profile of the [CII] emission line, $I_M(v)$. As discussed in section 4 of Liao et al. (submitted), we can write

$I_{\rm M}(v)\propto \int_{r_{\rm in}}^{r_{\rm out}} I_{\rm obs}(r_0,v)\ r_0\ dr_0$,

where $r_{\rm in}$ and $r_{\rm out}$ are, respectively, the inner and outer radii of the ring in question. As the rings are slightly elliptical, we consider the geometric average between the respective major and minor axes as the ring radius. $I_{\rm obs}(r_0,v)$ is the line intensity profile that would be observed at a distance $r_0$ from the center, and is given by 


$I_{\rm obs}(r_0,v)\propto \int_{0}^{\infty} I_{\rm int}(r,v)\ \exp{\left\\{-\frac{1}{2}\left(\frac{r_0^2+r^2}{\sigma_{\rm Beam}^2}\right)\right\\}}\ \mathcal{I}_0\left(\frac{r_0 r}{\sigma_B^2}\right)\ r\ dr$,

where $\sigma_{\rm Beam}$ is the standard deviation of the beam size, transformed from the geometric mean of the FWHM axes of the respective beams, and $`\mathcal{I}_0`$ is the zero-th order modified Bessel function of the first kind. In this equation $I_{\rm int}(r,v)$ is the intrinsic line intensity velocity profile that would be observed in the absence of beam smoothing, and is given by 

$I_{\rm int}(r,v) \propto I_R(r)\ \exp{-\frac{1}{2}\left(\frac{v}{\sigma(r)}\right)^2}$,

where $I_R(r)$ is the velocity-integrated intensity profile of [CII], taken to be a Gaussian with effective radius of 1.2~kpc from [Diaz-Santos et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...654A..37D/abstract), and $\sigma(r)$ is the velocity dispersion due to the gravity of the SMBH and the remainder of the components of the host (mostly gas and stars). 

We note that, for convenience, we can write 

$I_{\rm M}(v)\propto \int_{0}^{\infty} I_{\rm int}(r,v)\ F(r, r_{\rm in}, r_{\rm out})\ r\ dr$, 

where 

$F(r, r_{\rm in}, r_{\rm out}) = \int_{r_{\rm in}}^{r_{\rm out}} \exp{\left\\{-\frac{1}{2}\left(\frac{r_0^2+r^2}{\sigma_{\rm Beam}^2}\right)\right\\}}\ \mathcal{I}_0\left(\frac{r_0 r}{\sigma_B^2}\right)\ r_0\ dr_0$

To speed up the calculation, we pre-compute $F$ in a grid of $r$ for each combination of $r_{\rm in}$ and $r_{\rm out}$, and then interpolate in $r$. This is implemented by the [Ffunc](fFunc.py) object, which is initialized by the [ModelVelocity](modelVelocity.py) object. 

We assume that 

$\sigma(r) = \sqrt{\frac{2}{3}~\frac{G_{\rm N} [M_{\rm BH} + M_{\rm Host}(<r)]}{r}}$, 

where $M_{\rm BH}$ is the SMBH mass, $M_{\rm Host}(<r)$ is the combined mass of all other gravitational components in the host galaxy interior to the radius $r$, and $G_{\rm N}$ is the gravitational constant.

We assume that the surface density distribution $M_{\rm Host}$ follows a Sérsic Profile with Sérsic index $n$ and effective radius $R_{\rm eff}$. Hence, we can conveniently write $M_{\rm Host}(<r)$ as

$M_{\rm Host}(<r) = M_{\rm Host}^{\rm Total}\ \frac{G(n, r/R_{\rm eff})}{G(n, r/R_{\infty})}$

where $M_{\rm Host}^{\rm Total}$ is the total mass of the host component we consider, and 

$G(n, x) = \int_0^x e^{-b_n x^{1/n}} x dx$, 

in which $b_n$ is the traditional exponential coefficient of the Sérsic Profile. To speed up the calculations, we pre-compute the G function in a large grid of $n$ and $r/R_{\rm eff}$ and then interpolate between them. This is implemented in the [Gfunc](gFunc.py) object, which is initialized by the [ModelVelocity](modelVelocity.py) object. 

## Offset-host model

The JWST/MIRI F560W observations of W2246-0526, imply a potential offset ($r_{\rm offset}$ = 1kpc) between the stellar center and the SMBH position, See these detailed discussions in Sections XXX and XXX in Liao et al. This model is designed to fit the outermost dispersion point to obtain the offset-host total mass for the case that the SMBH position is not same as the host center. 

Since the gravity potential would be highly asymmetric between the SMBH and the offset-host center, it would be very complex to build a model, including SMBH and host components, to do the full fitting for the whole dispersion profile. Instead, we simply assumed the outermost dispersion data point is dominated by the gravity from offset-host mass. When doing the fitting, we assumed its host model with following a Sérsic Profile with $n$ = 1.64 and $R_{e}^{\rm Host}$ = 1.5 kpc to match the morphology of the JWST F560W imaging. Its total mass $M_{\rm Host}^{\rm Total}$ is left as a free parameter.

Compared to the default dynamical model described above, the only different calculation of the offset-host model is the intrinsic dispersion contributed by the enclosed offset-host mass seen by $r$ (the radius from the SMBH position). In this case, the intrinsic velocity dispersion at a given distance from the center of offset-host mass distribution, $r_{\rm g}$, would be expressed as:

$\sigma_{\rm Host}(r_{\rm g}) = \sqrt{\frac{2}{3}\frac{G_{\rm N}\big[M_{\rm Host}(<r_{\rm g})\big]}{r_{\rm g}}}$,

Above equation can be further written at a given distance from the center of SMBH, 𝑟, as

$\sigma_{\rm Host}(r, \theta_1) = \sqrt{\frac{2}{3}\frac{G_{\rm N}\big[M_{\rm Host}(<r_{\rm g}[r, \theta_1])\big]}{r_{\rm g}[r, \theta_1]}}$,

Here, $r_{\rm g} = \sqrt{r^2 + r_{\rm offset}^2 - 2rr_{\rm offset}\cos\theta_1}$,

So then we can replace the $\sigma_{r}$ with $\sigma_{\rm Host}(r, \theta_1)$ in the default model described above, and consider the smoothing effect of ALMA beam on the model, as well as the size of the integration regions (concentric elliptical rings), to calculate the model dispersion. 

To speed up the calculations, we pre-compute the term with carrying out
the angular integration in $\theta_1$:

$\exp\left[-\frac{v^2r_{\rm g} [r, \theta_1]}{3G_{\rm N} M_{\rm Host}(<r_{\rm g} [r, \theta_1])}\right]$,

here, $M_{\rm Host}(<r) = M_{\rm Host}^{\rm Total}\ \frac{G(n, r/R_{\rm eff})}{G(n, r/R_{\infty})}$,

in a large grid of $v$, $r$, $\log M_{\rm Host}^{\rm Total}$, $R_{\rm eff}$, and $n$, and then interpolate between them. This is implemented in `ThetaIntegral`, which is initialized by the `ModelVelocity_offset_host` object.


