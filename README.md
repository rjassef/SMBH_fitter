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

## Offset host model

This model is for the potential offset (~ 1kpc) between the stellar center and the SMBH position, indicated by the JWST/MIRI F560W observations. See these detailed discussions in Sections XXX and XXX in Liao et al.

In this case, since the gravity potential would be highly asymmetric between the SMBH and offset stellar center, it would be very complex to build a model to do the full fitting for the whole dispersion profile. Instead, we simply assumed the outermost [C II] dispersion data point is doFor a projected radius $r$ measured from the SMBH, the line-of-sight integral requires an additional azimuthal average over the host–SMBH geometry. 

At azimuth angle $\theta_1$, the physical radius measured from the host's mass center is

$r_{\rm g} = \sqrt{r^2 + r_{\rm offset}^2 - 2\,r\,r_{\rm offset}\cos\theta_1}$The intrinsic velocity dispersion contributed by the host galaxy at this shifted radius reads

$\sigma_{\rm Host}(r,\theta_1) = \sqrt{\frac{2}{3}\frac{G_{\rm N}\big[M_{\rm Host}(<r_{\rm g})\big]}{r_{\rm g}}}$The intrinsic line profile at projected radius $r$ is no longer a simple Gaussian in $v$; instead, the velocity-dependent exponential term must be azimuthally averaged over $\theta_1$,$I_{\rm int}(r,v) \propto I_R(r)\,\theta_{\rm int}(v, r, M_{\rm Host}^{\rm Total}, R_{\rm eff}, n)$with$\theta_{\rm int} = \frac{1}{2\pi}\int_0^{2\pi}\exp\left[-\frac{v^2\,r_{\rm g}}{3\,G_{\rm N}\big(M_{\rm BH}+M_{\rm Host}(<r_{\rm g})\big)}\right] d\theta_1$The Sérsic enclosed mass $M_{\rm Host}(<r_{\rm g})$ is evaluated at the offset radius $r_{\rm g}$ and uses the same $G(n, x)$ interpolation table as the non-offset model. The host stellar component follows a Sérsic profile with index $n_{\rm Host}$ and effective radius $R_e^{\rm Host}$, calibrated to match the JWST F560W morphology. The total host mass $M_{\rm Host}^{\rm Total}$ is treated as a free fitting parameter.The beam-convolution and ring-integration kernel $F(r, r_{\rm in}, r_{\rm out})$ (implemented in `Ffunc`) is unchanged; the offset modification enters only through $I_{\rm int}(r,v)$. We modify equations 4–7 from the base model to perform the angular integration over $\theta_1$ and compute the [CII] line velocity profile for each elliptical ring.To keep the evaluation fast, $\theta_{\rm int}$ is precomputed once on a regular 5-D grid $(v,\ r,\ \log M_{\rm Host}^{\rm Total},\ R_{\rm eff},\ n)$ and stored in a `RegularGridInterpolator`. During fitting, the lookup is vectorized over the radial integration grid, so no Python-level loop over $\theta_1$ or $r$ is required. This is implemented in `ThetaIntegral` (`Offset_model_theta_integral.py`) and wrapped by the `ModelVelocity_offset_host` class.
0 commit commentsComments0 (0)Lock conversationComment
