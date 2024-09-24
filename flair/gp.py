import celerite2
from celerite2 import terms
from scipy.optimize import minimize as sp_minimize
import numpy as np
from astropy.timeseries import LombScargle

from nuance.kernels import rotation
from nuance.utils import minimize
from nuance.core import gp_model



def fit_GP(lc, flare_mask):
    """Fit a Gaussian Process to a lightcurve, ignoring flares.

    Parameters
    ----------
    lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve to fit
    flare_mask : :class:`numpy.ndarray`
        Boolean mask of flares in the lightcurve

    Returns
    -------
    opt_gp : :class:`celerite2.GaussianProcess`
        Optimized Gaussian Process
    """
    # mask the flares out of the lightcurve
    x = lc.time.value[~flare_mask]
    y = lc.flux.value[~flare_mask]
    y_err = lc.flux_err.value[~flare_mask]

    # update the GP parameters
    def set_params(params, gp):
        gp.mean = params[0]
        theta = np.exp(params[1:])
        gp.kernel = celerite2.terms.SHOTerm(
            sigma=theta[0], rho=theta[1], tau=theta[2]
        ) + celerite2.terms.SHOTerm(sigma=theta[3], rho=theta[4], Q=0.25)
        gp.compute(x, diag=y_err**2 + theta[5], quiet=True)
        return gp

    # calculate the negative log likelihood
    def neg_log_like(params, gp):
        gp = set_params(params, gp)
        return -gp.log_likelihood(y)

    # initialize the GP and optimize
    # Quasi-periodic term
    term1 = terms.SHOTerm(sigma=1.0, rho=1.0, tau=10.0)

    # Non-periodic component
    term2 = terms.SHOTerm(sigma=1.0, rho=5.0, Q=0.25)
    kernel = term1 + term2

    # Setup the GP
    gp = celerite2.GaussianProcess(kernel, mean=0.0)
    gp.compute(x, yerr=y_err)
    
    initial_params = [0, 0, 0, np.log(10.0), 0, np.log(5.0), np.log(0.01)]
    set_params(initial_params, gp)

    # Define bounds: (lower_bound, upper_bound) or (None, None) for no bounds
    bounds = [
        (None, None),  # No bounds for the first parameter #parameter[0] 
        (-10, 0),  # Lower bound for the second parameter #theta[0]
        (-10, 0),  # Lower bound for the third parameter #theta[1]
        (-10, 10),  # No bounds for the fourth parameter #theta[2]
        (-10, 10),     # Lower bound for the fifth parameter #theta[3]
        (-10, 10),     # Lower bound for the sixth parameter #theta[4]
        (-10, None)   # Lower bound for the seventh parameter #theta[5]
        ]

    soln = sp_minimize(neg_log_like, initial_params, method="Nelder-Mead", args=(gp,),
                    bounds=bounds)
    opt_gp = set_params(soln.x, gp)
    
    return opt_gp

def rotation_period(time, flux):
    """rotation period based on LS periodogram"""
    ls = LombScargle(time, flux)
    frequency, power = ls.autopower(minimum_frequency=1 / 10, maximum_frequency=1 / 0.1)
    period = 1 / frequency[np.argmax(power)]
    return period

def fit_gp_tiny(lc, flare_mask):
    """Fit a Gaussian Process to a lightcurve, ignoring flares.

    Parameters
    ----------
    lc : :class:`lightkurve.lightcurve.TessLightCurve`
        Lightcurve to fit
    flare_mask : :class:`numpy.ndarray`
        Boolean mask of flares in the lightcurve

    Returns
    -------
    gp_mean : :class:`numpy.ndarray`
        Optimized Gaussian Process mean
    """
    # mask the flares out of the lightcurve
    x = lc.time.value[~flare_mask]
    y = lc.flux.value[~flare_mask]
    y_err = lc.flux_err.value[~flare_mask]

    period = rotation_period(x, y.astype(np.float64))

    build_gp, init = rotation(period, y_err.mean(), long_scale=5)
    mu, nll = gp_model(x, y.astype(np.float64), build_gp)

    # optimization
    gp_params = minimize(
        nll, init, ["log_sigma", "log_short_scale", "log_short_sigma", "log_long_sigma"]
    )
    gp_params = minimize(nll, gp_params)

    mu, _ = gp_model(x, y.astype(np.float64), build_gp)
    gp_mean = mu(gp_params)

    return gp_mean
