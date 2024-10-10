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

import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize
from jax import jit
from tinygp import kernels, GaussianProcess
from astropy.timeseries import LombScargle

def rotation_period(time, flux):
    """rotation period based on LS periodogram"""
    ls = LombScargle(time, flux)
    frequency, power = ls.autopower(minimum_frequency=1 / 10, maximum_frequency=1 / 0.1)
    period = 1 / frequency[np.argmax(power)]
    return period

def build_gp(params, time, kernel_type):
    if kernel_type == "SHO":
        kernel = kernels.quasisep.SHO(
            jnp.exp(params["log_sigma"]),
            jnp.exp(params["log_period"]),
            jnp.exp(params["log_Q"]),
        )
    elif kernel_type == "ExpSquared":
        kernel = kernels.ExpSquared(
            jnp.exp(params["log_sigma"]),
            jnp.exp(params["log_scale"]),
        )
    elif kernel_type == "Matern32":
        kernel = kernels.Matern32(
            jnp.exp(params["log_sigma"]),
            jnp.exp(params["log_scale"]),
        )
    elif kernel_type == "Combined":
        kernel = kernels.ExpSquared(
            jnp.exp(params["log_sigma"]),
            jnp.exp(params["log_scale"]),
        ) + kernels.Matern32(
            jnp.exp(params["log_sigma"]),
            jnp.exp(params["log_scale"]),
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return GaussianProcess(kernel, time, diag=params["error"] ** 2, mean=1.0)

@jit
def gp_model(params, time, flux, kernel_type):
    gp = build_gp(params, time, kernel_type)
    return -gp.log_probability(flux)

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
    gp_full_mean : :class:`numpy.ndarray`
        GP prediction for the entire lightcurve, including flares
    """
    # Mask the flares out of the lightcurve
    x = lc.time.value[~flare_mask]
    y = lc.flux.value[~flare_mask]
    y_err = lc.flux_err.value[~flare_mask]

    period = rotation_period(x, y.astype(np.float64))

    initial_params = {
        "log_period": jnp.log(period),
        "log_Q": jnp.log(100),
        "log_sigma": jnp.log(1e-1),
        "log_scale": jnp.log(1.0),  # For ExpSquared and Matern32 kernels
        "error": jnp.mean(y_err),
    }

    kernel_types = ["SHO", "ExpSquared", "Matern32", "Combined"]
    best_nll = float('inf')
    best_params = None
    best_kernel_type = None

    for kernel_type in kernel_types:
        nll = lambda params: gp_model(params, x, y.astype(np.float64), kernel_type)
        gp_params = minimize(nll, initial_params, method="Nelder-Mead")
        current_nll = nll(gp_params)

        if current_nll < best_nll:
            best_nll = current_nll
            best_params = gp_params
            best_kernel_type = kernel_type

    # Predict the GP mean for the entire lightcurve, including flares
    full_time = lc.time.value
    gp = build_gp(best_params, full_time, best_kernel_type)
    gp_full_mean, gp_full_var = gp.predict(y.astype(np.float64), full_time, return_var=True)

    # Calculate residuals and check if the fit is within 3 sigma
    residuals = y - gp_full_mean[~flare_mask]
    sigma = np.sqrt(gp_full_var[~flare_mask])
    if np.any(np.abs(residuals) > 3 * sigma):
        print("Fit is not within 3 sigma, trying combined kernels.")
        combined_kernel_types = ["SHO+ExpSquared", "SHO+Matern32", "ExpSquared+Matern32"]
        for combined_kernel_type in combined_kernel_types:
            nll = lambda params: gp_model(params, x, y.astype(np.float64), combined_kernel_type)
            gp_params = minimize(nll, initial_params, method="Nelder-Mead")
            current_nll = nll(gp_params)

            if current_nll < best_nll:
                best_nll = current_nll
                best_params = gp_params
                best_kernel_type = combined_kernel_type

        # Predict again with the best combined kernel
        gp = build_gp(best_params, full_time, best_kernel_type)
        gp_full_mean, gp_full_var = gp.predict(y.astype(np.float64), full_time, return_var=True)

    return gp_full_mean
