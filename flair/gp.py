import celerite2
from celerite2 import terms
from scipy.optimize import minimize as sp_minimize
import numpy as np
from astropy.timeseries import LombScargle


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

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from tinygp import kernels, GaussianProcess
from astropy.timeseries import LombScargle
import jaxopt

jax.config.update("jax_enable_x64", True)

def rotation_period(time, flux):
    """rotation period based on LS periodogram"""
    ls = LombScargle(time, flux)
    frequency, power = ls.autopower(minimum_frequency=1 / 10, maximum_frequency=1 / 0.1)
    period = 1 / frequency[np.argmax(power)]
    return period

@jit
def build_gp(theta, X):
    # We want most of our parameters to be positive so we take the `exp` here
    # Note that we're using `jnp` instead of `np`
    amps = jnp.exp(theta["log_amps"])
    scales = jnp.exp(theta["log_scales"])

    # Construct the kernel by multiplying and adding `Kernel` objects
    k1 = amps[0] * kernels.ExpSquared(scales[0])
    k2 = (
        amps[1]
        * kernels.ExpSquared(scales[1])
        * kernels.ExpSineSquared(
            scale=jnp.exp(theta["log_period"]),
            gamma=jnp.exp(theta["log_gamma"]),
        )
    )
    k3 = amps[2] * kernels.RationalQuadratic(
        alpha=jnp.exp(theta["log_alpha"]), scale=scales[2]
    )
    k4 = amps[3] * kernels.ExpSquared(scales[3])
    kernel = k1 + k2 + k3 + k4

    return GaussianProcess(
        kernel, X, diag=jnp.exp(theta["log_diag"]), mean=theta["mean"]
    )
    
@jit
def loss(theta, y, time):
    return -build_gp(theta, time).log_probability(y)


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
    x = lc.time.value[~flare_mask].astype(np.float64)
    y = lc.flux.value[~flare_mask].astype(np.float64)
    y_err = lc.flux_err.value[~flare_mask].astype(np.float64)
    full_time = lc.time.value.astype(np.float64)
    

    period = rotation_period(x, y)

    theta_init = {
        "mean": np.float64(np.mean(y)),
        "log_diag": np.log(np.mean(y_err)),
        "log_amps": np.log([66.0, 24, 0.66, 0.18]),
        "log_scales": np.log([3, 15, 0.78, 1.6]),
        "log_period": np.log(period),
        "log_gamma": np.log(4.3),
        "log_alpha": np.log(1.2),
    }
    solver = jaxopt.ScipyMinimize(fun=loss)
    soln = solver.run(theta_init, y=y, time=x)
        
    opt_gp = build_gp(soln.params, x)
    gp_cond = opt_gp.condition(y, full_time).gp
    gp_full_mean, var = gp_cond.loc, gp_cond.variance

    return gp_full_mean, var
