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
import jaxopt
#from nuance.kernels import rotation
from nuance.utils import minimize
from nuance.core import gp_model
from nuance.utils import sigma_clip_mask

jax.config.update("jax_enable_x64", True)

from tinygp import GaussianProcess, kernels

@jit
def Rotation(sigma, period, Q0, dQ, f):
    """
    A kernel for a rotating star with a single mode of oscillation.
    """
    Q1 = 1 / 2 + Q0 + dQ
    w1 = (4 * jnp.pi * Q1) / (period * jnp.sqrt(4 * Q1**2 - 1))
    s1 = sigma**2 / ((1 + f) * w1 * Q1)

    Q2 = 1 / 2 + Q0
    w2 = (8 * jnp.pi * Q1) / (period * jnp.sqrt(4 * Q1**2 - 1))
    s2 = f * sigma**2 / ((1 + f) * w2 * Q2)
    kernel = kernels.quasisep.SHO(w1, Q1, s1) + kernels.quasisep.SHO(w2, Q2, s2)
    return kernel


def rotation(period=None, error=None, mean=None, long_scale=0.5):
    initial_params = {
        "log_period": jnp.log(period) if period is not None else jnp.log(1.0),
        "log_Q": jnp.log(100),
        "log_sigma": jnp.log(1e-1),
        "log_dQ": jnp.log(100),
        "log_f": jnp.log(2.0),
        "log_long_sigma": jnp.log(1e-2),
        "log_jitter": jnp.log(1.0) if error is None else jnp.log(error),
    }

    def build_gp(params, time):
        jitter2 = 2*jnp.exp(2 * params["log_jitter"])
        long_sigma = jnp.exp(params["log_long_sigma"])

        kernel = (
            Rotation(
                jnp.exp(params["log_sigma"]),
                jnp.exp(params["log_period"]),
                jnp.exp(params["log_Q"]),
                jnp.exp(params["log_dQ"]),
                jnp.exp(params["log_f"]),
            )
            + kernels.quasisep.Exp(long_scale, long_sigma)
        )

        return GaussianProcess(kernel, time, diag=jitter2, mean=mean)

    return build_gp, initial_params

def gp_model(x, y, build_gp, X=None):

    if X is None:
        X = jnp.atleast_2d(jnp.ones_like(x))

    @jax.jit
    def nll_w(params):
        gp = build_gp(params, x)
        Liy = gp.solver.solve_triangular(y)
        LiX = gp.solver.solve_triangular(X.T)
        LiXT = LiX.T
        LiX2 = LiXT @ LiX
        w = jnp.linalg.lstsq(LiX2, LiXT @ Liy)[0]
        nll = -gp.log_probability(y - w @ X)
        return nll, w

    @jax.jit
    def nll(params):
        return nll_w(params)[0]

    @jax.jit
    def gp_(params):
        gp = build_gp(params, x)
        _, w = nll_w(params)
        cond_gp = gp.condition(y - w @ X, x).gp
        return cond_gp.loc + w @ X

    return gp_, nll

def full_time_incorporated(x, y, full_time, build_gp, gp_params):
    
    @jax.jit
    def fit(gp_params): 
        opt_gp = build_gp(gp_params, x)
        gp_cond = opt_gp.condition(y, full_time).gp
        return gp_cond.loc
    
    return fit

def fit_gp_tiny(lc, flare_mask, period):
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
    
    build_gp, init = rotation(period, y_err.mean(), mean=np.mean(y), long_scale=200)

    gp_, nll = gp_model(x, y.astype(np.float64), build_gp)

    # optimization
    gp_params = minimize(nll, init, ["log_sigma"])
    gp_params = minimize(nll, gp_params)
    clipped_mask = np.ones_like(x).astype(bool)

    # Perform sigma clipping
    for _ in range(3):
        residuals = y - gp_(gp_params)
        clipped_mask = clipped_mask & sigma_clip_mask(residuals, sigma=4.0, window=20)
        gp_params = minimize(nll, gp_params)

    time_clipped_masked = x[clipped_mask]
    flux_clipped_masked = y[clipped_mask]

    gp_2, nll2 = gp_model(time_clipped_masked, flux_clipped_masked, build_gp)

    # optimization V2
    gp_params = minimize(nll2, gp_params, ["log_sigma", "log_long_sigma"])
    gp_params = minimize(nll2, gp_params, ["log_period"])
    gp_params = minimize(nll2, gp_params)

    # Fit the GP to the full time array
    gp_full = full_time_incorporated(x, y, full_time, build_gp, gp_params)
    gp_full_mean = gp_full(gp_params)
    
    print("GP fit complete")
    
    return gp_full_mean

# jax_mapping jax transform 

# The optimization might need to be done in parallel 

# Try and do masking outside of the function