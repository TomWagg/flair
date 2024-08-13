import celerite2
from celerite2 import terms
from scipy.optimize import minimize
import numpy as np


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
        print(theta)
        if theta[1] <= 0.0 or theta[3] <= 0.0 or theta[4] <= 0.0 or np.isinf(theta[1]) or np.isinf(theta[3]) or np.isinf(theta[4]) or theta[1] >= 10**100 or theta[3] >= 10**100 or theta[4] >= 10**100:
            return gp, False
        gp.kernel = celerite2.terms.SHOTerm(
            sigma=theta[0], rho=theta[1], tau=theta[2]
        ) + celerite2.terms.SHOTerm(sigma=theta[3], rho=theta[4], Q=0.25)
        gp.compute(x, diag=y_err**2 + theta[5], quiet=True)
        return gp, True

    # calculate the negative log likelihood
    def neg_log_like(params, gp):
        gp, flag = set_params(params, gp)
        if not flag:
            return 1e10
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
    soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(gp,))
    opt_gp, _ = set_params(soln.x, gp)
    
    return opt_gp
