import celerite2
from scipy.optimize import minimize
import numpy as np


def fit_GP(lc, flare_mask):
    x = lc.time.value[~flare_mask]
    y = lc.flux.value[~flare_mask]
    y_err = lc.flux_err.value[~flare_mask]

    def set_params(params, gp):
        gp.mean = params[0]
        theta = np.exp(params[1:])
        gp.kernel = celerite2.terms.SHOTerm(
            sigma=theta[0], rho=theta[1], tau=theta[2]
        ) + celerite2.terms.SHOTerm(sigma=theta[3], rho=theta[4], Q=0.25)
        gp.compute(x, diag=y_err**2 + theta[5], quiet=True)
        return gp

    def neg_log_like(params, gp):
        gp = set_params(params, gp)
        return -gp.log_likelihood(y)

    gp = celerite2.GaussianProcess(mean=0.0, kernel=celerite2.terms.SHOTerm(sigma=1.0))
    initial_params = [0.0, 0.0, 0.0, np.log(10.0), 0.0, np.log(5.0), np.log(0.01)]
    set_params(initial_params, gp)
    soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(gp,))
    opt_gp = set_params(soln.x, gp)
    
    return opt_gp
