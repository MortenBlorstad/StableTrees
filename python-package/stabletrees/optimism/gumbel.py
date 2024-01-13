import numpy as np
import math

PI = math.pi

def pgumbel(q, location, scale, lower_tail=True, log_p=False):
    z = (q - location) / scale
    log_px = -np.exp(-z)  # log p(X <= x)

    if lower_tail and log_p:
        res = log_px
    elif lower_tail and not log_p:
        res = np.exp(log_px)
    elif not lower_tail and log_p:
        res = np.log(1.0 - np.exp(log_px))
    else:
        res = 1.0 - np.exp(log_px)

    if np.isnan(res):
        return 1.0
    else:
        return res

def grad_scale_est_obj(scale, x):
    exp_x_beta = np.exp(-1.0 * x / scale)
    f = scale + np.sum(x * exp_x_beta) / np.sum(exp_x_beta) - np.mean(x)
    grad = 2.0 * f * (1.0 + (np.sum(x * x * exp_x_beta) * np.sum(exp_x_beta) - np.power(np.sum(x * exp_x_beta), 2.0)) / np.power(scale * np.sum(exp_x_beta), 2.0))
    return grad

def scale_estimate(x):
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    scale_est = np.sqrt(var * 6.0) / PI

    NITER = 50  # max iterations
    EPS = 1e-2  # precision
    step_length = 0.2  # conservative
    for i in range(NITER):
        step = -step_length * grad_scale_est_obj(scale_est, x)
        scale_est += step
        if abs(step) <= EPS:
            break

    return scale_est

def par_gumbel_estimates(x):
    scale_est = scale_estimate(x)
    location_est = scale_est * (math.log(len(x)) - np.log(np.sum(np.exp(-1.0 * x / scale_est))))
    return np.array([location_est, scale_est])
