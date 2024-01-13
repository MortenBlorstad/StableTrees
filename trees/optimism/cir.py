import numpy as np

def rnchisq(df, lambda_):
    if df < 0 or lambda_ < 0:
        return np.nan

    if lambda_ == 0:
        return 0 if df == 0 else np.random.gamma(df / 2.0, 2.0)

    r = np.random.poisson(lambda_ / 2.)
    result = np.random.chisquare(2. * r) if r > 0 else 0
    if df > 0:
        result += np.random.gamma(df / 2.0, 2.0)
    return result

def cir_sim_vec(m):
    EPS = 1e-12
    delta_time = 1.0 / (m + 1.0)
    u_cirsim = np.linspace(delta_time, 1.0 - delta_time, m)
    tau = 0.5 * np.log((u_cirsim * (1 - EPS)) / (EPS * (1.0 - u_cirsim)))
    tau_delta = tau[1:] - tau[:-1]

    # Parameters of CIR
    a, b, sigma = 2.0, 1.0, 2.0 * np.sqrt(2.0)
    c, ncchisq = 0.0, 0.0

    res = np.zeros(m)
    res[0] = np.random.gamma(0.5, 2.0)

    for i in range(1, m):
        c = 2.0 * a / (sigma**2 * (1.0 - np.exp(-a * tau_delta[i-1])))
        ncchisq = rnchisq(4.0 * a * b / sigma**2, 2.0 * c * res[i-1] * np.exp(-a * tau_delta[i-1]))
        res[i] = ncchisq / (2.0 * c)

    return res

def cir_sim_mat(nsim, nobs,random_state=1):
    np.random.seed(random_state)
    res = np.zeros((nsim, nobs))
    for i in range(nsim):
        res[i, :] = cir_sim_vec(nobs)
    return res

def simpson(fval, grid):
    n = len(grid) - 1
    h = (grid[-1] - grid[0]) / n
    s = 0

    if n == 2:
        s = fval[0] + 4.0 * fval[1] + fval[2]
    else:
        s = fval[0] + fval[n]
        s += 4.0 * np.sum(fval[1:n:2])
        s += 2.0 * np.sum(fval[2:n-1:2])
    
    s = s * h / 3.0
    return s


def interpolate_cir(u, cir_sim):
    EPS = 1e-12
    cir_obs = cir_sim.shape[1]
    n_timesteps = len(u)
    n_sim = cir_sim.shape[0]

    delta_time = 1.0 / (cir_obs + 1.0)
    u_cirsim = np.linspace(delta_time, 1.0 - delta_time, cir_obs)
    tau_sim = 0.5 * np.log((u_cirsim * (1 - EPS)) / (EPS * (1.0 - u_cirsim)))
    tau = 0.5 * np.log((u * (1 - EPS)) / (EPS * (1.0 - u)))

    lower_ind = np.zeros(n_timesteps, dtype=int)
    upper_ind = np.zeros(n_timesteps, dtype=int)
    lower_weight = np.zeros(n_timesteps)
    upper_weight = np.zeros(n_timesteps)

    i = 0
    while i < n_timesteps and tau[i] <= tau_sim[0]:
        lower_ind[i] = 0
        upper_ind[i] = 0
        lower_weight[i] = 1.0
        upper_weight[i] = 0.0
        i += 1

    for i in range(i, n_timesteps):
        if tau_sim[cir_obs - 1] < tau[i]:
            break
        for j in range(cir_obs - 1):
            if tau_sim[j] < tau[i] and tau[i] <= tau_sim[j + 1]:
                lower_ind[i] = j
                upper_ind[i] = j + 1
                lower_weight[i] = 1.0 - (tau[i] - tau_sim[j]) / (tau_sim[j + 1] - tau_sim[j])
                upper_weight[i] = 1.0 - lower_weight[i]
                break

    for i in range(i, n_timesteps):
        if tau[i] > tau_sim[cir_obs - 1]:
            lower_ind[i] = cir_obs - 1
            upper_ind[i] = 0
            lower_weight[i] = 1.0
            upper_weight[i] = 0.0

    cir_interpolated = np.zeros((n_sim, n_timesteps))
    for i in range(n_sim):
        for j in range(n_timesteps):
            cir_interpolated[i, j] = cir_sim[i, lower_ind[j]] * lower_weight[j] + cir_sim[i, upper_ind[j]] * upper_weight[j]

    return cir_interpolated

def rmax_cir(u, cir_sim):
    nsplits = len(u)
    simsplits = cir_sim.shape[1]
    nsims = cir_sim.shape[0]
    max_cir_obs = np.zeros(nsims)

    if nsplits < simsplits:
        tau = 0.5 * np.log((u * (1 - 1e-12)) / (1e-12 * (1.0 - u)))
        cir_obs = interpolate_cir(u, cir_sim)
        max_cir_obs = np.max(cir_obs, axis=1)
    else:
        max_cir_obs = np.max(cir_sim, axis=1)

    return max_cir_obs

def estimate_shape_scale(max_cir):
    n = len(max_cir)
    mean = np.mean(max_cir)
    var = np.var(max_cir, ddof=1)
    shape = mean**2 / var
    scale = var / mean
    return np.array([shape, scale])

def pmax_cir(x, obs):
    return np.mean(obs <= x)



