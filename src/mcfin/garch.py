"""GARCH(1,1) volatility: fitting with `arch` and simulation from scratch.

    r_t = mu + eps_t,  eps_t = sigma_t z_t,
    sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2.
The process is stationary when alpha + beta < 1. Then the long-run variance
is omega / (1 - alpha - beta).
"""
import numpy as np
from arch import arch_model


def fit_garch(returns_pct, dist="t"):
    """Fit a constant-mean GARCH(1,1) to daily returns in percent."""
    return arch_model(returns_pct, mean="Constant", vol="GARCH", p=1, q=1, dist=dist).fit(disp="off")


def summarize(res):
    """Key numbers of a fitted model (daily returns in percent)."""
    p = res.params
    persistence = p["alpha[1]"] + p["beta[1]"]
    long_run_var = p["omega"] / (1 - persistence)
    return {
        "mu": p["mu"], "omega": p["omega"], "alpha": p["alpha[1]"], "beta": p["beta[1]"],
        "nu": p.get("nu", np.nan),
        "persistence": persistence,
        "half_life_days": np.log(0.5) / np.log(persistence),
        "long_run_vol_annual_pct": np.sqrt(252 * long_run_var),
    }


def simulate_garch(omega, alpha, beta, n, rng, mu=0.0):
    """Simulate a GARCH(1,1) path with normal shocks, started at the long-run variance."""
    var = omega / (1 - alpha - beta)
    eps_prev = 0.0
    sig = np.empty(n)
    r = np.empty(n)
    for t in range(n):
        var = omega + alpha * eps_prev**2 + beta * var
        sig[t] = np.sqrt(var)
        eps_prev = sig[t] * rng.standard_normal()
        r[t] = mu + eps_prev
    return r, sig
