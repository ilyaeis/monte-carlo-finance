"""Closed-form prices used as references for the Monte Carlo estimates."""
import numpy as np
from scipy.stats import norm


def bs_price(S0, K, T, r, sigma, kind="call"):
    """Black-Scholes price of a European call or put (no dividends)."""
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if kind == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if kind == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError(f"unknown option kind: {kind}")


def geometric_asian_call(S0, K, T, r, sigma, n_steps):
    """Exact price of a discretely monitored geometric-average Asian call.

    The geometric mean of GBM prices at t_i = i T / n is log-normal, so the
    price has a Black-Scholes-like closed form. It is the control variate
    for the arithmetic Asian call.
    """
    n = n_steps
    mu_g = np.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    var_g = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    sd_g = np.sqrt(var_g)
    d1 = (mu_g - np.log(K) + var_g) / sd_g
    d2 = d1 - sd_g
    return np.exp(-r * T) * (np.exp(mu_g + 0.5 * var_g) * norm.cdf(d1) - K * norm.cdf(d2))


def lookback_fixed_call(S0, K, T, r, sigma):
    """Continuous-monitoring fixed-strike lookback call, K >= S0 (Conze-Viswanathan)."""
    if K < S0:
        raise ValueError("formula assumes K >= S0")
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    k = 2 * r / sigma**2
    return (S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            + S0 * np.exp(-r * T) / k
            * (-(S0 / K) ** (-k) * norm.cdf(d1 - 2 * r * sqrt_t / sigma)
               + np.exp(r * T) * norm.cdf(d1)))
