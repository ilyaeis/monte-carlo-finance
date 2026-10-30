"""Monte Carlo option pricing and variance reduction.

All European estimators price a call or put under Black-Scholes:
    S_T = S0 exp((r - sigma^2/2) T + sigma sqrt(T) Z),   price = E[e^{-rT} payoff(S_T)].
Each one takes n (number of payoff evaluations) and a numpy Generator.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .black_scholes import geometric_asian_call
from .gbm import gbm_paths

Z95 = 1.959963984540054


@dataclass
class MCResult:
    price: float
    se: float  # standard error of the estimate
    n: int

    @property
    def ci95(self):
        return self.price - Z95 * self.se, self.price + Z95 * self.se


def _mean_se(y):
    return MCResult(float(y.mean()), float(y.std(ddof=1) / np.sqrt(len(y))), len(y))


def terminal_price(S0, T, r, sigma, z):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def payoff(s, K, kind):
    return np.maximum(s - K, 0.0) if kind == "call" else np.maximum(K - s, 0.0)


def discounted_payoffs(S0, K, T, r, sigma, z, kind="call"):
    return np.exp(-r * T) * payoff(terminal_price(S0, T, r, sigma, z), K, kind)


def price_plain(S0, K, T, r, sigma, n, rng, kind="call"):
    """Crude Monte Carlo: average of n independent discounted payoffs."""
    return _mean_se(discounted_payoffs(S0, K, T, r, sigma, rng.standard_normal(n), kind))


def price_antithetic(S0, K, T, r, sigma, n, rng, kind="call"):
    """Antithetic variates: n/2 pairs (Z, -Z), averaged inside each pair.

    Var of a pair mean = (Var Y + Cov(Y, Y')) / 2, so it helps when the
    payoff is monotone in Z (Cov < 0).
    """
    z = rng.standard_normal(n // 2)
    y = 0.5 * (discounted_payoffs(S0, K, T, r, sigma, z, kind)
               + discounted_payoffs(S0, K, T, r, sigma, -z, kind))
    res = _mean_se(y)
    return MCResult(res.price, res.se, 2 * len(y))


def price_control_variate(S0, K, T, r, sigma, n, rng, kind="call"):
    """Control variate X = e^{-rT} S_T, whose mean S0 is known exactly.

    Y_cv = Y - b (X - S0) with b = Cov(Y, X) / Var(X) estimated from the
    same sample. The variance drops by the factor (1 - rho^2).
    """
    z = rng.standard_normal(n)
    s_t = terminal_price(S0, T, r, sigma, z)
    y = np.exp(-r * T) * payoff(s_t, K, kind)
    x = np.exp(-r * T) * s_t
    b = np.cov(y, x)[0, 1] / x.var(ddof=1)
    return _mean_se(y - b * (x - S0))


def price_moment_matching(S0, K, T, r, sigma, n, rng, kind="call"):
    """Moment matching: rescale the normals to sample mean 0 and sd 1 exactly.

    The draws are no longer independent, so the reported SE is only a rough guide.
    """
    z = rng.standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)
    return _mean_se(discounted_payoffs(S0, K, T, r, sigma, z, kind))


def price_stratified(S0, K, T, r, sigma, n, rng, kind="call", n_strata=None):
    """Stratified sampling on U = Phi(Z): equal-probability strata, equal allocation."""
    n_strata = n_strata or min(n, 1000)
    per = n // n_strata
    u = (np.arange(n_strata)[:, None] + rng.random((n_strata, per))) / n_strata
    y = discounted_payoffs(S0, K, T, r, sigma, norm.ppf(u), kind)
    se = np.sqrt(y.var(axis=1, ddof=1).mean() / (n_strata * per))
    return MCResult(float(y.mean()), float(se), n_strata * per)


def price_importance(S0, K, T, r, sigma, n, rng, kind="call", theta=None):
    """Importance sampling: draw Z ~ N(theta, 1) and reweight by the likelihood ratio.

    The default theta moves the median of S_T onto the strike, which helps a
    lot for deep out-of-the-money calls where most plain draws pay zero.
    """
    if theta is None:
        theta = (np.log(K / S0) - (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    z = theta + rng.standard_normal(n)
    weight = np.exp(-theta * z + 0.5 * theta**2)
    return _mean_se(discounted_payoffs(S0, K, T, r, sigma, z, kind) * weight)


ESTIMATORS = {
    "Plain MC": price_plain,
    "Antithetic": price_antithetic,
    "Control variate": price_control_variate,
    "Moment matching": price_moment_matching,
    "Stratified": price_stratified,
    "Importance sampling": price_importance,
}


# --- path-dependent options -------------------------------------------------

def asian_call(S0, K, T, r, sigma, n_steps, n_paths, rng, control=True):
    """Arithmetic-average Asian call, optionally with the geometric Asian as control."""
    _, s = gbm_paths(S0, r, sigma, T, n_steps, n_paths, rng)
    path = s[:, 1:]  # average over t_1..t_n, not S0
    disc = np.exp(-r * T)
    y = disc * np.maximum(path.mean(axis=1) - K, 0.0)
    if not control:
        return _mean_se(y)
    x = disc * np.maximum(np.exp(np.log(path).mean(axis=1)) - K, 0.0)
    b = np.cov(y, x)[0, 1] / x.var(ddof=1)
    return _mean_se(y - b * (x - geometric_asian_call(S0, K, T, r, sigma, n_steps)))


def lookback_call_discrete(S0, K, T, r, sigma, n_steps, n_paths, rng):
    """Fixed-strike lookback call, max taken over n_steps grid points (plus S0).

    Discrete monitoring misses the peaks between grid points, so the estimate
    is biased low. The bias shrinks as n_steps grows.
    """
    _, s = gbm_paths(S0, r, sigma, T, n_steps, n_paths, rng)
    return _mean_se(np.exp(-r * T) * np.maximum(s.max(axis=1) - K, 0.0))
