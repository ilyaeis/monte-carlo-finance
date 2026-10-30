"""Mean-variance portfolios, Sharpe ratio and CAPM beta.

Written from my securities-course notes (there was no code there, only
worked examples), so this is a clean re-implementation.
"""
import numpy as np
import pandas as pd


def annualize(daily_returns, periods=252):
    """Annual mean vector and covariance matrix from daily simple returns."""
    return daily_returns.mean() * periods, daily_returns.cov() * periods


def portfolio_stats(w, mu, cov, rf=0.0):
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    return ret, vol, (ret - rf) / vol


def random_portfolios(mu, cov, n, rng, rf=0.0):
    """Monte Carlo cloud of long-only portfolios (weights ~ flat Dirichlet)."""
    w = rng.dirichlet(np.ones(len(mu)), n)
    ret = w @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    return w, ret, vol, (ret - rf) / vol


def tangency_weights(mu, cov, rf):
    """Maximum-Sharpe portfolio with short sales allowed: w ~ Sigma^{-1} (mu - rf)."""
    x = np.linalg.solve(cov, mu - rf)
    return x / x.sum()


def frontier(mu, cov, targets):
    """Minimum-variance frontier (short sales allowed) via the closed-form Lagrange solution."""
    inv = np.linalg.inv(cov)
    ones = np.ones(len(mu))
    a, b, c = ones @ inv @ ones, ones @ inv @ mu, mu @ inv @ mu
    d = a * c - b**2
    var = (a * targets**2 - 2 * b * targets + c) / d
    return np.sqrt(var)


def capm_table(asset_returns, market_returns, rf_daily):
    """OLS of excess returns on market excess returns: alpha, beta, R^2 per asset."""
    mx = market_returns - rf_daily
    rows = []
    for name in asset_returns:
        y = asset_returns[name] - rf_daily
        beta = np.cov(y, mx)[0, 1] / mx.var(ddof=1)
        alpha = y.mean() - beta * mx.mean()
        resid = y - alpha - beta * mx
        r2 = 1 - resid.var() / y.var()
        rows.append({"asset": name, "beta": beta, "alpha_annual": 252 * alpha, "R2": r2,
                     "ann_return": 252 * asset_returns[name].mean(),
                     "ann_vol": np.sqrt(252) * asset_returns[name].std(),
                     "sharpe": np.sqrt(252) * y.mean() / y.std()})
    return pd.DataFrame(rows).set_index("asset")
