"""Value at Risk and Expected Shortfall, plus backtests.

Conventions: `returns` are daily simple returns, VaR and ES are positive loss
numbers (fractions of portfolio value) at confidence level `level`, e.g. 0.99.
"""
import numpy as np
import pandas as pd
from scipy import stats

from arch import arch_model


def historical_var_es(returns, level=0.99):
    """Historical simulation: the empirical loss quantile and the mean loss beyond it."""
    losses = -np.asarray(returns)
    var = np.quantile(losses, level)
    return var, losses[losses >= var].mean()


def parametric_var_es(returns, level=0.99):
    """Normal (variance-covariance) VaR and ES from the sample mean and sd."""
    mu, sd = np.mean(returns), np.std(returns, ddof=1)
    z = stats.norm.ppf(level)
    return -mu + sd * z, -mu + sd * stats.norm.pdf(z) / (1 - level)


def mc_var_es(simulated_returns, level=0.99):
    """Monte Carlo VaR and ES: the same as historical, on simulated scenarios."""
    return historical_var_es(simulated_returns, level)


def kupiec_pof(n_obs, n_breaches, level=0.99):
    """Kupiec proportion-of-failures test. Returns (LR statistic, p-value).

    H0: the breach probability equals 1 - level.
    """
    p = 1 - level
    x, n = n_breaches, n_obs
    phat = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + (x * np.log(phat) if x > 0 else 0.0)
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, df=1)


def rolling_var(returns, window=750, level=0.99, refit_every=20, n_sims=100_000, seed=0):
    """One-day-ahead VaR forecasts with three methods, without look-ahead.

    - historical: empirical quantile of the last `window` returns
    - normal: normal VaR from the last `window` returns
    - garch_mc: GARCH(1,1)-t fitted on the last `window` returns (refit every
      `refit_every` days). The next-day loss distribution is simulated by
      Monte Carlo from the fitted Student-t shocks scaled by the GARCH sigma.
    Returns a DataFrame indexed by forecast date with VaR, ES and the realised return.
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna()
    r_pct = 100 * r
    rows = []
    for start in range(window, len(r), refit_every):
        hist = r.iloc[start - window:start]
        res = arch_model(r_pct.iloc[start - window:start], vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        p = res.params
        nu = p["nu"]
        # standardised t shocks (unit variance), simulated once per refit block
        shocks = rng.standard_t(nu, n_sims) * np.sqrt((nu - 2) / nu)
        q_var, q_es = historical_var_es(shocks, level)  # in units of sigma, sign handled below
        # sigma_t for the block, filtered with the fixed params (uses data up to t-1 only)
        end = min(start + refit_every, len(r))
        fixed = arch_model(r_pct.iloc[:end], vol="GARCH", p=1, q=1, dist="t").fix(p)
        sig = fixed.conditional_volatility.iloc[start:end] / 100
        mu = p["mu"] / 100
        h_var, h_es = historical_var_es(hist, level)
        n_var, n_es = parametric_var_es(hist, level)
        for t, date in enumerate(r.index[start:end]):
            rows.append({
                "date": date, "ret": r.iloc[start + t],
                "hist_var": h_var, "hist_es": h_es,
                "normal_var": n_var, "normal_es": n_es,
                "garch_var": -mu + sig.iloc[t] * q_var, "garch_es": -mu + sig.iloc[t] * q_es,
                "garch_sigma": sig.iloc[t],
            })
    return pd.DataFrame(rows).set_index("date")


def backtest(df, methods=("hist", "normal", "garch"), level=0.99):
    """Count VaR breaches (loss > VaR) and run the Kupiec test per method."""
    out = []
    for m in methods:
        breaches = int((-df["ret"] > df[f"{m}_var"]).sum())
        lr, pval = kupiec_pof(len(df), breaches, level)
        out.append({"method": m, "days": len(df), "breaches": breaches,
                    "expected": (1 - level) * len(df), "breach_rate": breaches / len(df),
                    "kupiec_LR": lr, "p_value": pval,
                    "avg_VaR": df[f"{m}_var"].mean(), "avg_ES": df[f"{m}_es"].mean()})
    return pd.DataFrame(out)


def ks_normality_by_subset(returns, n_subsets=3):
    """Kolmogorov-Smirnov test against N(mean, sd) on chronological subsets.

    Same idea as my lab script: split the series into equal parts and test each.
    Note: estimating mean and sd from the same data makes the plain KS p-value
    conservative (a Lilliefors-type issue). Rejections are still meaningful.
    """
    r = returns.dropna()
    rows = []
    bounds = np.linspace(0, len(r), n_subsets + 1).astype(int)
    for i in range(n_subsets):
        part = r.iloc[bounds[i]:bounds[i + 1]]
        mu, sd = part.mean(), part.std(ddof=1)
        ks, pval = stats.kstest(part.values, stats.norm(mu, sd).cdf)
        rows.append({"subset": i + 1, "from": part.index[0].date(), "to": part.index[-1].date(),
                     "n": len(part), "ann_vol": sd * np.sqrt(252), "kurtosis": stats.kurtosis(part),
                     "KS": ks, "p_value": pval})
    return pd.DataFrame(rows)
