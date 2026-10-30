"""Run every experiment, save all figures to figures/ and all numbers to results/.

    python scripts/run_all.py          # everything (about 2-3 minutes)
    python scripts/run_all.py --no-gif # skip the animation

All random draws use fixed seeds, so the numbers are reproducible.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcfin import garch, portfolio, risk  # noqa: E402
from mcfin.black_scholes import bs_price, geometric_asian_call, lookback_fixed_call  # noqa: E402
from mcfin.gbm import gbm_paths  # noqa: E402
from mcfin.mc_pricing import (ESTIMATORS, asian_call, lookback_call_discrete,  # noqa: E402
                              price_antithetic, price_control_variate, price_importance, price_plain)

FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# Option used throughout: at-the-money one-year call.
S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20

# --- one dark style for every figure ------------------------------------------------
BG, PANEL = "#1a1a19", "#1a1a19"
INK, INK2, GRID = "#ffffff", "#c3c2b7", "#383835"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED = (
    "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767")
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
    "figure.dpi": 100, "savefig.dpi": 140,
})

RESULTS = {}


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}")


def load_returns():
    px = pd.read_csv(ROOT / "data" / "prices.csv", index_col=0, parse_dates=True)
    rf_daily = (px["IRX"] / 100 / 252).shift(1)
    rets = px.drop(columns="IRX").pct_change()
    return px, rets.iloc[1:], rf_daily.iloc[1:]


# --- 1. hero: GBM fan + terminal distribution -----------------------------------------
def hero():
    print("1. GBM fan (hero)")
    rng = np.random.default_rng(7)
    mu, sigma = 0.08, 0.20
    t, s = gbm_paths(S0, mu, sigma, T, 252, 4000, rng)
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.03)
    ax, axh = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for path in s[:250]:
        ax.plot(t, path, color=BLUE, alpha=0.08, lw=0.8)
    lo, med, hi = np.percentile(s, [5, 50, 95], axis=0)
    ax.fill_between(t, lo, hi, color=BLUE, alpha=0.12, lw=0)
    ax.plot(t, med, color=INK, lw=2, label="median path")
    ax.plot(t, lo, color=ORANGE, lw=1.5, ls="--", label="5% / 95% quantiles")
    ax.plot(t, hi, color=ORANGE, lw=1.5, ls="--")
    ax.plot(t, S0 * np.exp(mu * t), color=AQUA, lw=1.5, label=r"mean  $S_0 e^{\mu t}$")
    ax.set(xlim=(0, T), ylim=(40, 200), xlabel="time (years)", ylabel="price")
    ax.set_title("Geometric Brownian motion: 4,000 one-year price paths (250 drawn)")
    ax.legend(loc="upper left")
    st = s[:, -1]
    bins = np.linspace(40, 200, 41)
    axh.hist(st, bins=bins, orientation="horizontal", density=True, color=BLUE, alpha=0.6, rwidth=0.85)
    y = np.linspace(40, 200, 400)
    m, v = np.log(S0) + (mu - 0.5 * sigma**2) * T, sigma * np.sqrt(T)
    axh.plot(stats.lognorm.pdf(y, v, scale=np.exp(m)), y, color=INK, lw=1.8)
    axh.set(ylim=(40, 200), yticklabels=[], xticks=[])
    axh.set_title(r"$S_T$ is log-normal", fontsize=11)
    axh.grid(False)
    save(fig, "hero_gbm_fan.png")


# --- 2. convergence plain vs antithetic vs control variate ------------------------------
def running(y):
    n = np.arange(1, len(y) + 1)
    mean = np.cumsum(y) / n
    var = (np.cumsum(y**2) - n * mean**2) / np.maximum(n - 1, 1)
    return n, mean, 1.96 * np.sqrt(var / n)


def convergence():
    print("2. Convergence to Black-Scholes")
    bs = bs_price(S0, K, T, R, SIGMA)
    rng = np.random.default_rng(11)
    n = 200_000
    z = rng.standard_normal(n)
    disc = np.exp(-R * T)
    st = S0 * np.exp((R - 0.5 * SIGMA**2) * T + SIGMA * np.sqrt(T) * z)
    y = disc * np.maximum(st - K, 0)
    st_a = S0 * np.exp((R - 0.5 * SIGMA**2) * T - SIGMA * np.sqrt(T) * z[: n // 2])
    y_anti = 0.5 * (y[: n // 2] + disc * np.maximum(st_a - K, 0))
    x = disc * st
    b = np.cov(y, x)[0, 1] / x.var()
    y_cv = y - b * (x - S0)

    fig, ax = plt.subplots(figsize=(11, 5))
    for label, yy, step, color in [("Plain MC", y, 1, BLUE), ("Antithetic", y_anti, 2, ORANGE),
                                   ("Control variate", y_cv, 1, AQUA)]:
        k, mean, half = running(yy)
        k = k * step  # x axis = number of payoff evaluations
        sel = k >= 100
        ax.fill_between(k[sel], (mean - half)[sel], (mean + half)[sel], color=color, alpha=0.15, lw=0)
        ax.plot(k[sel], mean[sel], color=color, lw=1.6, label=label)
    ax.axhline(bs, color=INK, lw=1.3, ls="--")
    ax.text(n * 0.95, bs - 0.35, f"Black-Scholes = {bs:.4f}", color=INK, fontsize=10, ha="right")
    ax.set_xscale("log")
    ax.set(xlim=(100, n), ylim=(bs - 2.5, bs + 2.5), xlabel="number of payoff evaluations  n",
           ylabel="option price")
    ax.set_title("Monte Carlo price of an ATM call with 95% confidence bands")
    ax.legend(loc="upper right")
    save(fig, "convergence.png")


# --- 3. variance reduction study ----------------------------------------------------------
def repeat(fn, reps, rng, **kw):
    t0 = time.perf_counter()
    est = np.array([fn(rng=rng, **kw).price for _ in range(reps)])
    return est, (time.perf_counter() - t0) / reps


def variance_reduction():
    print("3. Variance reduction (repeated runs)")
    rng = np.random.default_rng(2024)
    reps, n = 400, 10_000
    rows = []
    bs = bs_price(S0, K, T, R, SIGMA)
    base = dict(S0=S0, K=K, T=T, r=R, sigma=SIGMA, n=n)
    est_p, t_p = repeat(price_plain, reps, rng, **base)
    for name, fn in ESTIMATORS.items():
        if name == "Importance sampling":
            continue
        est, t = (est_p, t_p) if name == "Plain MC" else repeat(fn, reps, rng, **base)
        rows.append({"case": "ATM call (K=100)", "method": name, "mean": est.mean(), "exact": bs,
                     "sd": est.std(ddof=1), "var_reduction": est_p.var() / est.var(),
                     "efficiency_gain": est_p.var() * t_p / (est.var() * t)})
    # deep out-of-the-money call: importance sampling shines
    k_otm = 160.0
    bs_otm = bs_price(S0, k_otm, T, R, SIGMA)
    otm = dict(base, K=k_otm)
    e1, t1 = repeat(price_plain, reps, rng, **otm)
    e2, t2 = repeat(price_importance, reps, rng, **otm)
    for name, e, t in [("Plain MC", e1, t1), ("Importance sampling", e2, t2)]:
        rows.append({"case": "Deep OTM call (K=160)", "method": name, "mean": e.mean(), "exact": bs_otm,
                     "sd": e.std(ddof=1), "var_reduction": e1.var() / e.var(),
                     "efficiency_gain": e1.var() * t1 / (e.var() * t)})
    # arithmetic Asian call with the geometric Asian as control variate
    steps, reps_a, n_a = 50, 100, 10_000
    a_kw = dict(S0=S0, K=K, T=T, r=R, sigma=SIGMA, n_steps=steps, n_paths=n_a)
    t0 = time.perf_counter()
    a_plain = np.array([asian_call(rng=rng, control=False, **a_kw).price for _ in range(reps_a)])
    ta = (time.perf_counter() - t0) / reps_a
    t0 = time.perf_counter()
    a_cv = np.array([asian_call(rng=rng, control=True, **a_kw).price for _ in range(reps_a)])
    tc = (time.perf_counter() - t0) / reps_a
    for name, e, t in [("Plain MC", a_plain, ta), ("Geometric-Asian control", a_cv, tc)]:
        rows.append({"case": "Asian call (50 dates)", "method": name, "mean": e.mean(), "exact": np.nan,
                     "sd": e.std(ddof=1), "var_reduction": a_plain.var() / e.var(),
                     "efficiency_gain": a_plain.var() * ta / (e.var() * t)})
    df = pd.DataFrame(rows)
    RESULTS["variance_reduction"] = df.to_dict("records")
    RESULTS["variance_reduction_setup"] = dict(n=n, reps=reps, asian_reps=reps_a, asian_n=n_a)

    plot = df[df.method != "Plain MC"].copy()
    plot["label"] = plot.method + "\n" + plot.case
    plot = plot.sort_values("var_reduction")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    yy = np.arange(len(plot))
    ax.barh(yy, plot.var_reduction, color=BLUE, height=0.6)
    for yi, v in zip(yy, plot.var_reduction):
        ax.text(v * 1.08, yi, f"{v:,.1f}x", va="center", color=INK, fontsize=10)
    ax.set_xscale("log")
    ax.axvline(1, color=INK2, lw=1)
    ax.set_yticks(yy, plot.label, fontsize=9.5)
    ax.set_xlim(0.5, plot.var_reduction.max() * 4)
    ax.set_xlabel("variance reduction vs plain Monte Carlo (same n, log scale)")
    ax.set_title(f"Same budget, smaller error: variance of {reps} repeated estimates")
    ax.grid(axis="y", visible=False)
    save(fig, "variance_reduction.png")


# --- 4. lookback discretisation bias --------------------------------------------------------
def lookback():
    print("4. Lookback discretisation bias")
    rng = np.random.default_rng(5)
    exact = lookback_fixed_call(S0, K, T, R, SIGMA)
    rows = []
    for m in [12, 52, 252, 1000]:
        res = lookback_call_discrete(S0, K, T, R, SIGMA, m, 40_000, rng)
        rows.append({"monitoring_dates": m, "estimate": res.price, "se": res.se, "bias": res.price - exact})
    RESULTS["lookback"] = {"exact_continuous": exact, "rows": rows}


# --- 5. GARCH on SPY ----------------------------------------------------------------------
def garch_fit(rets):
    print("5. GARCH(1,1) on SPY")
    spy = rets["SPY"].dropna()
    res = garch.fit_garch(100 * spy, dist="t")
    summ = garch.summarize(res)
    RESULTS["garch"] = {k: float(v) for k, v in summ.items()}
    RESULTS["garch"]["loglik"] = float(res.loglikelihood)
    RESULTS["garch"]["sample"] = f"{spy.index[0].date()} to {spy.index[-1].date()}"
    RESULTS["ks"] = risk.ks_normality_by_subset(spy).astype({"from": str, "to": str}).to_dict("records")
    sig_ann = res.conditional_volatility * np.sqrt(252)
    roll = spy.rolling(21).std() * np.sqrt(252) * 100

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(roll.index, roll, color=INK2, lw=1, alpha=0.7, label="21-day realised volatility")
    ax.plot(sig_ann.index, sig_ann, color=ORANGE, lw=1.4, label="GARCH(1,1)-t conditional volatility")
    sample_vol = 100 * spy.std() * np.sqrt(252)
    RESULTS["garch"]["sample_vol_annual_pct"] = float(sample_vol)
    ax.axhline(sample_vol, color=AQUA, ls="--", lw=1.3, label=f"sample volatility {sample_vol:.1f}%")
    peak = sig_ann.idxmax()
    ax.annotate(f"COVID crash\n{sig_ann.max():.0f}% annualised", xy=(peak, sig_ann.max()),
                xytext=(pd.Timestamp("2021-06-01"), sig_ann.max() * 0.85), color=INK, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set(ylabel="annualised volatility (%)", ylim=(0, None))
    ax.set_title(f"Volatility clusters: SPY daily returns, persistence alpha+beta = {summ['persistence']:.3f}")
    ax.legend(loc="upper left")
    save(fig, "garch_volatility.png")


# --- 6. VaR backtest ---------------------------------------------------------------------
def var_backtest(rets):
    print("6. Rolling 99% VaR backtest (takes a minute)")
    spy = rets["SPY"].dropna()
    df = risk.rolling_var(spy, window=750, level=0.99, refit_every=20)
    bt = risk.backtest(df, level=0.99)
    RESULTS["var_backtest"] = bt.to_dict("records")
    RESULTS["var_backtest_period"] = f"{df.index[0].date()} to {df.index[-1].date()}"
    # whole-sample one-day 99% VaR / ES today (last window), three methods
    last = spy.iloc[-750:]
    h = risk.historical_var_es(last)
    p = risk.parametric_var_es(last)
    g = (df["garch_var"].iloc[-1], df["garch_es"].iloc[-1])
    RESULTS["var_latest"] = {"historical": h, "normal": p, "garch_mc": g, "date": str(df.index[-1].date())}

    fig, ax = plt.subplots(figsize=(12, 5))
    ret = 100 * df["ret"]
    ax.bar(ret.index, ret, width=1.5, color=INK2, alpha=0.55, label="SPY daily return")
    ax.plot(df.index, -100 * df["hist_var"], color=BLUE, lw=1.4, label="historical VaR (750 days)")
    ax.plot(df.index, -100 * df["garch_var"], color=ORANGE, lw=1.2, label="GARCH-t Monte Carlo VaR")
    for col, color, marker in [("hist_var", BLUE, "o"), ("garch_var", ORANGE, "^")]:
        br = df[-df["ret"] > df[col]]
        ax.scatter(br.index, 100 * br["ret"], s=26, color=color, marker=marker, zorder=3,
                   edgecolor=BG, linewidth=0.8)
    ax.set(ylabel="daily return (%)", ylim=(-14, 11))
    b = bt.set_index("method")["breaches"]
    exp_ = bt["expected"].iloc[0]
    ax.set_title(f"99% one-day VaR backtest: breaches  historical {b['hist']}, GARCH {b['garch']}"
                 f"  (expected {exp_:.0f})")
    ax.legend(loc="upper left", ncol=3)
    save(fig, "var_backtest.png")


# --- 7. efficient frontier + CAPM -----------------------------------------------------------
def frontier(rets, rf_daily):
    print("7. Efficient frontier and CAPM")
    assets = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "KO", "NVDA"]
    r = rets[assets].dropna()
    rf = float(rf_daily.loc[r.index].mean() * 252)
    mu, cov = portfolio.annualize(r)
    mu, cov = mu.values, cov.values
    rng = np.random.default_rng(3)
    w, pret, pvol, psh = portfolio.random_portfolios(mu, cov, 30_000, rng, rf)
    wt = portfolio.tangency_weights(mu, cov, rf)
    t_ret, t_vol, t_sh = portfolio.portfolio_stats(wt, mu, cov, rf)
    best = np.argmax(psh)
    capm = portfolio.capm_table(r, rets["SPY"].loc[r.index], rf_daily.loc[r.index])
    RESULTS["capm"] = capm.reset_index().to_dict("records")
    RESULTS["portfolio"] = {
        "rf": rf, "tangency_return": t_ret, "tangency_vol": t_vol, "tangency_sharpe": t_sh,
        "tangency_weights": dict(zip(assets, wt.round(3))),
        "best_long_only_sharpe": float(psh[best]), "best_long_only_weights": dict(zip(assets, w[best].round(3))),
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(100 * pvol, 100 * pret, c=psh, cmap="Blues", s=4, alpha=0.8, vmin=psh.min())
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Sharpe ratio", color=INK2)
    cb.outline.set_visible(False)
    targets = np.linspace(mu.min(), max(mu.max(), t_ret) * 1.05, 300)
    ax.plot(100 * portfolio.frontier(mu, cov, targets), 100 * targets, color=INK, lw=1.6,
            label="minimum-variance frontier (shorting allowed)")
    xs = np.linspace(0, 100 * pvol.max(), 50)
    ax.plot(xs, 100 * rf + t_sh * xs, color=ORANGE, ls="--", lw=1.5, label="capital market line")
    ax.scatter([100 * t_vol], [100 * t_ret], s=120, color=ORANGE, marker="*", zorder=4, label="tangency portfolio")
    ax.scatter([100 * pvol[best]], [100 * pret[best]], s=70, color=AQUA, marker="D", zorder=4,
               edgecolor=BG, label="best long-only random portfolio")
    for i, a in enumerate(assets):
        v, m_ = 100 * np.sqrt(cov[i, i]), 100 * mu[i]
        ax.scatter([v], [m_], s=30, color=INK, zorder=4)
        off = (-30, 2) if a == "JNJ" else (5, 3)
        ax.annotate(a, (v, m_), xytext=off, textcoords="offset points", fontsize=9.5, color=INK)
    ax.set(xlabel="annualised volatility (%)", ylabel="annualised mean return (%)",
           xlim=(0, 100 * max(np.sqrt(np.diag(cov)).max(), t_vol) * 1.08))
    ax.set_title(f"30,000 random long-only portfolios of 7 stocks ({r.index[0].year}-{r.index[-1].year}, in-sample)")
    ax.legend(loc="lower right", fontsize=9.5)
    save(fig, "efficient_frontier.png")


# --- 8. GIF: paths fan out, price converges -------------------------------------------------
def gif():
    print("8. Animation (GIF)")
    rng = np.random.default_rng(42)
    n_max, shown = 20_000, 150
    t, s = gbm_paths(S0, R, SIGMA, T, 100, n_max, rng)
    y = np.exp(-R * T) * np.maximum(s[:, -1] - K, 0)
    k, mean, half = running(y)
    bs = bs_price(S0, K, T, R, SIGMA)
    frames = np.unique(np.geomspace(1, n_max, 70).astype(int))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.2]})
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.86, wspace=0.22)
    ax1.set(xlim=(0, T), ylim=(40, 200), xlabel="time (years)", ylabel="price")
    ax1.axhline(K, color=INK2, lw=1, ls=":")
    ax2.set(xscale="log", xlim=(1, n_max), ylim=(bs - 6, bs + 6), xlabel="number of paths  n")
    ax2.axhline(bs, color=INK, ls="--", lw=1.3)
    ax2.text(1.3, bs + 0.4, f"Black-Scholes {bs:.3f}", color=INK, fontsize=10)
    line, = ax2.plot([], [], color=ORANGE, lw=2)
    band = [ax2.fill_between([], [], [], color=ORANGE, alpha=0.2)]
    title = fig.suptitle("", x=0.07, ha="left", fontsize=13, fontweight="bold")
    drawn = [0]

    def update(i):
        n = frames[i]
        for j in range(drawn[0], min(n, shown)):
            itm = s[j, -1] > K
            ax1.plot(t, s[j], color=AQUA if itm else BLUE, alpha=0.35, lw=0.8)
        drawn[0] = max(drawn[0], min(n, shown))
        line.set_data(k[:n], mean[:n])
        band[0].remove()
        sel = slice(1, n)
        band[0] = ax2.fill_between(k[sel], (mean - half)[sel], (mean + half)[sel], color=ORANGE, alpha=0.2, lw=0)
        title.set_text(f"n = {n:>6,}   MC price = {mean[n-1]:.3f} +/- {half[n-1]:.3f}")
        return line,

    seq = list(range(len(frames))) + [len(frames) - 1] * 15  # hold the last frame
    anim = FuncAnimation(fig, update, frames=seq, init_func=lambda: (line,), blit=False)
    anim.save(FIG / "mc_convergence.gif", writer=PillowWriter(fps=11), dpi=72)
    plt.close(fig)
    print("  saved figures/mc_convergence.gif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()
    _, rets, rf_daily = load_returns()
    RESULTS["black_scholes"] = {"call": bs_price(S0, K, T, R, SIGMA), "put": bs_price(S0, K, T, R, SIGMA, "put"),
                                "geometric_asian_50": geometric_asian_call(S0, K, T, R, SIGMA, 50)}
    hero()
    convergence()
    variance_reduction()
    lookback()
    garch_fit(rets)
    var_backtest(rets)
    frontier(rets, rf_daily)
    if not args.no_gif:
        gif()
    (RES / "results.json").write_text(json.dumps(RESULTS, indent=2, default=float))
    print("saved results/results.json")


if __name__ == "__main__":
    main()
