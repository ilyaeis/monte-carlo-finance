import numpy as np
import pytest

from mcfin.black_scholes import bs_price, geometric_asian_call, lookback_fixed_call
from mcfin.garch import simulate_garch
from mcfin.mc_pricing import (ESTIMATORS, asian_call, lookback_call_discrete, price_antithetic,
                              price_plain)
from mcfin.risk import historical_var_es, kupiec_pof, parametric_var_es
from mcfin.rng import box_muller, lcg, lcg_normals

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def test_park_miller_known_value():
    # Park & Miller (1988): from seed 1, the 10,000th value is 1043618065.
    assert round(lcg(10_000)[-1] * (2**31 - 1)) == 1_043_618_065


def test_box_muller_gives_standard_normals():
    z = lcg_normals(100_000)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1) < 0.02
    z1, z2 = box_muller(np.array([np.exp(-0.5)]), np.array([0.0]))
    assert z1[0] == pytest.approx(1.0) and z2[0] == pytest.approx(0.0)


def test_black_scholes_reference_and_put_call_parity():
    call, put = bs_price(S0, K, T, R, SIGMA), bs_price(S0, K, T, R, SIGMA, "put")
    assert call == pytest.approx(10.450583572, abs=1e-8)  # textbook value
    assert call - put == pytest.approx(S0 - K * np.exp(-R * T))


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_every_estimator_brackets_black_scholes(name):
    rng = np.random.default_rng(1)
    res = ESTIMATORS[name](S0, K, T, R, SIGMA, 200_000, rng)
    exact = bs_price(S0, K, T, R, SIGMA)
    assert abs(res.price - exact) < 4 * res.se + 1e-3
    lo, hi = res.ci95
    assert lo < res.price < hi


def test_put_pricing():
    rng = np.random.default_rng(2)
    res = price_plain(S0, K, T, R, SIGMA, 200_000, rng, kind="put")
    assert abs(res.price - bs_price(S0, K, T, R, SIGMA, "put")) < 4 * res.se


def test_antithetic_reduces_standard_error_for_monotone_payoff():
    rng = np.random.default_rng(3)
    plain = price_plain(S0, K, T, R, SIGMA, 100_000, rng)
    anti = price_antithetic(S0, K, T, R, SIGMA, 100_000, rng)
    assert anti.se < plain.se


def test_asian_control_variate_matches_and_shrinks_error():
    rng = np.random.default_rng(4)
    kw = dict(S0=S0, K=K, T=T, r=R, sigma=SIGMA, n_steps=12, n_paths=50_000, rng=rng)
    plain, cv = asian_call(control=False, **kw), asian_call(control=True, **kw)
    assert cv.se < plain.se / 10
    assert abs(cv.price - plain.price) < 4 * plain.se
    # arithmetic mean >= geometric mean, so the arithmetic Asian is worth more
    assert cv.price > geometric_asian_call(S0, K, T, R, SIGMA, 12)


def test_lookback_discrete_is_biased_low():
    rng = np.random.default_rng(5)
    exact = lookback_fixed_call(S0, K, T, R, SIGMA)
    coarse = lookback_call_discrete(S0, K, T, R, SIGMA, 12, 50_000, rng)
    assert coarse.price < exact - 4 * coarse.se


def test_var_es_on_known_sample():
    returns = -np.arange(1, 101) / 1000  # losses 0.001 .. 0.100
    var, es = historical_var_es(returns, level=0.95)
    assert var == pytest.approx(np.quantile(np.arange(1, 101) / 1000, 0.95))
    assert es >= var
    r = np.random.default_rng(6).normal(0, 0.01, 500_000)
    var_n, es_n = parametric_var_es(r, 0.99)
    assert var_n == pytest.approx(0.02326, rel=0.01)
    assert es_n == pytest.approx(0.02665, rel=0.01)


def test_kupiec():
    lr, p = kupiec_pof(1000, 10, 0.99)  # exactly the expected count
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    _, p_bad = kupiec_pof(1000, 30, 0.99)
    assert p_bad < 0.01


def test_garch_simulation_has_long_run_variance():
    rng = np.random.default_rng(7)
    omega, alpha, beta = 0.05, 0.08, 0.90
    r, _ = simulate_garch(omega, alpha, beta, 200_000, rng)
    assert r.var() == pytest.approx(omega / (1 - alpha - beta), rel=0.1)
