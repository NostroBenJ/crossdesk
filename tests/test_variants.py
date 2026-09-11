"""The variant engine, and the three properties that make its numbers readable.

The load-bearing test is `test_generalised_engine_reproduces_the_registered_l5`.
Every variant is `spread()` with one argument changed, so if `spread()` does not
reproduce `cross_sectional_momentum` exactly when handed the registered scoring
and weighting rules, then every "improvement" measured against the baseline is
partly an artefact of a second, unnoticed change. The sibling `cisd-bot` repo
found five such divergences between a research harness and a live engine that
were each individually invisible.

The other two that matter:

* `test_jobson_korkie_agrees_with_a_paired_bootstrap` -- the Sharpe-difference
  standard error is an algebraic claim, and this project does not accept
  algebraic claims without a numerical check.
* the causality pair -- corrupting the FUTURE must not move a score or a
  volatility scale. That is the only defence against the failure mode that
  produced a flawless meaningless equity curve elsewhere in this work.
"""

import math
import random
import statistics as st
from datetime import date

import pytest

from crossdesk.data.panel import PricePanel
from crossdesk.data.universe import Listing, PointInTimeUniverse
from crossdesk.screen.families import cross_sectional_momentum
from crossdesk.screen.variants import (
    _bought,
    align,
    bootstrap_sharpe_difference,
    equal_weights,
    inverse_vol_weights,
    market_series,
    monthly_returns,
    ols,
    raw_momentum_score,
    residual_momentum_score,
    sharpe_difference_z,
    spread,
    volatility_scaled,
)

D = date


def _dates(n: int) -> list[date]:
    """`n` month ends, 2010-01-31 onward."""
    out = []
    for k in range(n):
        y, m = 2010 + (k // 12), (k % 12) + 1
        last = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}[m]
        out.append(D(y, m, last))
    return out


def _honest_builder(listings, asof):
    return sorted(l.symbol for l in listings if l.is_active(asof))


def _random_panel(n_names: int, dates: list[date], seed: int,
                  delist: dict[str, tuple[date, float]] | None = None,
                  ) -> PricePanel:
    """A panel with a market factor plus idiosyncratic noise, and delistings.

    The delistings are the part that matters: the baseline pays terminal
    returns, so a reproduction test on a panel without them would not prove the
    generalised engine handles the case the panel exists for.
    """
    rng = random.Random(seed)
    delist = delist or {}
    prices: dict[str, dict[date, float]] = {}
    listings = []
    market = [rng.gauss(0.008, 0.04) for _ in dates]
    for k in range(n_names):
        s = f"N{k:03d}"
        beta = rng.uniform(0.4, 1.6)
        px = 100.0
        series: dict[date, float] = {}
        end = delist.get(s)
        for i, d in enumerate(dates):
            series[d] = px
            px *= 1.0 + beta * market[i] + rng.gauss(0.0, 0.06)
            if end is not None and d >= end[0]:
                break
        prices[s] = series
        if end is None:
            listings.append(Listing(s, dates[0]))
        else:
            listings.append(Listing(s, dates[0], end=end[0],
                                    delisting_return=end[1]))
    return PricePanel(prices, PointInTimeUniverse(listings))


# --------------------------------------------------------------------------
# THE LOAD-BEARING TEST
# --------------------------------------------------------------------------

def test_generalised_engine_reproduces_the_registered_l5():
    dates = _dates(48)
    panel = _random_panel(40, dates, seed=11,
                          delist={"N007": (dates[30], -1.0),
                                  "N019": (dates[22], 0.0)})

    baseline = cross_sectional_momentum(panel, _honest_builder, dates,
                                        n_per_side=5, lookback_months=12,
                                        cost_bp=10.0)
    generalised = spread(panel, _honest_builder, dates,
                         score_fn=raw_momentum_score(panel, dates, 12),
                         weight_fn=equal_weights, n_per_side=5,
                         cost_bp=10.0, start=12)

    assert generalised.months == baseline.months
    assert generalised.long_names == baseline.long_names
    assert generalised.short_names == baseline.short_names
    assert generalised.unscored == baseline.unscored
    assert generalised.provenance == baseline.provenance
    for a, b in zip(generalised.gross_bp, baseline.gross_bp):
        assert a == pytest.approx(b, abs=1e-9)
    for a, b in zip(generalised.cost_fraction, baseline.cost_fraction):
        assert a == pytest.approx(b, abs=1e-12)
    for a, b in zip(generalised.net_bp(10.0), baseline.net_bp(10.0)):
        assert a == pytest.approx(b, abs=1e-9)
    # and the reproduction must be on a panel where delistings actually paid
    assert baseline.provenance.get("delisted", 0) > 0


def test_weighted_turnover_reduces_to_the_registered_set_measure():
    old = {s: 0.2 for s in ("A", "B", "C", "D", "E")}
    new = {s: 0.2 for s in ("A", "B", "C", "X", "Y")}
    assert _bought(old, new) == pytest.approx(2 / 5)
    assert _bought({}, new) == 0.0          # first month sells nothing
    assert _bought(old, old) == pytest.approx(0.0)


def test_weighted_turnover_sees_a_reweight_the_set_measure_misses():
    """Same names, different weights, is real trading and must be charged."""
    old = {"A": 0.5, "B": 0.5}
    new = {"A": 0.8, "B": 0.2}
    assert _bought(old, new) == pytest.approx(0.3)


# --------------------------------------------------------------------------
# regression arithmetic
# --------------------------------------------------------------------------

def test_ols_recovers_a_known_line_and_leaves_orthogonal_residuals():
    x = [0.01 * k - 0.1 for k in range(40)]
    y = [2.0 + 3.0 * xi for xi in x]
    a, b, resid = ols(y, x)
    assert a == pytest.approx(2.0, abs=1e-12)
    assert b == pytest.approx(3.0, abs=1e-12)
    assert max(abs(r) for r in resid) < 1e-12

    rng = random.Random(5)
    noisy = [2.0 + 3.0 * xi + rng.gauss(0, 0.05) for xi in x]
    _a, _b, resid = ols(noisy, x)
    assert sum(resid) == pytest.approx(0.0, abs=1e-12)
    assert sum(r * xi for r, xi in zip(resid, x)) == pytest.approx(0.0, abs=1e-12)


def test_residual_momentum_ignores_a_name_that_only_rose_with_the_market():
    """This is the variant's whole claim, as a test.

    RIDER doubles every market move and has no alpha, so it will be near the
    top of a RAW momentum ranking in a rising sample purely for its beta.
    PICKER matches the market and then adds idiosyncratic return in the
    formation window. Residual momentum must separate them.
    """
    dates = _dates(40)
    rng = random.Random(19)
    market = [None] + [rng.gauss(0.01, 0.04) for _ in range(len(dates) - 1)]
    rider = [None] + [2.0 * m for m in market[1:]]           # beta 2, no alpha
    picker = [None] + [m + rng.gauss(0.0, 0.005) for m in market[1:]]
    for k in range(len(dates) - 12, len(dates)):             # late alpha
        picker[k] += 0.03
    rets = {"RIDER": rider, "PICKER": picker}
    score = residual_momentum_score(rets, market, estimation=36, formation=12)

    # A perfectly explained series has zero residual variance -> unrankable,
    # which is the honest answer rather than a fabricated zero.
    assert score("RIDER", 39) is None
    assert score("PICKER", 39) > 2.0

    # ... and the score responds to the ALPHA, not to the market exposure:
    # double the idiosyncratic return and the score rises; double the beta and
    # it does not exist at all, which is the separation the variant claims.
    louder = list(picker)
    for k in range(len(dates) - 12, len(dates)):
        louder[k] += 0.03
    doubled = residual_momentum_score({"PICKER": louder}, market)("PICKER", 39)
    assert doubled > score("PICKER", 39)


# --------------------------------------------------------------------------
# causality -- the future must not be readable
# --------------------------------------------------------------------------

def test_residual_momentum_score_cannot_see_the_future():
    dates = _dates(60)
    rng = random.Random(3)
    market = [None] + [rng.gauss(0.008, 0.04) for _ in range(len(dates) - 1)]
    row = [None] + [rng.gauss(0.01, 0.06) for _ in range(len(dates) - 1)]
    score = residual_momentum_score({"A": list(row)}, market)

    at_40 = score("A", 40)
    corrupted = list(row)
    for i in range(41, len(dates)):
        corrupted[i] = 99.0
    assert score.__call__ is not None
    later = residual_momentum_score({"A": corrupted}, market)("A", 40)
    assert later == pytest.approx(at_40, abs=1e-15)


def test_volatility_scale_cannot_see_the_month_it_scales():
    from crossdesk.screen.families import SpreadResult

    dates = _dates(40)
    rng = random.Random(7)
    res = SpreadResult(name="base")
    res.months = dates[:36]
    res.gross_bp = [rng.gauss(50, 300) for _ in range(36)]
    res.long_bp = list(res.gross_bp)
    res.short_bp = [0.0] * 36
    res.cost_fraction = [0.5] * 36
    res.long_names = [()] * 36
    res.short_names = [()] * 36

    scaled = volatility_scaled(res, lookback=12, cap=99.0)
    first_scale = scaled.scales[0]

    bumped = SpreadResult(name="base")
    bumped.months = list(res.months)
    bumped.gross_bp = list(res.gross_bp)
    bumped.gross_bp[12] = 50_000.0            # the month being scaled
    bumped.long_bp = list(bumped.gross_bp)
    bumped.short_bp = list(res.short_bp)
    bumped.cost_fraction = list(res.cost_fraction)
    bumped.long_names = list(res.long_names)
    bumped.short_names = list(res.short_names)

    assert volatility_scaled(bumped, lookback=12, cap=99.0).scales[0] == \
        pytest.approx(first_scale, abs=1e-15)


def test_volatility_scale_is_exactly_target_over_realised():
    from crossdesk.screen.families import SpreadResult

    dates = _dates(30)
    pattern = [100.0, -100.0] * 13            # sd is exactly 100 * sqrt(26/25)
    res = SpreadResult(name="base")
    res.months = dates[:26]
    res.gross_bp = pattern
    res.long_bp = list(pattern)
    res.short_bp = [0.0] * 26
    res.cost_fraction = [0.0] * 26
    res.long_names = [()] * 26
    res.short_names = [()] * 26

    scaled = volatility_scaled(res, target_annual_vol=0.12, lookback=12, cap=99.0)
    sd = st.stdev(pattern[:12])
    expected = 10_000.0 * 0.12 / math.sqrt(12.0) / sd
    assert scaled.scales[0] == pytest.approx(expected, rel=1e-12)
    assert scaled.result.gross_bp[0] == pytest.approx(expected * pattern[12], rel=1e-12)


def test_volatility_scaling_scales_cost_with_the_position():
    from crossdesk.screen.families import SpreadResult

    dates = _dates(30)
    res = SpreadResult(name="base", cost_bp=10.0)
    res.months = dates[:26]
    res.gross_bp = [100.0, -100.0] * 13
    res.long_bp = list(res.gross_bp)
    res.short_bp = [0.0] * 26
    res.cost_fraction = [0.4] * 26
    res.long_names = [()] * 26
    res.short_names = [()] * 26

    scaled = volatility_scaled(res, lookback=12, cap=99.0)
    assert scaled.result.cost_fraction[0] == pytest.approx(
        0.4 * scaled.scales[0], rel=1e-12)


# --------------------------------------------------------------------------
# weighting
# --------------------------------------------------------------------------

def test_inverse_vol_halves_the_weight_of_a_name_with_twice_the_volatility():
    dates = _dates(40)
    calm = [None] + [0.01 if k % 2 else -0.01 for k in range(len(dates) - 1)]
    wild = [None] + [0.02 if k % 2 else -0.02 for k in range(len(dates) - 1)]
    w = inverse_vol_weights({"CALM": calm, "WILD": wild}, lookback=36)(
        ("CALM", "WILD"), 39)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)
    assert w["CALM"] == pytest.approx(2.0 * w["WILD"], rel=1e-9)


def test_inverse_vol_falls_back_rather_than_dropping_a_selected_name():
    dates = _dates(40)
    calm = [None] + [0.01 if k % 2 else -0.01 for k in range(len(dates) - 1)]
    w = inverse_vol_weights({"CALM": calm}, lookback=36)(("CALM", "NEW"), 39)
    assert set(w) == {"CALM", "NEW"}
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)
    assert w["NEW"] > 0.0


# --------------------------------------------------------------------------
# the Sharpe-difference instrument, checked numerically
# --------------------------------------------------------------------------

def _correlated_pair(n: int, rho: float, mu_a: float, mu_b: float, seed: int):
    rng = random.Random(seed)
    a, b = [], []
    for _ in range(n):
        z1, z2 = rng.gauss(0, 1), rng.gauss(0, 1)
        a.append(mu_a + z1)
        b.append(mu_b + rho * z1 + math.sqrt(1 - rho * rho) * z2)
    return a, b


def test_jobson_korkie_agrees_with_a_paired_bootstrap():
    """The closed form and the resampling must tell the same story.

    Checked as a decision, not as a decimal: on a pair with no true difference
    both must fail to reject, and on a pair with a large one both must reject.
    A tolerance on the p-values themselves would be a tolerance on bootstrap
    noise, not on the algebra.
    """
    same_a, same_b = _correlated_pair(240, 0.9, 0.05, 0.05, seed=17)
    z_same = sharpe_difference_z(same_a, same_b)
    _diff, p_same = bootstrap_sharpe_difference(same_a, same_b, draws=4000)
    assert abs(z_same) < 1.96
    assert p_same > 0.05

    big_a, big_b = _correlated_pair(240, 0.9, 0.45, 0.05, seed=17)
    z_big = sharpe_difference_z(big_a, big_b)
    _diff, p_big = bootstrap_sharpe_difference(big_a, big_b, draws=4000)
    assert abs(z_big) > 2.58
    assert p_big < 0.01


def test_correlation_is_what_makes_the_paired_test_sharper():
    """Same Sharpe gap, more correlation, larger statistic. The whole reason
    an unpaired comparison of these variants would be the wrong instrument."""
    lo_a, lo_b = _correlated_pair(240, 0.0, 0.25, 0.05, seed=23)
    hi_a, hi_b = _correlated_pair(240, 0.95, 0.25, 0.05, seed=23)
    assert abs(sharpe_difference_z(hi_a, hi_b)) > abs(sharpe_difference_z(lo_a, lo_b))


def test_identical_series_have_exactly_zero_difference():
    a, _b = _correlated_pair(120, 0.5, 0.1, 0.1, seed=31)
    assert sharpe_difference_z(a, list(a)) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------

def test_align_restricts_both_series_to_shared_months():
    from crossdesk.screen.families import SpreadResult

    dates = _dates(6)
    a = SpreadResult(name="a")
    a.months = dates[:5]
    a.gross_bp = [10.0, 20.0, 30.0, 40.0, 50.0]
    a.cost_fraction = [0.0] * 5

    b = SpreadResult(name="b")
    b.months = dates[2:6]
    b.gross_bp = [1.0, 2.0, 3.0, 4.0]
    b.cost_fraction = [0.0] * 4

    xa, xb, months = align(a, b, 0.0)
    assert months == dates[2:5]
    assert xa == [30.0, 40.0, 50.0]
    assert xb == [1.0, 2.0, 3.0]


def test_monthly_return_matrix_excludes_delisting_returns_from_the_series():
    """A terminal return ends a name's series; it does not extend it.

    The holding leg is still paid it -- that is `period_return`'s job, and
    `test_generalised_engine_reproduces_the_registered_l5` proves it still
    happens. This only keeps the terminal month out of the REGRESSION inputs.
    """
    dates = _dates(12)
    panel = _random_panel(3, dates, seed=2, delist={"N001": (dates[8], -1.0)})
    rets = monthly_returns(panel, ["N000", "N001"], dates)
    assert all(v is None for v in rets["N001"][9:])
    assert rets["N000"][5] is not None


def test_market_series_uses_membership_from_the_start_of_the_month():
    dates = _dates(6)
    panel = _random_panel(5, dates, seed=4)
    rets = monthly_returns(panel, [f"N{k:03d}" for k in range(5)], dates)
    mkt = market_series(rets, _honest_builder, panel.universe.listings, dates)
    assert mkt[0] is None
    assert all(m is not None for m in mkt[1:])
    expected = st.fmean([rets[f"N{k:03d}"][3] for k in range(5)])
    assert mkt[3] == pytest.approx(expected, rel=1e-12)
