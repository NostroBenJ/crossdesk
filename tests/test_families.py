"""The family tests: selection, delisting, costs, and the aggregation rule.

`test_pooling_name_months_inflates_the_t_by_sqrt_names` is the point of the
file. It builds a panel where the only thing that exists is the market factor,
so the honest number of independent observations is the number of months, and
measures what pooling name-months would have reported instead.
"""

import math
from datetime import date

import pytest

from crossdesk.data.panel import PricePanel
from crossdesk.data.universe import Listing, PointInTimeUniverse
from crossdesk.screen.families import (
    cross_sectional_momentum,
    max_drawdown,
    monthly_stats,
    pooled_t,
    trend_following,
)

D = date
DATES = [D(2020, 1, 31), D(2020, 2, 29), D(2020, 3, 31), D(2020, 4, 30)]


def _honest_builder(listings, asof):
    return sorted(l.symbol for l in listings if l.is_active(asof))


def _panel(series: dict[str, list[float]], listings=None) -> PricePanel:
    """Prices given as one list per name, aligned to DATES."""
    prices = {s: {d: v for d, v in zip(DATES, vals) if v is not None}
              for s, vals in series.items()}
    listings = listings or [Listing(s, DATES[0]) for s in series]
    return PricePanel(prices, PointInTimeUniverse(listings))


# --------------------------------------------------------------------------
# L5 selection and arithmetic
# --------------------------------------------------------------------------

def _momentum_fixture():
    # Lookback DATES[0] -> DATES[2]; hold DATES[2] -> DATES[3].
    # Ranking ascending: F(-20) E(-10) D(0) C(+5) B(+10) A(+20)
    return _panel({
        "A": [100, 100, 120, 132],   # top momentum, forward +10%
        "B": [100, 100, 110, 110],   # second, forward 0
        "C": [100, 100, 105, 105],
        "D": [100, 100, 100, 100],
        "E": [100, 100, 90, 90],     # second worst, forward 0
        "F": [100, 100, 80, 80],     # worst, forward 0
    })


def test_momentum_ranks_on_the_lookback_and_holds_the_next_month():
    r = cross_sectional_momentum(_momentum_fixture(), _honest_builder, DATES,
                                 n_per_side=2, lookback_months=2, cost_bp=0.0)
    assert r.n == 1
    assert r.long_names[0] == ("B", "A")     # ascending order, top two
    assert r.short_names[0] == ("F", "E")
    assert r.long_bp[0] == pytest.approx(500.0)   # mean(+10%, 0)
    assert r.short_bp[0] == pytest.approx(0.0)
    assert r.gross_bp[0] == pytest.approx(500.0)


def test_first_month_pays_no_turnover_because_nothing_is_sold():
    r = cross_sectional_momentum(_momentum_fixture(), _honest_builder, DATES,
                                 n_per_side=2, lookback_months=2, cost_bp=25.0)
    assert r.cost_fraction[0] == 0.0
    assert r.net_bp()[0] == pytest.approx(r.gross_bp[0])


def test_turnover_is_charged_on_both_legs_at_a_round_trip_each():
    """Two months, the long leg fully replaced between them."""
    dates = DATES + [D(2020, 5, 31)]
    prices = {
        # A leads at month 2, collapses so B leads at month 3.
        "A": [100, 100, 120, 121, 121],
        "B": [100, 100, 110, 200, 200],
        "C": [100, 100, 105, 105, 105],
        "D": [100, 100, 100, 100, 100],
        "E": [100, 100, 90, 90, 90],
        "F": [100, 100, 80, 80, 80],
    }
    panel = PricePanel(
        {s: {d: v for d, v in zip(dates, vals)} for s, vals in prices.items()},
        PointInTimeUniverse([Listing(s, dates[0]) for s in prices]),
    )
    r = cross_sectional_momentum(panel, _honest_builder, dates,
                                 n_per_side=1, lookback_months=2, cost_bp=10.0)
    assert r.n == 2
    assert r.long_names == [("A",), ("B",)]
    # One name in, one out, on the long leg only: turnover 1.0 + 0.0.
    assert r.cost_fraction[1] == pytest.approx(1.0)
    assert r.net_bp()[1] == pytest.approx(r.gross_bp[1] - 10.0)


def test_a_name_that_delists_while_held_contributes_its_terminal_return():
    """The whole point of the panel: a bankruptcy is -100%, not an absence."""
    listings = [Listing(s, DATES[0]) for s in "BCDEF"]
    listings.append(Listing("Z", DATES[0], DATES[2], reason="bankrupt"))
    panel = _panel({
        "Z": [100, 100, 200, None],   # best momentum, then delisted
        "B": [100, 100, 110, 110],
        "C": [100, 100, 105, 105],
        "D": [100, 100, 100, 100],
        "E": [100, 100, 90, 90],
        "F": [100, 100, 80, 80],
    }, listings)
    r = cross_sectional_momentum(panel, _honest_builder, DATES,
                                 n_per_side=2, lookback_months=2, cost_bp=0.0)
    assert r.n == 1
    assert "Z" in r.long_names[0]
    # mean(Z -100%, B 0%) = -50%
    assert r.long_bp[0] == pytest.approx(-5000.0)
    assert r.provenance["delisted"] == 1


def test_a_name_with_a_gap_in_the_lookback_is_not_ranked():
    """A restarted index alongside the old one is not a return."""
    listings = [Listing(s, DATES[0]) for s in "BCDEF"]
    listings.append(Listing("G", DATES[0], DATES[1], reason="priced"))
    listings.append(Listing("G", DATES[2]))
    panel = _panel({
        "G": [100, 100, 100, 300],
        "B": [100, 100, 110, 110],
        "C": [100, 100, 105, 105],
        "D": [100, 100, 100, 100],
        "E": [100, 100, 90, 90],
        "F": [100, 100, 80, 80],
    }, listings)
    r = cross_sectional_momentum(panel, _honest_builder, DATES,
                                 n_per_side=2, lookback_months=2, cost_bp=0.0)
    assert r.unscored >= 1
    assert "G" not in r.long_names[0] and "G" not in r.short_names[0]


def test_momentum_is_skipped_when_too_few_names_can_be_scored():
    panel = _panel({"A": [100, 100, 120, 130], "B": [100, 100, 110, 110]})
    r = cross_sectional_momentum(panel, _honest_builder, DATES,
                                 n_per_side=2, lookback_months=2)
    assert r.n == 0


def test_momentum_reads_no_price_after_the_formation_date():
    """Truncation test: removing the future cannot change the ranking.

    The registered form holds from the formation date, so the only future
    price it may touch is the one that ends the holding month.
    """
    full = _momentum_fixture()
    r_full = cross_sectional_momentum(full, _honest_builder, DATES,
                                      n_per_side=2, lookback_months=2)
    truncated = _panel({
        "A": [100, 100, 120, 132], "B": [100, 100, 110, 110],
        "C": [100, 100, 105, 105], "D": [100, 100, 100, 100],
        "E": [100, 100, 90, 90], "F": [100, 100, 80, 80],
    })
    r_cut = cross_sectional_momentum(truncated, _honest_builder, DATES[:3] + [DATES[3]],
                                     n_per_side=2, lookback_months=2)
    assert r_full.long_names == r_cut.long_names
    assert r_full.short_names == r_cut.short_names


# --------------------------------------------------------------------------
# L6 trend following
# --------------------------------------------------------------------------

def test_trend_filter_holds_above_the_average_and_sits_out_below():
    dates = DATES
    # UP rises so it is above its 3-month mean; DOWN falls so it is below.
    panel = PricePanel(
        {"UP": {dates[0]: 100.0, dates[1]: 110.0, dates[2]: 120.0, dates[3]: 132.0},
         "DOWN": {dates[0]: 120.0, dates[1]: 110.0, dates[2]: 100.0, dates[3]: 90.0}},
        PointInTimeUniverse([Listing("UP", dates[0]), Listing("DOWN", dates[0])]),
    )
    r = trend_following(panel, _honest_builder, dates, ma_months=3, cost_bp=0.0)
    assert r.n == 1
    # buy and hold: mean(+10%, -10%) = 0
    assert r.short_bp[0] == pytest.approx(0.0)
    # filtered: UP held (+10%), DOWN flat (0) -> mean +5%
    assert r.long_bp[0] == pytest.approx(500.0)
    assert r.gross_bp[0] == pytest.approx(500.0)
    # one of two names held, and the registered form charges every held month
    assert r.cost_fraction[0] == pytest.approx(0.5)


def test_trend_needs_a_full_unbroken_window():
    panel = PricePanel(
        {"X": {DATES[1]: 100.0, DATES[2]: 110.0, DATES[3]: 120.0}},
        PointInTimeUniverse([Listing("X", DATES[1])]),
    )
    r = trend_following(panel, _honest_builder, DATES, ma_months=3, cost_bp=0.0)
    assert r.n == 0
    assert r.unscored >= 1


# --------------------------------------------------------------------------
# THE AGGREGATION RULE
# --------------------------------------------------------------------------

def test_pooling_name_months_inflates_the_t_by_sqrt_names():
    """A market factor and nothing else: 30 names carry one month of news.

    Pooling treats them as 30 independent draws. With identical names the
    inflation has a closed form, so it is asserted exactly rather than
    approximately:

        t_pooled / t_monthly = sqrt(k) * sqrt( (k*M - 1) / (k * (M - 1)) )

    The first factor is the sqrt(names) everyone expects. The second is the
    Bessel correction, which differs between a 12-observation sample and a
    360-observation one and makes the real inflation slightly WORSE than
    sqrt(k) rather than slightly better.
    """
    monthly = [40.0, -25.0, 60.0, 10.0, -15.0, 35.0, -5.0, 20.0,
               -30.0, 50.0, 5.0, -10.0]
    k, m = 30, len(monthly)
    pooled = [r for r in monthly for _ in range(k)]

    honest = monthly_stats(monthly)
    assert honest is not None
    inflated = pooled_t(pooled)

    expected = math.sqrt(k) * math.sqrt((k * m - 1) / (k * (m - 1)))
    assert inflated / honest.t == pytest.approx(expected, rel=1e-12)
    assert expected > math.sqrt(k)
    # And the practical consequence: the same data, one verdict each way.
    assert abs(inflated) > 2.0 > abs(honest.t)


def test_monthly_stats_match_hand_computation():
    xs = [10.0, 20.0, 30.0, 40.0]
    s = monthly_stats(xs)
    assert s is not None
    assert s.n == 4
    assert s.mean_bp == pytest.approx(25.0)
    sd = math.sqrt(((15.0 ** 2) + (5.0 ** 2) + (5.0 ** 2) + (15.0 ** 2)) / 3)
    assert s.sd_bp == pytest.approx(sd)
    assert s.se_bp == pytest.approx(sd / 2.0)
    assert s.t == pytest.approx(25.0 / (sd / 2.0))
    assert s.annual_sharpe == pytest.approx(25.0 / sd * math.sqrt(12))
    assert s.annual_return_pct == pytest.approx(12 * 25.0 / 100.0)


def test_monthly_stats_refuse_a_degenerate_series():
    assert monthly_stats([1.0, 1.0]) is None          # too few
    assert monthly_stats([5.0, 5.0, 5.0, 5.0]) is None  # zero volatility


def test_max_drawdown_is_a_compounded_path_not_a_sum():
    # +50%, then -50%: equity 1.0 -> 1.5 -> 0.75, a 50% fall from the peak.
    assert max_drawdown([5000.0, -5000.0]) == pytest.approx(0.5)
    # A rising series never draws down.
    assert max_drawdown([100.0, 100.0, 100.0]) == pytest.approx(0.0)
    # The trough is remembered after a partial recovery.
    dd = max_drawdown([-2000.0, 1000.0])
    assert dd == pytest.approx(0.2)


def test_trend_following_reports_how_many_names_it_evaluated():
    panel = PricePanel(
        {"UP": {DATES[0]: 100.0, DATES[1]: 110.0, DATES[2]: 120.0, DATES[3]: 132.0},
         "DOWN": {DATES[0]: 120.0, DATES[1]: 110.0, DATES[2]: 100.0, DATES[3]: 90.0}},
        PointInTimeUniverse([Listing("UP", DATES[0]), Listing("DOWN", DATES[0])]),
    )
    r = trend_following(panel, _honest_builder, DATES, ma_months=3, cost_bp=0.0)
    assert r.mean_ranked == pytest.approx(2.0)
