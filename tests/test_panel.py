"""The price panel, and the size of the bias it prevents.

`test_survivorship_inflates_the_mean_by_a_known_amount` is the point of the
file. It builds a panel whose true answer is known by construction, runs it both
ways, and asserts the exact gap. That number is what makes "survivorship bias"
an engineering problem rather than a caveat in a footnote.
"""

from datetime import date, timedelta

import pytest

from crossdesk.data.panel import (
    PricePanel,
    assert_no_silent_dropout,
    equal_weight_long_short,
    momentum_scores,
)
from crossdesk.data.universe import (
    Listing,
    PointInTimeUniverse,
    SurvivorshipError,
)

D = date
START = D(2020, 1, 1)


def _days(n):
    return [START + timedelta(days=i) for i in range(n)]


# --------------------------------------------------------------------------
# period_return
# --------------------------------------------------------------------------

def _simple_panel():
    universe = PointInTimeUniverse([
        Listing("UP", D(2019, 1, 1)),
        Listing("DOWN", D(2019, 1, 1)),
        Listing("DEAD", D(2019, 1, 1), D(2020, 1, 3), reason="bankrupt"),
    ])
    prices = {
        "UP":   {D(2020, 1, 1): 100.0, D(2020, 1, 10): 110.0},
        "DOWN": {D(2020, 1, 1): 100.0, D(2020, 1, 10): 90.0},
        # DEAD has a price on the 1st and then stops existing.
        "DEAD": {D(2020, 1, 1): 100.0},
    }
    return PricePanel(prices, universe)


def test_traded_return_is_the_price_ratio():
    panel = _simple_panel()
    value, how = panel.period_return("UP", D(2020, 1, 1), D(2020, 1, 10))
    assert value == pytest.approx(0.10)
    assert how == "traded"


def test_delisted_name_returns_its_terminal_value_not_none():
    """The whole point: DEAD has no price on the 10th but still has a return."""
    panel = _simple_panel()
    value, how = panel.period_return("DEAD", D(2020, 1, 1), D(2020, 1, 10))
    assert value == -1.0
    assert how == "delisted"


def test_missing_start_price_is_missing_not_zero():
    panel = _simple_panel()
    value, how = panel.period_return("UP", D(2020, 1, 5), D(2020, 1, 10))
    assert value is None and how == "missing"


def test_period_returns_includes_the_delisted_member():
    panel = _simple_panel()
    out, holes = panel.period_returns(D(2020, 1, 1), D(2020, 1, 10))
    assert holes == []
    assert set(out) == {"UP", "DOWN", "DEAD"}
    assert out["DEAD"] == -1.0


def test_period_returns_takes_members_asof_start_not_end():
    """Membership on the START date. Using the end date is the selection leak."""
    universe = PointInTimeUniverse([
        Listing("OLD", D(2019, 1, 1), D(2020, 1, 5), reason="acquired"),
        Listing("NEW", D(2020, 1, 8)),
    ])
    prices = {
        "OLD": {D(2020, 1, 1): 100.0},
        "NEW": {D(2020, 1, 8): 50.0, D(2020, 1, 10): 55.0},
    }
    panel = PricePanel(prices, universe)
    out, _ = panel.period_returns(D(2020, 1, 1), D(2020, 1, 10))
    assert set(out) == {"OLD"}          # NEW had not listed on the 1st
    assert out["OLD"] == 0.0            # acquired -> settles at last price


def test_strict_mode_refuses_to_silently_drop_a_member():
    universe = PointInTimeUniverse([Listing("HOLE", D(2019, 1, 1))])
    panel = PricePanel({"HOLE": {D(2020, 1, 1): 100.0}}, universe)
    with pytest.raises(SurvivorshipError, match="produced no return"):
        panel.period_returns(D(2020, 1, 1), D(2020, 1, 10))


def test_non_strict_mode_reports_the_holes_instead_of_raising():
    universe = PointInTimeUniverse([Listing("HOLE", D(2019, 1, 1))])
    panel = PricePanel({"HOLE": {D(2020, 1, 1): 100.0}}, universe)
    out, holes = panel.period_returns(D(2020, 1, 1), D(2020, 1, 10), strict=False)
    assert out == {}
    assert [h.symbol for h in holes] == ["HOLE"]


def test_end_must_follow_start():
    panel = _simple_panel()
    with pytest.raises(ValueError, match="must follow"):
        panel.period_returns(D(2020, 1, 10), D(2020, 1, 1))


def test_provenance_counts_each_path():
    panel = _simple_panel()
    counts = panel.provenance(D(2020, 1, 1), D(2020, 1, 10))
    assert counts == {"traded": 2, "delisted": 1, "missing": 0}


# --------------------------------------------------------------------------
# The bias, measured
# --------------------------------------------------------------------------

def _bias_panel(n_survivors=8, n_doomed=2, survivor_return=0.10):
    """A panel whose correct answer is known by construction.

    Survivors each return exactly `survivor_return`. The doomed names go to
    zero. Nothing else happens, so any deviation from the arithmetic mean is the
    bias and nothing but the bias.
    """
    listings, prices = [], {}
    for i in range(n_survivors):
        s = f"LIVE{i}"
        listings.append(Listing(s, D(2019, 1, 1)))
        prices[s] = {D(2020, 1, 1): 100.0,
                     D(2020, 1, 10): 100.0 * (1 + survivor_return)}
    for i in range(n_doomed):
        s = f"DEAD{i}"
        listings.append(Listing(s, D(2019, 1, 1), D(2020, 1, 5),
                                reason="bankrupt"))
        prices[s] = {D(2020, 1, 1): 100.0}
    return PricePanel(prices, PointInTimeUniverse(listings))


def test_survivorship_inflates_the_mean_by_a_known_amount():
    """8 names at +10%, 2 at -100%. The honest mean is -12%; the biased one +10%.

    The gap is 22 percentage points in a single period, from deleting one fifth
    of the cross-section. This is why the panel raises instead of skipping.
    """
    panel = _bias_panel()
    start, end = D(2020, 1, 1), D(2020, 1, 10)

    honest, holes = panel.period_returns(start, end)
    assert holes == []
    assert len(honest) == 10
    honest_mean = sum(honest.values()) / len(honest)
    assert honest_mean == pytest.approx((8 * 0.10 + 2 * -1.0) / 10)
    assert honest_mean == pytest.approx(-0.12)

    # The biased run: keep only names still listed at the end of the sample.
    survivors = panel.universe.survivors_only(D(2026, 1, 1))
    biased = {s: r for s, r in honest.items() if s in survivors}
    assert len(biased) == 8
    biased_mean = sum(biased.values()) / len(biased)
    assert biased_mean == pytest.approx(0.10)

    assert biased_mean - honest_mean == pytest.approx(0.22)


@pytest.mark.parametrize("n_doomed, expected_gap", [
    (0, 0.00),
    (1, 0.11),
    (2, 0.22),
    (5, 0.55),
])
def test_bias_scales_with_the_failure_rate(n_doomed, expected_gap):
    """The bias is linear in how much of the cross-section is deleted.

    With a 10-name panel, survivors at +10% and the doomed at -100%, the gap
    works out to exactly 0.11 per deleted name:

        gap = 0.10 - [(10-d)(0.10) - d] / 10 = 0.11 * d

    Linear, and steep: losing one name in ten to an unpriced delisting moves the
    measured mean by 11 percentage points.
    """
    n_survivors = 10 - n_doomed
    panel = _bias_panel(n_survivors=n_survivors, n_doomed=n_doomed)
    honest, _ = panel.period_returns(D(2020, 1, 1), D(2020, 1, 10))
    honest_mean = sum(honest.values()) / len(honest)
    survivors = panel.universe.survivors_only(D(2026, 1, 1))
    kept = [r for s, r in honest.items() if s in survivors]
    biased_mean = sum(kept) / len(kept) if kept else 0.0
    assert biased_mean - honest_mean == pytest.approx(expected_gap, abs=1e-12)


def test_assert_no_silent_dropout_passes_a_complete_panel():
    panel = _bias_panel()
    assert_no_silent_dropout(panel, [D(2020, 1, 1), D(2020, 1, 10)])


def test_assert_no_silent_dropout_catches_a_hole():
    universe = PointInTimeUniverse([
        Listing("OK", D(2019, 1, 1)),
        Listing("HOLE", D(2019, 1, 1)),
    ])
    prices = {
        "OK": {D(2020, 1, 1): 100.0, D(2020, 1, 10): 105.0},
        "HOLE": {D(2020, 1, 1): 100.0},     # listed, alive, and simply absent
    }
    panel = PricePanel(prices, universe)
    with pytest.raises(SurvivorshipError, match="silent dropout"):
        assert_no_silent_dropout(panel, [D(2020, 1, 1), D(2020, 1, 10)])


# --------------------------------------------------------------------------
# Portfolio construction and scores
# --------------------------------------------------------------------------

def test_long_short_is_top_minus_bottom():
    scores = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}
    forward = {"A": 0.10, "B": 0.05, "C": -0.05, "D": -0.10}
    # quantile 0.25 of 4 names -> 1 long (A), 1 short (D)
    assert equal_weight_long_short(scores, forward, 0.25) == pytest.approx(0.20)


def test_long_short_needs_two_names():
    assert equal_weight_long_short({"A": 1.0}, {"A": 0.1}) == 0.0


def test_long_short_rejects_a_bad_quantile():
    with pytest.raises(ValueError, match="quantile"):
        equal_weight_long_short({"A": 1.0, "B": 2.0}, {"A": 0.1, "B": 0.2}, 0.9)


def test_long_short_ignores_names_without_a_forward_return():
    scores = {"A": 3.0, "B": 2.0, "C": 1.0, "GHOST": 99.0}
    forward = {"A": 0.10, "B": 0.05, "C": -0.05}
    assert equal_weight_long_short(scores, forward, 0.34) == pytest.approx(0.15)


def test_momentum_scores_use_only_prices_up_to_the_as_of_date():
    dates = _days(80)
    universe = PointInTimeUniverse([Listing("X", D(2019, 1, 1))])
    prices = {"X": {d: 100.0 + i for i, d in enumerate(dates)}}
    panel = PricePanel(prices, universe)
    asof = dates[60]
    scores = momentum_scores(panel, asof, lookback_days=20, skip_days=5)
    # Rebuild with the future deleted; the score must not move.
    truncated = PricePanel({"X": {d: p for d, p in prices["X"].items() if d <= asof}},
                           universe)
    assert scores == momentum_scores(truncated, asof, lookback_days=20, skip_days=5)


def test_momentum_scores_are_empty_without_enough_history():
    dates = _days(10)
    universe = PointInTimeUniverse([Listing("X", D(2019, 1, 1))])
    panel = PricePanel({"X": {d: 100.0 for d in dates}}, universe)
    assert momentum_scores(panel, dates[-1], lookback_days=60) == {}
