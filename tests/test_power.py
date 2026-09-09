"""Power arithmetic, checked against simulation rather than asserted.

`test_power_matches_monte_carlo` is the one that earns the module's trust. The
closed form says a design has probability p of rejecting; the simulation
generates return series with a known true Sharpe, runs the actual t-test, and
counts. If those disagree, the closed form is wrong -- and every "underpowered"
verdict built on it would be wrong in the same direction.
"""

import math
import random
import statistics

import pytest

from crossdesk.power import (
    Design,
    bonferroni_threshold,
    breadth,
    ic_required,
    min_detectable_sharpe,
    names_required,
    power,
    sharpe_from_ic,
    years_required,
)

TRADING_DAYS = 252


def _simulate_t(true_sharpe: float, years: float, rng: random.Random) -> float:
    """One realisation of the t-statistic for a strategy with a known Sharpe."""
    n = int(round(years * TRADING_DAYS))
    mu = true_sharpe / math.sqrt(TRADING_DAYS)   # daily mean, unit daily sd
    xs = [rng.gauss(mu, 1.0) for _ in range(n)]
    return math.sqrt(n) * statistics.fmean(xs) / statistics.stdev(xs)


@pytest.mark.parametrize("true_sharpe, years", [
    (0.00, 14.0),   # no effect -> power collapses to the false-positive rate
    (0.40, 14.0),   # literature-strength sector momentum
    (0.70, 14.0),   # roughly the threshold itself
    (1.00, 10.0),
])
def test_power_matches_monte_carlo(true_sharpe, years):
    rng = random.Random(20260908)
    t_threshold = bonferroni_threshold(6)
    trials = 4000
    hits = sum(abs(_simulate_t(true_sharpe, years, rng)) > t_threshold
               for _ in range(trials))
    empirical = hits / trials
    analytic = power(true_sharpe, years, t_threshold)
    # 4000 trials -> se <= 0.008; 0.03 is ~4 se and still tight enough to catch
    # a wrong formula, which would be off by far more than this.
    assert abs(empirical - analytic) < 0.03, (
        f"analytic {analytic:.3f} vs simulated {empirical:.3f}"
    )


def test_power_at_zero_effect_is_the_alpha_level():
    """With no effect, rejection probability must equal the corrected alpha."""
    t = bonferroni_threshold(6)
    assert power(0.0, 14.0, t) == pytest.approx(0.05 / 6, abs=1e-12)


def test_bonferroni_reproduces_the_screens_threshold():
    """The library screen used |t| > 2.64 for six families. Confirm that number."""
    assert bonferroni_threshold(6) == pytest.approx(2.64, abs=0.005)
    assert bonferroni_threshold(1) == pytest.approx(1.96, abs=0.005)


def test_bonferroni_is_monotone_in_test_count():
    ts = [bonferroni_threshold(k) for k in (1, 2, 6, 20, 158)]
    assert ts == sorted(ts)


def test_min_detectable_sharpe_inverts_the_t_formula():
    """Identity: a strategy at exactly the minimum has expected t = threshold."""
    for years in (5.0, 14.0, 20.0):
        for t in (1.96, 2.64):
            sr = min_detectable_sharpe(years, t)
            assert sr * math.sqrt(years) == pytest.approx(t, rel=1e-12)


def test_power_is_one_half_at_the_minimum_detectable_sharpe():
    """At the minimum, the t-stat is centred on the threshold -- a coin flip.

    This is the fact that makes 'minimum detectable' a misleading name: a design
    sitting exactly there fails half the time on a real effect.
    """
    t = bonferroni_threshold(6)
    sr = min_detectable_sharpe(14.0, t)
    assert power(sr, 14.0, t) == pytest.approx(0.50, abs=0.001)


def test_power_is_monotone_in_effect_and_in_years():
    t = bonferroni_threshold(6)
    assert power(0.2, 14.0, t) < power(0.4, 14.0, t) < power(0.8, 14.0, t)
    assert power(0.4, 5.0, t) < power(0.4, 14.0, t) < power(0.4, 30.0, t)


def test_years_required_inverts_power():
    t = bonferroni_threshold(6)
    for sr in (0.3, 0.5, 0.9):
        y = years_required(sr, t, target_power=0.80)
        assert power(sr, y, t) == pytest.approx(0.80, abs=0.002)


def test_names_required_reaches_target_power():
    t = bonferroni_threshold(6)
    n = names_required(true_ic=0.03, years=14.0, t_threshold=t,
                       rebalances_per_year=12, haircut=0.25)
    sr = sharpe_from_ic(0.03, int(round(n)), 12, 0.25)
    assert power(sr, 14.0, t) == pytest.approx(0.80, abs=0.02)


def test_breadth_haircut_reduces_breadth():
    assert breadth(100, 12, haircut=1.0) == 1200
    assert breadth(100, 12, haircut=0.25) == 300


def test_ic_required_falls_as_the_universe_grows():
    """The whole argument for breadth: more names, less skill needed per bet."""
    t = bonferroni_threshold(6)
    small = ic_required(t, 14.0, n_names=15, rebalances_per_year=12)
    large = ic_required(t, 14.0, n_names=500, rebalances_per_year=12)
    assert large < small
    # IC scales as 1/sqrt(N): 500/15 names -> sqrt(33.3) = 5.77x less skill.
    assert small / large == pytest.approx(math.sqrt(500 / 15), rel=1e-9)


def test_verdict_thresholds():
    weak = Design("15 ETFs", years=14.0, n_names=15, rebalances_per_year=12,
                  n_tests=6)
    assert weak.verdict(0.40) == "UNDERPOWERED"
    strong = Design("long sample", years=60.0, n_names=15,
                    rebalances_per_year=12, n_tests=1)
    assert strong.verdict(0.40) == "ADEQUATE"


@pytest.mark.parametrize("bad", [
    dict(n_tests=0), dict(n_tests=-1),
])
def test_bonferroni_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        bonferroni_threshold(**bad)


def test_rejects_nonpositive_years_and_bad_haircut():
    with pytest.raises(ValueError):
        min_detectable_sharpe(0.0, 2.64)
    with pytest.raises(ValueError):
        breadth(10, 12, haircut=0.0)
    with pytest.raises(ValueError):
        breadth(0, 12)
