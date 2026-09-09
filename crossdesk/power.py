"""Can this design detect the effect it is looking for?

The prerequisite question, and the cross-sectional analogue of the coin-flip
control in quantdesk: that control proves the instrument can detect the ABSENCE
of edge. This module asks the other half -- whether the instrument could detect
the PRESENCE of an edge of the size the literature actually claims.

A screen that demands a Sharpe of 0.71 to clear its own significance threshold
cannot find a strategy whose published Sharpe is 0.40. It will return "null",
and the null will be a fact about the test rather than about the market. That
is not a small distinction: it is the difference between "this does not work"
and "I cannot tell", and only one of them should stop a research programme.

Standard library only -- `statistics.NormalDist` supplies the normal CDF and
its inverse, which is the only special function needed here.

The Fundamental Law of Active Management (Grinold) supplies the link between
breadth and detectability:

    IR  =  IC * sqrt(BR)

where BR is the number of independent bets per year: names * rebalances. It is
an approximation -- it assumes bets are independent, which cross-sectional
equity bets emphatically are not -- so `breadth_haircut` exists and defaults to
something pessimistic. Treat the output as an upper bound on detectability, and
therefore a LOWER bound on the universe size you need.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

_N = NormalDist()


def bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """Two-sided critical |t| after Bonferroni correction for `n_tests`.

    >>> round(bonferroni_threshold(6), 2)   # the six-family library screen
    2.64
    >>> round(bonferroni_threshold(1), 2)
    1.96
    """
    if n_tests < 1:
        raise ValueError(f"n_tests must be >= 1, got {n_tests}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    return _N.inv_cdf(1 - alpha / (2 * n_tests))


def min_detectable_sharpe(years: float, t_threshold: float) -> float:
    """The smallest annual Sharpe whose expected t-stat reaches the threshold.

    Uses t = SR * sqrt(years), the standard large-sample result for the
    t-statistic of a mean return divided by its standard error.

    >>> round(min_detectable_sharpe(14.0, 2.64), 3)
    0.706
    """
    if years <= 0:
        raise ValueError(f"years must be positive, got {years}")
    return t_threshold / math.sqrt(years)


def breadth(n_names: int, rebalances_per_year: float,
            haircut: float = 0.25) -> float:
    """Independent bets per year, haircut for cross-sectional correlation.

    `haircut` is the fraction of nominal breadth that survives. Sector ETFs
    move together; 15 sectors are nowhere near 15 independent bets. The default
    of 0.25 is deliberately pessimistic -- an optimistic breadth estimate is
    exactly how a screen convinces itself it had power that it did not.
    """
    if n_names < 1:
        raise ValueError(f"n_names must be >= 1, got {n_names}")
    if not 0 < haircut <= 1:
        raise ValueError(f"haircut must be in (0,1], got {haircut}")
    return n_names * rebalances_per_year * haircut


def sharpe_from_ic(ic: float, n_names: int, rebalances_per_year: float,
                   haircut: float = 0.25) -> float:
    """Annual information ratio implied by a per-bet information coefficient."""
    return ic * math.sqrt(breadth(n_names, rebalances_per_year, haircut))


def ic_required(t_threshold: float, years: float, n_names: int,
                rebalances_per_year: float, haircut: float = 0.25) -> float:
    """The per-bet IC a design needs before it can clear its own threshold."""
    sr = min_detectable_sharpe(years, t_threshold)
    return sr / math.sqrt(breadth(n_names, rebalances_per_year, haircut))


def power(true_sharpe: float, years: float, t_threshold: float) -> float:
    """P(the test rejects) when the effect is real and this strong.

    The t-statistic is approximately Normal(true_SR * sqrt(years), 1). Power is
    the mass of that distribution beyond +/- the threshold.

    >>> round(power(0.40, 14.0, 2.64), 3)   # literature-strength sector momentum
    0.114
    """
    ncp = true_sharpe * math.sqrt(years)
    return (1 - _N.cdf(t_threshold - ncp)) + _N.cdf(-t_threshold - ncp)


def years_required(true_sharpe: float, t_threshold: float,
                   target_power: float = 0.80) -> float:
    """Years of data needed to reach `target_power` against a given true Sharpe."""
    if true_sharpe <= 0:
        raise ValueError(f"true_sharpe must be positive, got {true_sharpe}")
    if not 0 < target_power < 1:
        raise ValueError(f"target_power must be in (0,1), got {target_power}")
    # Ignores the far tail, which contributes < 1e-6 at any useful power.
    z = _N.inv_cdf(target_power)
    return ((t_threshold + z) / true_sharpe) ** 2


def names_required(true_ic: float, years: float, t_threshold: float,
                   rebalances_per_year: float, haircut: float = 0.25,
                   target_power: float = 0.80) -> float:
    """Universe size needed to reach `target_power` at a given per-bet IC.

    This is the number that decides whether a cross-sectional programme is
    worth starting on the universe you have, or whether the first task is
    getting more names.
    """
    z = _N.inv_cdf(target_power)
    sr_needed = (t_threshold + z) / math.sqrt(years)
    br_needed = (sr_needed / true_ic) ** 2
    return br_needed / (rebalances_per_year * haircut)


@dataclass(frozen=True, slots=True)
class Design:
    """A screen's shape, and whether it can see what it is hunting."""

    name: str
    years: float
    n_names: int
    rebalances_per_year: float
    n_tests: int = 1
    haircut: float = 0.25
    alpha: float = 0.05

    @property
    def t_threshold(self) -> float:
        return bonferroni_threshold(self.n_tests, self.alpha)

    @property
    def min_sharpe(self) -> float:
        return min_detectable_sharpe(self.years, self.t_threshold)

    @property
    def breadth(self) -> float:
        return breadth(self.n_names, self.rebalances_per_year, self.haircut)

    def power_against(self, true_sharpe: float) -> float:
        return power(true_sharpe, self.years, self.t_threshold)

    def verdict(self, literature_sharpe: float) -> str:
        """PASS if the design could plausibly detect a literature-strength effect.

        The 0.80 convention is arbitrary but standard. Below 0.50 the screen is
        closer to a coin flip than a test, and a null from it carries almost no
        information -- report it as UNDERPOWERED, never as evidence of absence.
        """
        p = self.power_against(literature_sharpe)
        if p >= 0.80:
            return "ADEQUATE"
        if p >= 0.50:
            return "WEAK"
        return "UNDERPOWERED"
