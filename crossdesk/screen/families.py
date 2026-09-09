"""The cross-sectional family tests, on a monthly point-in-time panel.

Two of quantdesk's six pre-registered library families survive the move to a
monthly equity panel. The other four do not, and the reasons are worth stating
in the code rather than in a commit message, because "we ran the six families
on 300 names" would be false:

* **L1 overnight vs intraday** decomposes a session into close-to-open and
  open-to-close. A monthly panel has neither leg. It is also not a
  breadth-limited test -- it is one time series of session decompositions, and
  ranking 300 names does not add a single independent observation to it.
* **L2 turn of the month** and **L3 pre-holiday** are daily calendar effects on
  the index. Same two problems: the resolution is absent, and their power comes
  from the number of SESSIONS, which more names does not change.
* **L4 VIX percentile -> forward index return** is a market-level timing test.
  One series, no cross-section.
* **L5 sector momentum** is cross-sectional by construction. It is the family
  the power audit named at 12.7% power, and generalising three sectors long
  against three short to a decile spread over 300 names is exactly the breadth
  fix the audit argued for.
* **L6 trend following** is a per-name time-series rule pooled across names.
  Breadth helps it, though less directly: averaging the rule over 300 names
  cancels idiosyncratic noise and leaves the market factor, so the effective
  number of independent observations stays close to the number of MONTHS.

**THE AGGREGATION RULE, and it is load-bearing.** Both tests here produce one
observation per name per month, and a t-test over those pooled name-months is
wrong by a factor of roughly sqrt(names). Three hundred names in one month
share the market factor; they are not 300 independent draws. Every statistic in
this module is therefore computed on the MONTHLY PORTFOLIO return -- names
averaged first, then one observation per month -- which is the same correction
the VRP study applies for overlapping windows, arrived at for the same reason.
`tests/test_families.py` measures the inflation on a panel built so that only
the market factor exists: pooling reports a t-statistic several times the
honest one.

ASSUMPTION (microstructure): L5's registered form ranks on the return THROUGH
the formation date and starts holding at that same date, so one price ends the
ranking window and begins the holding window. Bid-ask bounce in that shared
price induces a spurious negative correlation between the two, which biases the
measured spread DOWNWARD -- against the hypothesis. Kept because it is the
registered form, and noted because it is conservative rather than flattering.
The standard remedy is to skip a month between ranking and holding; doing that
here would be a second trial on the same data and is deliberately not done.

Standard library only.
"""

from __future__ import annotations

import math
import statistics as st
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

from ..data.panel import PricePanel
from ..data.universe import Listing

#: Basis points per unit return.
BP = 10_000.0

Builder = Callable[[Sequence[Listing], date], Sequence[str]]


@dataclass
class SpreadResult:
    """A long-short family test, one observation per month.

    `gross_bp` and `net_bp` are parallel: quantdesk's rule is that a net figure
    never appears without its gross beside it, because the gap between them is
    the finding as often as the level is.
    """

    name: str
    months: list[date] = field(default_factory=list)
    long_bp: list[float] = field(default_factory=list)
    short_bp: list[float] = field(default_factory=list)
    gross_bp: list[float] = field(default_factory=list)
    #: The fraction of the position that PAYS a round trip this month. For a
    #: rebalanced spread that is the turnover of both legs; for the trend rule
    #: as registered it is the fraction of names held. Named for what it does
    #: rather than for one of the two things it means.
    cost_fraction: list[float] = field(default_factory=list)
    long_names: list[tuple[str, ...]] = field(default_factory=list)
    short_names: list[tuple[str, ...]] = field(default_factory=list)
    cost_bp: float = 0.0
    #: names in the universe that could not be scored (insufficient history)
    unscored: int = 0
    #: names dropped because the panel could price neither a trade nor an exit
    unpriced: int = 0
    provenance: dict[str, int] = field(default_factory=dict)
    #: mean number of eligible names ranked each month
    mean_ranked: float = 0.0

    def net_bp(self, cost_bp: float | None = None) -> list[float]:
        """Gross spread less the cost of the month's actual turnover.

        `cost_bp` is a ROUND TRIP per name. Turning over a fraction f of a leg
        sells f and buys f, which is f round trips, so the leg pays
        f * cost_bp and the spread pays the sum over both legs
        (`cost_fraction` already holds that sum).
        """
        c = self.cost_bp if cost_bp is None else cost_bp
        return [g - f * c for g, f in zip(self.gross_bp, self.cost_fraction)]

    @property
    def n(self) -> int:
        return len(self.gross_bp)


def _turnover(previous: frozenset[str], current: frozenset[str]) -> float:
    """Fraction of the leg replaced. 0.0 on the first month (nothing to sell)."""
    if not previous:
        return 0.0
    if not current:
        return 1.0
    return len(current - previous) / len(current)


def _leg_return(panel: PricePanel, names: Sequence[str], start: date, end: date,
                provenance: dict[str, int]) -> tuple[float | None, int]:
    """Equal-weight return of `names` over the period, in basis points.

    Delisted names are INCLUDED at their terminal return -- that is the entire
    reason the panel exists. A name the panel can price neither way is counted
    and excluded, and the count is reported rather than swallowed.
    """
    total = 0.0
    used = 0
    unpriced = 0
    for s in names:
        value, how = panel.period_return(s, start, end)
        provenance[how] = provenance.get(how, 0) + 1
        if value is None:
            unpriced += 1
            continue
        total += value
        used += 1
    if used == 0:
        return None, unpriced
    return BP * total / used, unpriced


def cross_sectional_momentum(
    panel: PricePanel,
    builder: Builder,
    dates: Sequence[date],
    *,
    n_per_side: int = 30,
    lookback_months: int = 12,
    cost_bp: float = 10.0,
    name: str = "L5 cross-sectional momentum",
) -> SpreadResult:
    """L5, generalised: top `n_per_side` by 12-month return against the bottom.

    Ranked on the total return from `dates[i - lookback_months]` to `dates[i]`
    and held from `dates[i]` to `dates[i + 1]`, monthly and non-overlapping, so
    consecutive observations share no data and need no overlap correction.

    A name is ranked only if its lookback return has provenance "traded". That
    excludes names whose window ended inside the lookback (a gap would compare
    two different total-return indices for the same PERMNO) and names too young
    to have the history. Requiring PAST history is causal; the leak the universe
    guards catch is requiring TOTAL history, which reaches into the future.
    """
    listings = panel.universe.listings
    result = SpreadResult(name=name, cost_bp=cost_bp)
    prev_long: frozenset[str] = frozenset()
    prev_short: frozenset[str] = frozenset()
    ranked_counts: list[int] = []

    for i in range(lookback_months, len(dates) - 1):
        back, now, nxt = dates[i - lookback_months], dates[i], dates[i + 1]
        members = builder(listings, now)

        scores: dict[str, float] = {}
        for s in members:
            value, how = panel.period_return(s, back, now)
            if value is not None and how == "traded":
                scores[s] = value
            else:
                result.unscored += 1
        if len(scores) < 2 * n_per_side:
            continue
        ranked_counts.append(len(scores))

        order = sorted(scores, key=lambda s: scores[s])
        shorts = tuple(order[:n_per_side])
        longs = tuple(order[-n_per_side:])

        long_bp, u1 = _leg_return(panel, longs, now, nxt, result.provenance)
        short_bp, u2 = _leg_return(panel, shorts, now, nxt, result.provenance)
        result.unpriced += u1 + u2
        if long_bp is None or short_bp is None:
            continue

        turn = (_turnover(prev_long, frozenset(longs))
                + _turnover(prev_short, frozenset(shorts)))
        prev_long, prev_short = frozenset(longs), frozenset(shorts)

        result.months.append(nxt)
        result.long_bp.append(long_bp)
        result.short_bp.append(short_bp)
        result.gross_bp.append(long_bp - short_bp)
        result.cost_fraction.append(turn)
        result.long_names.append(longs)
        result.short_names.append(shorts)

    result.mean_ranked = st.fmean(ranked_counts) if ranked_counts else 0.0
    return result


def trend_following(
    panel: PricePanel,
    builder: Builder,
    dates: Sequence[date],
    *,
    ma_months: int = 10,
    cost_bp: float = 10.0,
    name: str = "L6 trend following",
) -> SpreadResult:
    """L6, generalised: hold each name only above its `ma_months` average.

    The comparison is trend-filtered against buy-and-hold on the SAME names in
    the SAME months, so the market's drift is present in both legs and cancels
    in the difference. `long_bp` is the filtered leg and `short_bp` the
    buy-and-hold leg; the spread is what the filter added.

    The registered form charges a full round trip for every month a name is
    held rather than only when the position changes, which overcharges a rule
    that mostly stays put. That is left as registered and the cost sweep in the
    report shows what it costs: the zero-cost row is the gross result.

    A name needs `ma_months` consecutive month-end prices to be evaluated,
    which also guarantees the window lies inside one listing window -- a gap
    would put a restarted index alongside the old one.
    """
    listings = panel.universe.listings
    result = SpreadResult(name=name, cost_bp=cost_bp)
    evaluated: list[int] = []

    for i in range(ma_months - 1, len(dates) - 1):
        now, nxt = dates[i], dates[i + 1]
        members = builder(listings, now)

        filtered: list[float] = []
        held: list[float] = []
        n_held = 0
        for s in members:
            window = [panel.price(s, dates[j]) for j in range(i - ma_months + 1, i + 1)]
            if any(p is None or p <= 0 for p in window):
                result.unscored += 1
                continue
            value, how = panel.period_return(s, now, nxt)
            result.provenance[how] = result.provenance.get(how, 0) + 1
            if value is None:
                result.unpriced += 1
                continue
            r_bp = BP * value
            held.append(r_bp)
            above = window[-1] > st.fmean(window)
            filtered.append(r_bp if above else 0.0)
            n_held += 1 if above else 0
        if not held:
            continue
        evaluated.append(len(held))

        result.months.append(nxt)
        result.long_bp.append(st.fmean(filtered))
        result.short_bp.append(st.fmean(held))
        result.gross_bp.append(st.fmean(filtered) - st.fmean(held))
        # The registered cost: one round trip per name-month actually held.
        result.cost_fraction.append(n_held / len(held) if held else 0.0)

    result.mean_ranked = st.fmean(evaluated) if evaluated else 0.0
    return result


# ---------------------------------------------------------------------------
# statistics on a monthly series
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonthlyStats:
    """What a monthly long-short series supports, and nothing more."""

    n: int
    mean_bp: float
    sd_bp: float
    se_bp: float
    t: float
    lo_bp: float
    hi_bp: float
    annual_sharpe: float
    per_period_sharpe: float

    @property
    def annual_return_pct(self) -> float:
        """Arithmetic, 12 x the monthly mean. Not compounded."""
        return 12.0 * self.mean_bp / 100.0


def monthly_stats(xs: Sequence[float]) -> MonthlyStats | None:
    """Mean, error bar and Sharpe for one observation per month.

    No overlap correction, because monthly non-overlapping observations need
    none -- and saying so explicitly is the point, since every other statistic
    in this project does need one.
    """
    n = len(xs)
    if n < 3:
        return None
    mean = st.fmean(xs)
    sd = st.stdev(xs)
    if sd == 0:
        return None
    se = sd / math.sqrt(n)
    per_period = mean / sd
    return MonthlyStats(
        n=n, mean_bp=mean, sd_bp=sd, se_bp=se, t=mean / se,
        lo_bp=mean - 1.96 * se, hi_bp=mean + 1.96 * se,
        annual_sharpe=per_period * math.sqrt(12.0),
        per_period_sharpe=per_period,
    )


def max_drawdown(returns_bp: Sequence[float]) -> float:
    """Deepest peak-to-trough fall of the compounded series, as a fraction.

    Returned POSITIVE: 0.35 means the series lost 35% from a high. Compounded
    rather than summed, because a drawdown is a path and arithmetic sums
    understate how deep one gets.
    """
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in returns_bp:
        equity *= 1.0 + r / BP
        peak = max(peak, equity)
        worst = max(worst, 1.0 - equity / peak)
    return worst


def pooled_t(name_months: Sequence[float]) -> float:
    """The WRONG statistic, provided so the right one can be measured against it.

    A t-test over pooled name-months treats co-moving names as independent
    draws. Named to be uncomfortable, the way `survivors_only` is.
    """
    n = len(name_months)
    if n < 3:
        return float("nan")
    sd = st.stdev(name_months)
    if sd == 0:
        return float("nan")
    return st.fmean(name_months) / (sd / math.sqrt(n))
