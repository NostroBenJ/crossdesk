"""Three published variants of cross-sectional momentum, on the same panel.

`screen/families.py` measured plain momentum on CRSP top-300: net +88 bp/mo,
t = +1.65, annual Sharpe 0.34, with kurtosis 10.4 and leg drawdowns of 61% and
86%. That is the momentum-crash signature, and each variant here is a
PUBLISHED response to one specific part of it. None of them is a parameter
search, and that distinction is the only thing standing between this file and
the overfitting the rest of the project is built to catch:

* **V1 residual momentum** (Blitz, Huij & Martens 2011) -- rank on the part of
  the return the market does not explain. The claim is that raw momentum's
  ranking is contaminated by time-varying factor exposure, so the winners are
  partly just high-beta names in a rising market.
* **V2 volatility-scaled momentum** (Barroso & Santa-Clara 2015) -- scale the
  whole spread by the inverse of the strategy's OWN recent realised
  volatility. The claim is that momentum's crashes are forecastable from its
  own vol even though its returns are not.
* **V3 inverse-volatility weighting** -- weight within each leg by 1/sigma
  instead of equally, so one high-vol name is not most of the tracking error.

THE ENGINE IS SHARED, AND THAT IS THE POINT. `spread()` below is a
generalisation of `families.cross_sectional_momentum` with the scoring rule and
the weighting rule lifted out as arguments. `tests/test_variants.py` asserts
that calling it with the raw-return score and equal weights reproduces the
registered L5 result EXACTLY -- same months, same names, same basis points. A
variant that cannot reproduce its own baseline is measuring two changes at
once, and the same discipline caught five silent divergences in the sibling
`cisd-bot` repo.

WEIGHTED TURNOVER REDUCES TO THE REGISTERED ONE. Cost is charged on

    f = sum_i max(0, w_new_i - w_old_i)

the fraction of the leg BOUGHT this month. At equal weights that is exactly
`len(new - old) / n`, which is what `families._turnover` computes, so the
generalisation does not quietly re-price the baseline. Pinned in the tests.

ASSUMPTION (weight drift): weights are compared at rebalance dates, so a held
position whose weight drifted with its return between rebalances is treated as
if it were still at its target. The registered baseline makes the same
simplification and it understates turnover for both.

Standard library only.
"""

from __future__ import annotations

import math
import random
import statistics as st
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from ..data.panel import PricePanel
from .families import BP, Builder, SpreadResult, monthly_stats

#: symbol, date-index -> score, or None when the name cannot be ranked
ScoreFn = Callable[[str, int], float | None]
#: names, date-index -> weights summing to 1.0
WeightFn = Callable[[Sequence[str], int], dict[str, float]]


# ---------------------------------------------------------------------------
# monthly return matrix -- computed once, causal by construction
# ---------------------------------------------------------------------------

def monthly_returns(panel: PricePanel, symbols: Sequence[str],
                    dates: Sequence[date]) -> dict[str, list[float | None]]:
    """symbol -> [None, r_1, r_2, ...] where r_i spans dates[i-1] -> dates[i].

    Only "traded" returns are kept. A delisting terminal return ENDS a series
    rather than extending it: the regressions and volatilities below need a
    contiguous run of ordinary monthly returns, and splicing a terminal return
    onto one would put two different regimes in the same window. The delisting
    return is still paid by the HOLDING leg -- that happens in `spread()` via
    `panel.period_return`, which is where it belongs.
    """
    out: dict[str, list[float | None]] = {}
    for s in symbols:
        row: list[float | None] = [None] * len(dates)
        for i in range(1, len(dates)):
            value, how = panel.period_return(s, dates[i - 1], dates[i])
            row[i] = value if how == "traded" else None
        out[s] = row
    return out


def market_series(rets: Mapping[str, Sequence[float | None]],
                  builder: Builder, listings, dates: Sequence[date],
                  ) -> list[float | None]:
    """Equal-weighted return of the month's members, as the market proxy.

    Equal-weighted rather than cap-weighted because both legs of every spread
    here are equal-weighted, so this is the benchmark whose beta the residual
    is actually being purged of. Membership is taken as of the START of the
    month, so nothing in the series is known before it happened.
    """
    out: list[float | None] = [None] * len(dates)
    for i in range(1, len(dates)):
        vals = []
        for s in builder(listings, dates[i - 1]):
            row = rets.get(s)
            if row is not None and row[i] is not None:
                vals.append(row[i])
        out[i] = st.fmean(vals) if len(vals) >= 2 else None
    return out


def ols(y: Sequence[float], x: Sequence[float]) -> tuple[float, float, list[float]]:
    """Univariate least squares. Returns (alpha, beta, residuals)."""
    mx, my = st.fmean(x), st.fmean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx <= 0:
        return my, 0.0, [yi - my for yi in y]
    beta = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sxx
    alpha = my - beta * mx
    return alpha, beta, [yi - alpha - beta * xi for xi, yi in zip(x, y)]


# ---------------------------------------------------------------------------
# scoring rules
# ---------------------------------------------------------------------------

def raw_momentum_score(panel: PricePanel, dates: Sequence[date],
                       lookback: int = 12) -> ScoreFn:
    """The registered L5 rule: total return over the previous `lookback` months.

    Requires provenance "traded" over the whole window, which is the baseline's
    own rule -- a window spanning a listing gap would compare two different
    total-return indices for one PERMNO.
    """
    def score(symbol: str, i: int) -> float | None:
        value, how = panel.period_return(symbol, dates[i - lookback], dates[i])
        return value if how == "traded" else None
    return score


def residual_momentum_score(rets: Mapping[str, Sequence[float | None]],
                            market: Sequence[float | None],
                            *, estimation: int = 36, formation: int = 12,
                            ) -> ScoreFn:
    """V1: mean residual over the last `formation` months, divided by its sd.

    Blitz, Huij & Martens (2011). Beta is estimated over `estimation` months
    ending at the ranking date, residuals are taken from the last `formation`
    of them, and the score is their t-statistic -- their standardisation, not
    an addition here. Everything reads returns at or before the ranking date.

    A name needs the full estimation window with no missing month. That is a
    stricter history requirement than raw momentum's and it changes the
    sample, which is why the report runs the baseline over the same months
    rather than comparing against the number from the earlier screen.
    """
    def score(symbol: str, i: int) -> float | None:
        row = rets.get(symbol)
        if row is None or i - estimation + 1 < 0:
            return None
        window = row[i - estimation + 1: i + 1]
        mkt = market[i - estimation + 1: i + 1]
        if any(v is None for v in window) or any(v is None for v in mkt):
            return None
        _a, _b, resid = ols(window, mkt)
        tail = resid[-formation:]
        sd = st.stdev(tail)
        return st.fmean(tail) / sd if sd > 0 else None
    return score


# ---------------------------------------------------------------------------
# weighting rules
# ---------------------------------------------------------------------------

def equal_weights(names: Sequence[str], _i: int) -> dict[str, float]:
    w = 1.0 / len(names)
    return {s: w for s in names}


def inverse_vol_weights(rets: Mapping[str, Sequence[float | None]],
                        *, lookback: int = 36) -> WeightFn:
    """V3: w_i proportional to 1/sigma_i over the previous `lookback` months.

    A name with no usable volatility falls back to the average weight rather
    than being dropped: the ranking already selected it, and dropping it here
    would silently change the portfolio the OTHER variants hold.
    """
    def weights(names: Sequence[str], i: int) -> dict[str, float]:
        raw: dict[str, float] = {}
        for s in names:
            row = rets.get(s)
            if row is None:
                window: list[float] = []
            else:
                window = [v for v in row[max(0, i - lookback + 1): i + 1]
                          if v is not None]
            sd = st.stdev(window) if len(window) >= 12 else 0.0
            raw[s] = 1.0 / sd if sd > 0 else 0.0
        usable = [v for v in raw.values() if v > 0]
        if not usable:
            return equal_weights(names, i)
        fallback = st.fmean(usable)
        raw = {s: (v if v > 0 else fallback) for s, v in raw.items()}
        total = sum(raw.values())
        return {s: v / total for s, v in raw.items()}
    return weights


# ---------------------------------------------------------------------------
# the shared engine
# ---------------------------------------------------------------------------

def _weighted_leg(panel: PricePanel, weights: Mapping[str, float],
                  start: date, end: date, provenance: dict[str, int],
                  ) -> tuple[float | None, int]:
    """Weighted return of a leg in basis points, renormalised over what priced.

    Delisted names are INCLUDED at their terminal return. A name the panel can
    price neither way is counted and its weight redistributed proportionally,
    which is what a portfolio that could not hold it would have done.
    """
    total = 0.0
    used = 0.0
    unpriced = 0
    for s, w in weights.items():
        value, how = panel.period_return(s, start, end)
        provenance[how] = provenance.get(how, 0) + 1
        if value is None:
            unpriced += 1
            continue
        total += w * value
        used += w
    if used <= 0:
        return None, unpriced
    return BP * total / used, unpriced


def _bought(previous: Mapping[str, float], current: Mapping[str, float]) -> float:
    """Fraction of the leg BOUGHT: sum of positive weight changes.

    At equal weights this is exactly `len(current - previous) / len(current)`,
    the registered turnover measure. Pinned in `tests/test_variants.py`.
    """
    if not previous:
        return 0.0
    return sum(max(0.0, w - previous.get(s, 0.0)) for s, w in current.items())


def spread(panel: PricePanel, builder: Builder, dates: Sequence[date], *,
           score_fn: ScoreFn, weight_fn: WeightFn = equal_weights,
           n_per_side: int = 30, cost_bp: float = 10.0, start: int = 12,
           name: str = "spread") -> SpreadResult:
    """Long the top `n_per_side` by score, short the bottom, held one month.

    `families.cross_sectional_momentum` with the score and the weights lifted
    out. `start` is the first date index that can be ranked; it is an argument
    rather than derived from the lookback because the variants have different
    history requirements and the comparison has to run them over the SAME
    months.
    """
    listings = panel.universe.listings
    result = SpreadResult(name=name, cost_bp=cost_bp)
    prev_long: dict[str, float] = {}
    prev_short: dict[str, float] = {}
    ranked_counts: list[int] = []

    for i in range(start, len(dates) - 1):
        now, nxt = dates[i], dates[i + 1]
        scores: dict[str, float] = {}
        for s in builder(listings, now):
            value = score_fn(s, i)
            if value is None:
                result.unscored += 1
            else:
                scores[s] = value
        if len(scores) < 2 * n_per_side:
            continue
        ranked_counts.append(len(scores))

        order = sorted(scores, key=lambda s: scores[s])
        shorts = tuple(order[:n_per_side])
        longs = tuple(order[-n_per_side:])
        w_long = weight_fn(longs, i)
        w_short = weight_fn(shorts, i)

        long_bp, u1 = _weighted_leg(panel, w_long, now, nxt, result.provenance)
        short_bp, u2 = _weighted_leg(panel, w_short, now, nxt, result.provenance)
        result.unpriced += u1 + u2
        if long_bp is None or short_bp is None:
            continue

        turn = _bought(prev_long, w_long) + _bought(prev_short, w_short)
        prev_long, prev_short = dict(w_long), dict(w_short)

        result.months.append(nxt)
        result.long_bp.append(long_bp)
        result.short_bp.append(short_bp)
        result.gross_bp.append(long_bp - short_bp)
        result.cost_fraction.append(turn)
        result.long_names.append(longs)
        result.short_names.append(shorts)

    result.mean_ranked = st.fmean(ranked_counts) if ranked_counts else 0.0
    return result


# ---------------------------------------------------------------------------
# V2 -- volatility scaling, applied to a finished spread
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaledSpread:
    """A spread rescaled month by month, and the scale path that produced it."""

    result: SpreadResult
    scales: list[float]
    capped_months: int
    mean_scale: float
    max_scale: float
    levered_months: int


def volatility_scaled(res: SpreadResult, *, target_annual_vol: float = 0.12,
                      lookback: int = 12, cap: float = 2.0,
                      name: str = "V2 volatility-scaled momentum") -> ScaledSpread:
    """V2: hold `scale_t = min(cap, target / sigma_t)` units of the spread.

    Barroso & Santa-Clara (2015). `sigma_t` is the standard deviation of the
    GROSS spread over the previous `lookback` months, strictly before t, so the
    scale is knowable on the rebalance date. The first `lookback` months have
    no estimate and are dropped rather than held unscaled -- holding them
    unscaled would mix two strategies in one series.

    ASSUMPTION (resolution): the paper estimates sigma from 126 DAILY returns.
    A monthly panel has none, so this uses 12 monthly observations. That is a
    slower estimate and it will react to a volatility spike a month or two
    later than the published version, which biases the variant DOWNWARD
    against its own claim.

    Costs scale with the position: a half-sized book turns over half as much
    notional, so `cost_fraction` is scaled by the same factor.

    `cap` bounds the leverage. Above 1.0 the strategy is borrowing, which the
    account this feeds cannot do; the report prints how many months exceeded
    1.0 so the constraint is visible rather than assumed away.
    """
    target_monthly_bp = BP * target_annual_vol / math.sqrt(12.0)
    out = SpreadResult(name=name, cost_bp=res.cost_bp,
                       unscored=res.unscored, unpriced=res.unpriced,
                       provenance=dict(res.provenance),
                       mean_ranked=res.mean_ranked)
    scales: list[float] = []
    capped = 0
    levered = 0

    for i in range(lookback, res.n):
        window = res.gross_bp[i - lookback:i]
        sd = st.stdev(window)
        if sd <= 0:
            continue
        scale = target_monthly_bp / sd
        if scale > cap:
            scale = cap
            capped += 1
        if scale > 1.0:
            levered += 1
        scales.append(scale)
        out.months.append(res.months[i])
        out.long_bp.append(scale * res.long_bp[i])
        out.short_bp.append(scale * res.short_bp[i])
        out.gross_bp.append(scale * res.gross_bp[i])
        out.cost_fraction.append(scale * res.cost_fraction[i])
        out.long_names.append(res.long_names[i])
        out.short_names.append(res.short_names[i])

    return ScaledSpread(result=out, scales=scales, capped_months=capped,
                        mean_scale=st.fmean(scales) if scales else float("nan"),
                        max_scale=max(scales) if scales else float("nan"),
                        levered_months=levered)


# ---------------------------------------------------------------------------
# comparing two correlated series
# ---------------------------------------------------------------------------

def align(a: SpreadResult, b: SpreadResult, cost_bp: float,
          ) -> tuple[list[float], list[float], list[date]]:
    """Net series of both, restricted to the months they share.

    Two variants with different history requirements produce different months.
    Comparing their headline numbers across different samples would attribute
    the sample to the variant, so every head-to-head statistic runs here first.
    """
    ma = dict(zip(a.months, a.net_bp(cost_bp)))
    mb = dict(zip(b.months, b.net_bp(cost_bp)))
    common = sorted(set(ma) & set(mb))
    return [ma[d] for d in common], [mb[d] for d in common], common


def sharpe_difference_z(a: Sequence[float], b: Sequence[float]) -> float:
    """Jobson-Korkie test of SR(a) - SR(b), with Memmel's (2003) correction.

    The two series are the SAME strategy modified, so they are strongly
    correlated and an unpaired comparison of their Sharpe ratios would have a
    standard error several times too large. This is the paired instrument, and
    `tests/test_variants.py` checks it against a paired bootstrap rather than
    trusting the algebra.
    """
    n = len(a)
    if n < 4 or len(b) != n:
        return float("nan")
    sa, sb = st.stdev(a), st.stdev(b)
    if sa <= 0 or sb <= 0:
        return float("nan")
    s1, s2 = st.fmean(a) / sa, st.fmean(b) / sb
    rho = st.correlation(a, b)
    var = (2.0 - 2.0 * rho
           + 0.5 * (s1 * s1 + s2 * s2 - 2.0 * rho * rho * s1 * s2)) / n
    if var > 0:
        return (s1 - s2) / math.sqrt(var)
    # Perfectly correlated series drive the variance to zero. When they are the
    # same strategy the difference is zero too, and 0/0 is 0 rather than
    # undefined -- a variant that changed nothing did not improve anything.
    # Any other numerator at zero variance is genuinely undefined and says so.
    return 0.0 if s1 == s2 else float("nan")


def bootstrap_sharpe_difference(a: Sequence[float], b: Sequence[float], *,
                                draws: int = 20_000, seed: int = 20260909,
                                ) -> tuple[float, float]:
    """Paired bootstrap of SR(a) - SR(b). Returns (observed difference, p).

    Months are resampled in PAIRS so the correlation between the two series is
    preserved. Provided as an independent check on the closed form above, in
    the spirit of the finite-difference rule in the sibling options project: an
    analytic standard error nobody has tested numerically is an assertion.

    Both series are centred before resampling, so the resampled world has no
    difference in mean and the p-value is the fraction of null draws at least
    as extreme as the observed one.
    """
    n = len(a)
    if n < 4 or len(b) != n:
        return float("nan"), float("nan")

    def sharpe(xs: Sequence[float]) -> float:
        sd = st.stdev(xs)
        return st.fmean(xs) / sd if sd > 0 else float("nan")

    observed = sharpe(a) - sharpe(b)
    rng = random.Random(seed)
    ma, mb = st.fmean(a), st.fmean(b)
    ca = [x - ma for x in a]
    cb = [x - mb for x in b]
    hits = 0
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sharpe([ca[i] for i in idx])
        db = sharpe([cb[i] for i in idx])
        if not math.isnan(da) and not math.isnan(db) and abs(da - db) >= abs(observed):
            hits += 1
    return observed, (hits + 1) / (draws + 1)


def summarise(xs: Sequence[float]) -> str:
    s = monthly_stats(xs)
    if s is None:
        return "too few months"
    return (f"n={s.n:>3}  {s.mean_bp:>+8.2f} bp/mo  t={s.t:>+6.2f}  "
            f"ann SR {s.annual_sharpe:>+5.2f}")
