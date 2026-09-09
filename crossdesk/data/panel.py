"""A cross-sectional price panel whose returns survive delisting.

The panel is the join between prices and the universe, and it is where the
delisting leak turns into a number. The rule it enforces has one sentence:

    Every symbol that was a member on `start` must produce a return for
    `start -> end`, including the ones that stopped trading in between.

A panel that instead returns "no data" for those symbols has silently deleted
the worst outcomes in the cross-section. Nothing downstream can recover them,
and every ranking, every decile spread, and every backtest built on top inherits
the bias with no trace of where it came from.

`period_returns` therefore raises rather than skipping. If a member has no price
and no delisting record, that is a hole in the data and the research stops until
someone decides what it means -- which is the correct outcome, because the
alternative is a silent default that flatters the result.

Standard library only.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from .universe import PointInTimeUniverse, SurvivorshipError


@dataclass(frozen=True, slots=True)
class MissingPrice:
    """A member with no usable price and no delisting record."""

    symbol: str
    on: date
    detail: str


class PricePanel:
    """Prices indexed by symbol and date, joined to a point-in-time universe.

    Prices are RAW, matching `quantdesk.data.etf_daily`: opens and closes are
    both unadjusted, so a dividend is absent from both legs rather than present
    in one. That makes period returns internally consistent and slightly
    understated, which is the safe direction.

    ASSUMPTION (corporate actions): splits are NOT handled here. A raw series
    across a 2-for-1 split books a -50% return that never happened. Any universe
    wider than the split-free ETF set needs a split-adjusted source before this
    class is trustworthy -- see `docs` in the module header of `universe.py`.
    """

    def __init__(self, prices: Mapping[str, Mapping[date, float]],
                 universe: PointInTimeUniverse) -> None:
        self._prices = {s: dict(series) for s, series in prices.items()}
        self._universe = universe
        all_dates: set[date] = set()
        for series in self._prices.values():
            all_dates.update(series)
        self._dates = tuple(sorted(all_dates))
        # Ended windows per symbol, so `period_return` does not scan every
        # listing in the universe on every call -- on a CRSP-sized universe
        # that scan is tens of thousands of listings per return.
        self._ended: dict[str, list[tuple[date, float]]] = {}
        for lst in universe.listings:
            if lst.end is not None:
                assert lst.delisting_return is not None  # enforced at construction
                self._ended.setdefault(lst.symbol, []).append((lst.end, lst.delisting_return))
        self._sorted_dates: dict[str, list[date]] = {}

    @property
    def dates(self) -> tuple[date, ...]:
        return self._dates

    @property
    def universe(self) -> PointInTimeUniverse:
        return self._universe

    def price(self, symbol: str, on: date) -> float | None:
        return self._prices.get(symbol, {}).get(on)

    def period_return(self, symbol: str, start: date, end: date
                      ) -> tuple[float | None, str]:
        """Return over `start -> end` for one symbol, and how it was obtained.

        The second element is the provenance: "traded" for a normal two-price
        return, "delisted" when the terminal return was used, "missing" when
        neither was available. Provenance is returned rather than logged because
        the CALLER has to decide what a hole means, and a count of how many
        returns came from each path is a data-quality statistic worth reporting
        beside any result.
        """
        p0 = self.price(symbol, start)
        if p0 is None or p0 <= 0:
            return None, "missing"

        # Did the listing end at or before `end`? Then the position was closed
        # by the delisting, not by a price on `end`. The exit is the LAST
        # observed price on or before the end date, and the terminal return is
        # applied on top of it:
        #
        #     r = p_exit / p0 * (1 + delisting_return) - 1
        #
        # so an acquisition (terminal 0.0) pays the price path to the takeover
        # and a bankruptcy (terminal -1.0) is a total loss whatever the path.
        # With no price after `start` the exit is `p0` itself and this reduces
        # to the bare terminal return, which is what the fixture panels pin.
        for end_date, terminal in self._ended.get(symbol, ()):
            if start <= end_date < end:
                p_exit = self._last_price_on_or_before(symbol, end_date)
                if p_exit is None or p_exit <= 0:
                    p_exit = p0
                return (p_exit / p0) * (1.0 + terminal) - 1.0, "delisted"

        p1 = self.price(symbol, end)
        if p1 is None or p1 <= 0:
            return None, "missing"
        return p1 / p0 - 1.0, "traded"

    def _last_price_on_or_before(self, symbol: str, on: date) -> float | None:
        dates = self._sorted_dates.get(symbol)
        if dates is None:
            dates = sorted(self._prices.get(symbol, {}))
            self._sorted_dates[symbol] = dates
        i = bisect_right(dates, on)
        if i == 0:
            return None
        return self._prices[symbol][dates[i - 1]]

    def period_returns(self, start: date, end: date, *,
                       strict: bool = True
                       ) -> tuple[dict[str, float], list[MissingPrice]]:
        """Returns for every universe member on `start`, over `start -> end`.

        Members are taken as of `start`, never as of `end` -- taking them as of
        `end` is the selection leak in one character of difference.

        With `strict=True` (the default) a member that produces neither a traded
        return nor a delisting return raises. Set it False only in a data-quality
        report that is explicitly counting the holes.
        """
        if end <= start:
            raise ValueError(f"end {end} must follow start {start}")

        out: dict[str, float] = {}
        holes: list[MissingPrice] = []
        for symbol in sorted(self._universe.members_asof(start)):
            value, how = self.period_return(symbol, start, end)
            if value is None:
                holes.append(MissingPrice(symbol, start,
                                          f"no price on {start}/{end} and no delisting"))
                continue
            out[symbol] = value
            del how

        if strict and holes:
            shown = ", ".join(f"{h.symbol}@{h.on}" for h in holes[:8])
            raise SurvivorshipError(
                f"{len(holes)} member(s) produced no return for {start} -> {end}: "
                f"{shown}{' ...' if len(holes) > 8 else ''}. "
                "Dropping them would delete part of the cross-section; supply a "
                "price or a delisting return instead."
            )
        return out, holes

    def provenance(self, start: date, end: date) -> dict[str, int]:
        """Count returns by how they were obtained. A data-quality statistic.

        Report this beside any cross-sectional result. A period where 5% of the
        cross-section came from `delisted` is a period whose ranking is driven by
        an assumption, not by prices.
        """
        counts = {"traded": 0, "delisted": 0, "missing": 0}
        for symbol in self._universe.members_asof(start):
            _, how = self.period_return(symbol, start, end)
            counts[how] += 1
        return counts


def assert_no_silent_dropout(panel: PricePanel,
                             rebalance_dates: Sequence[date]) -> None:
    """Every member on every rebalance date must produce a return.

    The panel-level companion to `assert_no_survivorship`. That one guards which
    symbols entered the cross-section; this one guards that none of them left it
    without being priced.
    """
    dates = list(rebalance_dates)
    problems: list[str] = []
    for start, end in zip(dates, dates[1:]):
        _, holes = panel.period_returns(start, end, strict=False)
        if holes:
            names = sorted(h.symbol for h in holes)[:6]
            problems.append(f"  {start} -> {end}: {len(holes)} missing {names}")
    if problems:
        raise SurvivorshipError(
            f"silent dropout on {len(problems)} period(s):\n"
            + "\n".join(problems[:20])
        )


def equal_weight_long_short(scores: Mapping[str, float],
                            forward: Mapping[str, float],
                            quantile: float = 0.2) -> float:
    """Return of an equal-weight long-top / short-bottom portfolio.

    A deliberately plain construction. It exists so the survivorship experiment
    measures the effect of the UNIVERSE, with the portfolio held constant and
    simple enough that it cannot contribute an artifact of its own.
    """
    if not 0 < quantile <= 0.5:
        raise ValueError(f"quantile must be in (0, 0.5], got {quantile}")
    common = [s for s in scores if s in forward]
    if len(common) < 2:
        return 0.0
    ranked = sorted(common, key=lambda s: scores[s])
    k = max(1, int(len(ranked) * quantile))
    shorts, longs = ranked[:k], ranked[-k:]
    long_leg = sum(forward[s] for s in longs) / len(longs)
    short_leg = sum(forward[s] for s in shorts) / len(shorts)
    return long_leg - short_leg


def momentum_scores(panel: PricePanel, on: date, lookback_days: int,
                    skip_days: int = 21) -> dict[str, float]:
    """Cross-sectional momentum: return over the lookback, skipping recent days.

    The skip is standard and not decorative -- the most recent month reverses, so
    including it mixes two effects with opposite signs. Scores are computed from
    prices at or before `on` only.
    """
    members = panel.universe.members_asof(on)
    dates = [d for d in panel.dates if d <= on]
    if len(dates) < lookback_days + skip_days + 1:
        return {}
    end = dates[-1 - skip_days]
    start = dates[-1 - skip_days - lookback_days]
    scores: dict[str, float] = {}
    for symbol in members:
        value, how = panel.period_return(symbol, start, end)
        if value is not None and how == "traded":
            scores[symbol] = value
    return scores
