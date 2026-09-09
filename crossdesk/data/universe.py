"""Point-in-time universe membership, and the two survivorship leaks.

Cross-sectional research adds a leak class that time-series research does not
have. quantdesk's `assert_causal` guards a FEATURE against reading the future.
It cannot guard the UNIVERSE, because the universe is chosen before any feature
is computed -- and choosing it wrong biases every result downstream in the same
optimistic direction.

There are two distinct leaks here. They are separated because they have
different causes, different fixes, and different magnitudes, and lumping them
together as "survivorship bias" is how one of them goes unfixed:

**1. Selection leak.** Using today's membership list for a past date. Backtest
the current S&P 500 over 2010-2025 and every name in it survived fifteen years
and was good enough to still be in the index -- a filter applied with perfect
hindsight. `PointInTimeUniverse` exists so membership is a function of date,
and `survivorship_leak` names the symbols a candidate universe is claiming too
early or holding too long.

**2. Delisting leak.** A symbol that stops trading vanishes from the price
panel. If the panel is joined on "dates where a price exists", a bankruptcy
becomes an ABSENCE rather than a -100% return, and the cross-section quietly
drops its worst outcomes. This one is the more dangerous of the two because it
looks like a data-cleanliness step. `Listing.delisting_return` is mandatory for
any listing that has ended, so the terminal outcome cannot be forgotten -- it
must be stated, even if what you state is an assumption.

Standard library only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date


class SurvivorshipError(AssertionError):
    """Raised when a universe or panel carries a leak. Never caught in research code."""


#: Terminal returns by delisting reason.
#: ASSUMPTION (corporate actions): a merger or acquisition settles near the last
#: traded price, so 0.0 is a fair terminal return. A bankruptcy or exchange
#: delisting for non-compliance is closer to a total loss; -1.0 is the
#: conservative choice and it is deliberately harsher than reality (shells often
#: retain a few cents) because the failure mode of the opposite error -- omitting
#: the loss entirely -- is unbounded.
DELISTING_RETURNS = {
    "acquired": 0.0,
    "merged": 0.0,
    "renamed": 0.0,
    "bankrupt": -1.0,
    "delisted": -1.0,
    "unknown": -1.0,
    # The terminal outcome is ALREADY in the price series -- e.g. a CRSP
    # total-return index with the delisting return folded into its final
    # month. The exit value is the last observed price and nothing further
    # is assumed. See `panel.PricePanel.period_return`.
    "priced": 0.0,
}


@dataclass(frozen=True, slots=True)
class Listing:
    """One symbol's membership window, with its terminal outcome if it ended.

    `end` is the LAST date the symbol was tradable, inclusive. `None` means the
    symbol is still listed as of the data's own as-of date.
    """

    symbol: str
    start: date
    end: date | None = None
    reason: str | None = None
    delisting_return: float | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if self.end is not None and self.end < self.start:
            raise ValueError(
                f"{self.symbol}: end {self.end} precedes start {self.start}"
            )
        if self.end is None:
            if self.delisting_return is not None:
                raise ValueError(
                    f"{self.symbol}: still listed but carries a delisting_return"
                )
            return
        # An ended listing MUST state its terminal outcome. This is the whole
        # point of the class: a delisting you forgot to price is a delisting
        # that silently vanishes from the cross-section.
        if self.delisting_return is None:
            if self.reason is None:
                raise ValueError(
                    f"{self.symbol}: listing ended {self.end} with neither a "
                    "reason nor a delisting_return. State the terminal outcome "
                    "-- an unpriced delisting drops the worst returns in the "
                    "panel and inflates every cross-sectional result."
                )
            if self.reason not in DELISTING_RETURNS:
                raise ValueError(
                    f"{self.symbol}: unknown delisting reason {self.reason!r}; "
                    f"expected one of {sorted(DELISTING_RETURNS)}"
                )
            object.__setattr__(
                self, "delisting_return", DELISTING_RETURNS[self.reason]
            )

    def is_active(self, on: date) -> bool:
        return self.start <= on and (self.end is None or on <= self.end)


class PointInTimeUniverse:
    """Membership as a function of date, and nothing else.

    Deliberately has no notion of "the current universe". There is no property
    that returns today's members without a date argument, because that method is
    precisely the one that gets called by accident inside a loop over history.
    """

    def __init__(self, listings: Iterable[Listing]) -> None:
        self._listings = tuple(listings)
        by_symbol: dict[str, list[Listing]] = {}
        for lst in self._listings:
            by_symbol.setdefault(lst.symbol, []).append(lst)
        for symbol, group in by_symbol.items():
            windows = sorted(group, key=lambda l: l.start)
            for a, b in zip(windows, windows[1:]):
                if a.end is None or b.start <= a.end:
                    raise ValueError(
                        f"{symbol}: overlapping listing windows "
                        f"({a.start}..{a.end}) and ({b.start}..{b.end})"
                    )
        self._by_symbol = by_symbol

    @property
    def listings(self) -> tuple[Listing, ...]:
        return self._listings

    @property
    def symbols(self) -> frozenset[str]:
        """Every symbol that was EVER a member. Not a tradable universe."""
        return frozenset(self._by_symbol)

    def members_asof(self, on: date) -> frozenset[str]:
        """Symbols tradable on `on`, using only information dated on or before it."""
        return frozenset(lst.symbol for lst in self._listings if lst.is_active(on))

    def size_asof(self, on: date) -> int:
        return len(self.members_asof(on))

    def delisting_return(self, symbol: str, on: date) -> float | None:
        """Terminal return if `symbol`'s listing ends exactly on `on`."""
        for lst in self._by_symbol.get(symbol, ()):
            if lst.end == on:
                return lst.delisting_return
        return None

    def survivors_only(self, asof: date) -> frozenset[str]:
        """The WRONG universe: symbols still listed at `asof`.

        Provided explicitly, and named to be uncomfortable, so that a
        survivorship-biased run has to be written on purpose and can be measured
        against the correct one. See `scripts/survivorship_bias.py`.
        """
        return frozenset(
            lst.symbol for lst in self._listings
            if lst.start <= asof and (lst.end is None or lst.end >= asof)
        )


def survivorship_leak(candidate: Iterable[str], universe: PointInTimeUniverse,
                      on: date) -> frozenset[str]:
    """Symbols in `candidate` that were NOT tradable on `on`.

    A non-empty result means the candidate universe knows something about `on`
    that nobody knew at the time -- either a name that had not listed yet, or one
    that had already stopped trading.
    """
    return frozenset(candidate) - universe.members_asof(on)


def assert_no_survivorship(candidate_by_date: dict[date, Iterable[str]],
                           universe: PointInTimeUniverse) -> None:
    """Fail loudly if any date's candidate set contains a non-member.

    The cross-sectional analogue of `quantdesk.backtest.lookahead.assert_causal`.
    Three lines to call, and it catches the bug that no amount of reading the
    strategy code will reveal, because the leak is in the universe rather than in
    the signal.
    """
    problems: list[str] = []
    for on in sorted(candidate_by_date):
        leaked = survivorship_leak(candidate_by_date[on], universe, on)
        if leaked:
            shown = sorted(leaked)[:8]
            problems.append(f"  {on}: {shown}{' ...' if len(leaked) > 8 else ''}")
    if problems:
        raise SurvivorshipError(
            f"universe leak on {len(problems)} date(s):\n" + "\n".join(problems[:20])
        )


def assert_delistings_priced(universe: PointInTimeUniverse) -> None:
    """Every ended listing must carry a terminal return.

    `Listing.__post_init__` enforces this at construction, so this is a second
    line of defence for universes built by deserialisation or by a source that
    bypassed the constructor.
    """
    unpriced = [lst.symbol for lst in universe.listings
                if lst.end is not None and lst.delisting_return is None]
    if unpriced:
        raise SurvivorshipError(
            f"{len(unpriced)} ended listing(s) without a delisting return: "
            f"{sorted(set(unpriced))[:10]}"
        )


def as_known_on(listings: Iterable[Listing], on: date) -> list[Listing]:
    """The listing records as they would have appeared on `on`.

    Listings that had not started yet are gone. Listings whose end had not
    happened yet are open, because on `on` nobody knew they would ever end.
    """
    out: list[Listing] = []
    for lst in listings:
        if lst.start > on:
            continue
        if lst.end is not None and lst.end > on:
            out.append(Listing(lst.symbol, lst.start, None))
        else:
            out.append(lst)
    return out


def assert_membership_is_causal(
    build: "Callable[[Sequence[Listing], date], Iterable[str]]",
    listings: Sequence[Listing],
    dates: Sequence[date],
) -> None:
    """Truncation test on a universe BUILDER, not on a finished universe.

    Note the signature. An earlier version of this function took a
    `PointInTimeUniverse` and rebuilt it from truncated listings -- which can
    never fail, because `members_asof` consults only `start` and `end` against
    the date it was given. It was a tautology wearing the costume of a test.

    The realistic bug is upstream of that, in the code that decides WHICH
    listings exist: filters like "symbols with at least N observations", "names
    still trading at the end of the sample", or "tickers Yahoo returns today".
    Each reads the whole history to decide membership at every date, and each
    looks like data hygiene rather than a leak.

    So `build` takes the listing records and an as-of date and returns the
    members it would trade. This runs it twice -- once with the full record, once
    with only what was knowable on that date -- and requires the same answer.
    """
    for d in dates:
        full = frozenset(build(list(listings), d))
        truncated = frozenset(build(as_known_on(listings, d), d))
        if full != truncated:
            raise SurvivorshipError(
                f"universe construction on {d} depends on post-{d} information; "
                f"only-when-future-visible={sorted(full - truncated)}, "
                f"only-when-hidden={sorted(truncated - full)}"
            )
