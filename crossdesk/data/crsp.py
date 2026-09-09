"""CRSP monthly stock file -> point-in-time listings and a total-return panel.

CRSP is the dataset the rest of this package was written in anticipation of.
Three things it has that no free source does, and how each lands here:

**1. The delisting return.** `dlret` is what a holder actually received in the
month a stock stopped trading -- the takeover price, or the last cents of a
bankruptcy. `research/wrds_pull.py` folds it into `ret_adj` for the final row,
so the total-return index built here already contains the terminal outcome.
A delisted window therefore ends at the LAST FULL MONTH the name could be
held, and the delisting month's `ret_adj` becomes the listing's explicit
`delisting_return` -- a CRSP number, not an assumption -- which
`PricePanel.period_return` applies on top of that month-end's index level. A
window that leaves the share-code/exchange filter without a delisting code
ends at its last row with a terminal return of 0.0: the exit is the last
index value and nothing further is assumed.

    ASSUMPTION (missing delisting returns): when CRSP records a delisting
    code but no return, the Shumway (1997) convention is applied -- -30% for
    performance-related delistings (codes 400-599), 0% for mergers and
    exchange moves (200-399). Rows so adjusted are counted in
    `CrspMonthly.shumway_adjusted` and reported by the audit.

**2. PERMNO.** Tickers are reused; PERMNO is not. The panel is keyed by
PERMNO (as a string) and the ticker/company name are carried as labels only.
Two different businesses that traded under the same ticker are two series.

**3. Dated share and exchange codes.** The pull keeps a row only in months the
security was ordinary common stock (SHRCD 10/11) on NYSE, AMEX or NASDAQ
(EXCHCD 1-3). A name that moves to OTC leaves the panel that month and may
come back later, so one PERMNO can have several listing windows. Contiguous
months form a window; a gap ends one and starts another. A window that ends
without a delisting code is a filter exit, and its terminal return is likewise
0.0 with the exit price being the last observed index value.

The membership rule that widens this to a usable universe is `top_by_mktcap`:
the N largest names by market cap on the rebalance date itself, which reads
nothing dated after that day and so passes `assert_membership_is_causal`.

Standard library only. `scripts/crsp_import.py` converts the pull's parquet
into the CSV this module reads.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from .panel import PricePanel
from .universe import Listing, PointInTimeUniverse

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "crsp_monthly.csv",
)

#: Shumway (1997): the average delisting return when CRSP has none.
SHUMWAY_PERFORMANCE = -0.30


@dataclass
class CrspMonthly:
    """The parsed file: one entry per (permno, month), in date order."""

    #: permno -> list of (date, ret_adj, mktcap, dlstcd, dlret_present)
    rows: dict[str, list[tuple[date, float, float, int, bool]]] = field(default_factory=dict)
    #: permno -> (last ticker, last company name)
    labels: dict[str, tuple[str, str]] = field(default_factory=dict)
    shumway_adjusted: int = 0
    dates: tuple[date, ...] = ()


def _parse_date(text: str) -> date:
    y, m, d = text[:10].split("-")
    return date(int(y), int(m), int(d))


def _month_index(d: date) -> int:
    return d.year * 12 + d.month


def _float(text: str) -> float:
    if text in ("", "nan", "NaN", "None"):
        return math.nan
    return float(text)


def load(path: str | None = None) -> CrspMonthly:
    """Read the CSV written by `scripts/crsp_import.py`.

    Columns: permno, date, ret_adj, mktcap, dlstcd, dlret, ticker, comnam.
    A row whose `ret_adj` is NaN is skipped for return purposes but still
    counts as presence for the listing window (CRSP leaves the first month's
    return blank because there is no prior price)."""
    path = path or DEFAULT_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Pull CRSP with `py -3.14 -m research.wrds_pull --pull` "
            "in dev/cisd-bot, then run `python scripts/crsp_import.py`."
        )
    out = CrspMonthly()
    date_cache: dict[str, date] = {}
    all_dates: set[date] = set()
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            permno = row["permno"]
            text = row["date"][:10]
            d = date_cache.get(text)
            if d is None:
                d = _parse_date(text)
                date_cache[text] = d
            all_dates.add(d)
            code_text = row.get("dlstcd", "")
            code = int(float(code_text)) if code_text not in ("", "nan", "NaN", "None") else 0
            dl_present = row.get("dlret", "") not in ("", "nan", "NaN", "None")
            out.rows.setdefault(permno, []).append(
                (d, _float(row["ret_adj"]), _float(row.get("mktcap", "")), code, dl_present)
            )
            out.labels[permno] = (row.get("ticker", "") or "", row.get("comnam", "") or "")
    # CRSP dates a security's monthly row on ITS last trading day of the
    # month, which for a halted or thinly traded name can precede the common
    # month end by a day or two. A rebalance on the common date would then
    # find the name a member (its window spans the date) with no value on it
    # -- the audit reported exactly that as silent dropout. Every row is
    # therefore re-dated to the latest date observed in its calendar month.
    canon: dict[int, date] = {}
    for d in all_dates:
        m = _month_index(d)
        if m not in canon or d > canon[m]:
            canon[m] = d
    for permno, series in out.rows.items():
        out.rows[permno] = [(canon[_month_index(r[0])],) + r[1:] for r in series]
        out.rows[permno].sort(key=lambda r: r[0])
    out.dates = tuple(sorted(set(canon.values())))
    return out


def _windows(series: Sequence[tuple[date, float, float, int, bool]]
             ) -> list[list[tuple[date, float, float, int, bool]]]:
    """Split one PERMNO's rows into runs of consecutive months."""
    out: list[list] = []
    current: list = []
    prev: int | None = None
    for row in series:
        m = _month_index(row[0])
        if prev is not None and m != prev + 1:
            out.append(current)
            current = []
        current.append(row)
        prev = m
    if current:
        out.append(current)
    return out


def build(data: CrspMonthly, asof: date | None = None
          ) -> tuple[PointInTimeUniverse, PricePanel, dict[str, tuple[int, bool]]]:
    """Listings, a total-return panel, and the delisting record per window end.

    `asof` is the data's own as-of date: a window whose last month is the
    final month in the file is still open, not a delisting.

    The panel's "price" for a PERMNO is a total-return index that starts at
    100 on its first month in a window and compounds `ret_adj` thereafter.
    `PricePanel.period_return` on two index values is therefore the total
    return with dividends, and on a window that ends inside the period it is
    the return to the exit value (see `panel.period_return`), which already
    includes the delisting return.
    """
    asof = asof or data.dates[-1]
    asof_m = _month_index(asof)
    listings: list[Listing] = []
    prices: dict[str, dict[date, float]] = {}
    exits: dict[str, tuple[int, bool]] = {}

    for permno, series in data.rows.items():
        index_series: dict[date, float] = {}
        for window in _windows(series):
            final_code = window[-1][3]
            # A delisting code is a delisting whenever it occurs, including in
            # the file's final month; the as-of date only decides whether a
            # window WITHOUT a code is still open.
            delisted = final_code >= 200
            if delisted and len(window) < 2:
                # Listed and delisted inside one month: never holdable at a
                # month end, so it is not a member of any rebalance.
                continue
            # The delisting-month row records a return that ENDED the
            # position during that month. The name is a member up to the
            # previous month end; the final row's return (dlret folded in)
            # is the listing's terminal return, applied on top of the
            # previous month's index level by `PricePanel.period_return`.
            tradable = window[:-1] if delisted else window
            level = 100.0
            for i, (d, ret, _cap, _code, _dl) in enumerate(tradable):
                if i > 0 and not math.isnan(ret):
                    level *= 1.0 + ret
                index_series[d] = level
            first, last = tradable[0][0], tradable[-1][0]
            if delisted:
                _d, ret, _cap, code, dl_present = window[-1]
                terminal = 0.0 if math.isnan(ret) else ret
                if not dl_present:
                    # Delisting code but no delisting return: Shumway convention.
                    if 400 <= code < 600:
                        terminal = (1.0 + terminal) * (1.0 + SHUMWAY_PERFORMANCE) - 1.0
                    data.shumway_adjusted += 1
                listings.append(Listing(permno, first, last, reason="priced",
                                        delisting_return=terminal))
                exits[f"{permno}@{last}"] = (code, dl_present)
            elif _month_index(last) >= asof_m:
                listings.append(Listing(permno, first, None))
            else:
                # Left the share-code / exchange filter: exit at the last
                # index value, nothing further assumed.
                listings.append(Listing(permno, first, last, reason="priced"))
                exits[f"{permno}@{last}"] = (final_code, window[-1][4])
        prices[permno] = index_series

    universe = PointInTimeUniverse(listings)
    return universe, PricePanel(prices, universe), exits


def mktcap_table(data: CrspMonthly) -> dict[date, dict[str, float]]:
    """date -> permno -> market cap, for the size-ranked membership rule."""
    out: dict[date, dict[str, float]] = {}
    for permno, series in data.rows.items():
        for d, _ret, cap, _code, _dl in series:
            if not math.isnan(cap) and cap > 0:
                out.setdefault(d, {})[permno] = cap
    return out


def top_by_mktcap(caps: dict[date, dict[str, float]], n: int):
    """A membership BUILDER: the N largest names by market cap on the date.

    Returns a callable with the `assert_membership_is_causal` signature. It
    consults only `caps[asof]`, which is dated on `asof`, and the listing
    windows against `asof`; nothing later is read, so the guard passes.

    Size is the standard liquidity screen in the cross-sectional literature
    and it is causal by construction. "Names with N years of history" and
    "names that survive the sample" are the filters that are not, and neither
    is offered here.
    """
    def build(listings, asof):
        active = {l.symbol for l in listings if l.is_active(asof)}
        today = caps.get(asof, {})
        ranked = sorted((p for p in today if p in active), key=lambda p: -today[p])
        return ranked[:n]
    return build


def month_ends(data: CrspMonthly) -> list[date]:
    return list(data.dates)
