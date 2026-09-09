"""Load the existing ETF daily cache into a point-in-time panel.

Reads `quantdesk`'s `etf_daily.csv` (raw OHLC, 15 tickers, 2006-2026) and wires
it through `PointInTimeUniverse` and `PricePanel` so the guards actually run on
real data rather than only on fixtures.

**What this universe does and does not fix.**

It removes the DELISTING leak by construction, because none of these fifteen
funds has ever stopped trading: every listing is open, and there is no terminal
return to forget. It does NOT remove the SELECTION leak, and the distinction
matters enough to state plainly rather than bury:

    These fifteen tickers were chosen in 2026 from funds that still exist in
    2026. Over a thousand US-listed ETFs have closed since 2006. A universe
    assembled today from survivors is a universe filtered by survival, however
    carefully its start dates are handled.

The bias is much smaller here than for single stocks -- large sector and
broad-market SPDRs close far less often than small thematic funds, and none of
these fifteen was ever a plausible closure candidate -- but "smaller" is not
"absent", and the honest description is that it is UNMEASURED, because measuring
it requires price history for the funds that closed, which no free source
provides.

`quantdesk.data.etf_daily` already notes that every ticker "existed before the
sample starts, so there is no survivorship in the selection." That is right
about the START dates and it is what makes this panel usable. It does not cover
the END dates, because nothing in this fifteen ever had one.

ASSUMPTION (data availability): a delisted-inclusive source -- Sharadar SEP,
Norgate, or CRSP -- is required before any equity universe wider than this can
be trusted. Until one exists, `PointInTimeUniverse` should be built from that
vendor's listing table, not from whatever tickers an API returns today.
"""

from __future__ import annotations

import csv
import os
from datetime import date

from .panel import PricePanel
from .universe import Listing, PointInTimeUniverse

#: The cache lives in the quantdesk repo, which is a sibling checkout.
DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "quantdesk", "etf_daily.csv",
)


def _parse_date(text: str) -> date:
    y, m, d = text.split("-")
    return date(int(y), int(m), int(d))


def load_closes(path: str | None = None) -> dict[str, dict[date, float]]:
    """Raw closes by ticker and date. Raw, not adjusted -- see `PricePanel`."""
    path = path or DEFAULT_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build it with "
            "`python -m quantdesk.data.etf_daily --build` in the quantdesk repo."
        )
    out: dict[str, dict[date, float]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            close = row["close"]
            if not close:
                continue
            out.setdefault(row["ticker"], {})[_parse_date(row["date"])] = float(close)
    return out


def build_universe(closes: dict[str, dict[date, float]]) -> PointInTimeUniverse:
    """Listings from the first observed trade date of each ticker.

    A fund's first trade is knowable on the day it happens, so a start date
    derived this way is causally sound. Every listing is left OPEN: all fifteen
    still trade, and inventing an end date for a live fund would be a fabrication
    that `Listing` would then dutifully price.

    The first observed date is the first date IN THIS CACHE, which is the later
    of the fund's inception and the cache's 20-year window. For the nine SPDR
    sectors that is the window, not the inception -- they launched in 1998. That
    is a truncation of history, not a leak: it makes the universe smaller and
    younger than reality, never more survivor-selected.
    """
    listings = []
    for ticker, series in sorted(closes.items()):
        if not series:
            continue
        listings.append(Listing(ticker, min(series)))
    return PointInTimeUniverse(listings)


def load_panel(path: str | None = None) -> PricePanel:
    closes = load_closes(path)
    return PricePanel(closes, build_universe(closes))


def month_ends(panel: PricePanel) -> list[date]:
    """Last available trading date of each month in the panel."""
    by_month: dict[tuple[int, int], date] = {}
    for d in panel.dates:
        by_month[(d.year, d.month)] = max(by_month.get((d.year, d.month), d), d)
    return sorted(by_month.values())
