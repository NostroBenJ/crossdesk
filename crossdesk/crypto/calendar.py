"""Crypto calendar constants.

Perpetual futures do not close. Every annualisation constant therefore differs
from the equity defaults baked into `quantdesk.core.clock`, and none of them is
252.

This module does not define a new clock. `quantdesk.core.clock.Timeframe`
already carries `seconds_per_session` and `sessions_per_year` as parameters
precisely so a 24/7 calendar can be expressed without forking it. All this file
does is supply the two numbers and pin them with tests.

The trap, stated numerically: using the SPY 5-minute constant (19_656) on
5-minute perp bars understates volatility by sqrt(105_120 / 19_656) = 2.31x,
and therefore oversizes every volatility-targeted position by the same factor.
"""

from __future__ import annotations

from quantdesk.core.clock import Timeframe

#: Perps trade every second of every day. There is no session boundary.
SECONDS_PER_CRYPTO_DAY = 86_400

#: Calendar days, not trading days. A crypto "year" is 365 days, never 252.
#: ASSUMPTION (calendar): leap years are ignored, as they are for equities.
#: The 0.27% error is far below the noise in any vol estimate built on it.
DAYS_PER_YEAR = 365


def crypto(spec: str) -> Timeframe:
    """A `Timeframe` on the 24/7 calendar.

    >>> crypto("5m").periods_per_year
    105120.0
    >>> crypto("1d").periods_per_year
    365.0
    """
    return Timeframe.parse(
        spec,
        seconds_per_session=SECONDS_PER_CRYPTO_DAY,
        sessions_per_year=DAYS_PER_YEAR,
    )


MINUTE_1 = crypto("1m")
MINUTE_5 = crypto("5m")
MINUTE_15 = crypto("15m")
HOUR_1 = crypto("1h")
HOUR_4 = crypto("4h")
DAY_1 = crypto("1d")
