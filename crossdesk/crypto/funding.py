"""Perpetual funding: sign, scale, and when it is actually a cashflow.

Three conventions here cause more wrong numbers than anything else in perp
research, so each one is a function with a test rather than a comment:

1. **Scale.** Funding is quoted PER INTERVAL, not per year. A 0.01% 8-hour rate
   is 10.95%/yr, not 0.01%. Comparing a per-interval rate against an annualised
   return makes real carry look like rounding error.

2. **Sign.** Positive funding means LONGS PAY SHORTS. A short position with
   positive funding receives. Getting this backwards inverts the entire
   strategy, and the backtest will look plausible either way.

3. **Timing.** Funding is paid only on positions open AT the settlement
   timestamp. Accruing it continuously earns carry that was never received, and
   the error is largest exactly when funding is most extreme -- which is when
   the strategy is most active.

The predicted rate and the realised rate are different series. The predicted
rate is knowable before settlement and is therefore a legitimate FEATURE. Only
the realised rate at settlement is P&L. Never let one stand in for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quantdesk.core.clock import ensure_utc
from .calendar import DAYS_PER_YEAR

#: Binance and Bybit settle every 8 hours: 00:00, 08:00, 16:00 UTC.
#: ASSUMPTION (venue): some venues and some symbols use 1h or 4h intervals,
#: and venues may switch a symbol's interval during stress. Verify per symbol
#: against the venue's funding history rather than assuming 8h globally.
DEFAULT_INTERVAL_HOURS = 8
DEFAULT_SETTLEMENT_HOURS_UTC = (0, 8, 16)


def intervals_per_day(interval_hours: int = DEFAULT_INTERVAL_HOURS) -> float:
    """How many funding settlements happen in a day."""
    if interval_hours <= 0 or 24 % interval_hours != 0:
        raise ValueError(
            f"interval_hours must divide 24 evenly, got {interval_hours}"
        )
    return 24 / interval_hours


def annualise(rate_per_interval: float,
              interval_hours: int = DEFAULT_INTERVAL_HOURS) -> float:
    """Convert a per-interval funding rate to a simple annual rate.

    Simple, not compounded: funding is a cashflow on notional, and the position
    is not automatically reinvested. Compounding it overstates the carry and
    makes the tail look survivable when it is not.

    >>> round(annualise(0.0001), 4)   # 0.01% per 8h
    0.1095
    """
    return rate_per_interval * intervals_per_day(interval_hours) * DAYS_PER_YEAR


def is_settlement(ts: datetime,
                  settlement_hours: tuple[int, ...] = DEFAULT_SETTLEMENT_HOURS_UTC
                  ) -> bool:
    """True if `ts` is exactly a funding settlement instant (UTC)."""
    t = ensure_utc(ts)
    return (t.hour in settlement_hours
            and t.minute == 0 and t.second == 0 and t.microsecond == 0)


def cashflow(position_qty: float, mark_price: float, rate: float) -> float:
    """Funding P&L for a position held THROUGH a settlement, in quote currency.

    `position_qty` is signed: positive is long, negative is short.

    Positive `rate` means longs pay shorts, so a long's cashflow is negative:

        cashflow = -position_qty * mark_price * rate

    >>> cashflow(1.0, 50_000.0, 0.0001)     # long, positive funding -> pays
    -5.0
    >>> cashflow(-1.0, 50_000.0, 0.0001)    # short, positive funding -> receives
    5.0
    >>> cashflow(1.0, 50_000.0, -0.0001)    # long, negative funding -> receives
    5.0
    """
    return -position_qty * mark_price * rate


@dataclass(frozen=True, slots=True)
class FundingEvent:
    """One realised settlement. Constructed only from venue history."""

    ts: datetime
    symbol: str
    rate: float
    mark_price: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts))
        if self.mark_price <= 0:
            raise ValueError(f"mark_price must be positive, got {self.mark_price}")

    def pnl(self, position_qty: float) -> float:
        return cashflow(position_qty, self.mark_price, self.rate)

    @property
    def annualised(self) -> float:
        return annualise(self.rate)
