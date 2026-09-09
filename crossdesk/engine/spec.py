"""The canonical execution spec every engine must reproduce.

Three engines mean three sets of assumptions about fills, costs and accounting.
Comparing them pairwise tells you they disagree; it does not tell you which one
is wrong. So this module is the pivot: forty lines of arithmetic, obviously
correct on inspection, and every engine is measured against IT rather than
against each other. A disagreement then has an owner.

The contract, which is quantdesk's and is inherited unchanged:

  * A decision is made from bar `i`'s close.
  * It executes at bar `i+1`'s OPEN. The decision bar is never the fill bar.
  * Equity at bar `i` marks the position held THROUGH bar i -- the one
    established at bar i's open -- against bar i's close.
  * A decision on the final bar expires unfilled. There is no bar to fill it in,
    and pretending otherwise books a trade that could not have happened.

Costs are deliberately linear and per-unit here, not the volatility-scaled model
quantdesk uses in research. The agreement test is about whether the engines AGREE,
and a cost model that only one of them can express turns a fill-convention test
into a cost-model test. Realistic costs come back once the conventions match.

Standard library only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


#: The two fill conventions the engines actually implement.
#:
#: "next_open"      -- decide on bar i's close, fill at bar i+1's OPEN.
#:                     quantdesk's contract; VectorBT reproduces it exactly.
#: "decision_close" -- decide on bar i's close, fill at bar i's OWN close.
#:                     What NautilusTrader does in bar-execution mode, measured
#:                     rather than assumed: 3 of 3 fills landed on the decision
#:                     bar's close in `scripts/_nautilus_probe.py`.
#:
#: "decision_close" is optimistic. Acting on a close at that same close requires
#: knowing the close before it prints. It is available here ONLY so Nautilus's
#: accounting can be verified against a reference that shares its convention --
#: never as a research setting.
FILL_CONVENTIONS = ("next_open", "decision_close")


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Costs and fill convention, expressed so any engine can be matched."""

    #: Cash per unit traded, charged on every fill regardless of direction.
    commission_per_unit: float = 0.0
    #: Price concession per unit, always against the trader: buys pay more,
    #: sells receive less.
    slippage_per_unit: float = 0.0
    initial_cash: float = 100_000.0
    fill_at: str = "next_open"

    def __post_init__(self) -> None:
        if self.fill_at not in FILL_CONVENTIONS:
            raise ValueError(
                f"fill_at must be one of {FILL_CONVENTIONS}, got {self.fill_at!r}"
            )


@dataclass(slots=True)
class SpecResult:
    equity: list[float] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)
    fills: list[tuple[int, float, float]] = field(default_factory=list)
    total_commission: float = 0.0
    total_slippage: float = 0.0
    expired_orders: int = 0

    @property
    def final_equity(self) -> float:
        return self.equity[-1] if self.equity else 0.0


def run_spec(opens: Sequence[float], closes: Sequence[float],
             targets: Sequence[float], spec: ExecutionSpec,
             start: int = 0) -> SpecResult:
    """Walk the bars applying the contract above, and return the equity curve.

    `targets[i]` is the position DECIDED at bar `i`, filled at bar `i+1`'s open.
    `start` skips a warmup so the curve can be aligned with an engine that needs
    history before it will trade.
    """
    n = len(closes)
    if len(opens) != n or len(targets) != n:
        raise ValueError(
            f"opens/closes/targets must be equal length, got "
            f"{len(opens)}/{n}/{len(targets)}"
        )
    if not 0 <= start < n:
        raise ValueError(f"start {start} outside 0..{n - 1}")

    out = SpecResult()
    cash = spec.initial_cash
    qty = 0.0

    next_open = spec.fill_at == "next_open"

    for i in range(start, n):
        # Mark the position held through this bar against this bar's close.
        # Under "decision_close" the fill happens on THIS bar, so the mark has
        # to come after it; that branch appends its equity below instead.
        if next_open:
            out.equity.append(cash + qty * closes[i])
            out.positions.append(qty)

        delta = targets[i] - qty
        if delta != 0:
            if next_open and i + 1 >= n:
                # No bar left to fill in. The order expires -- it does not
                # quietly fill at this bar's close.
                out.expired_orders += 1
            else:
                direction = 1.0 if delta > 0 else -1.0
                base = opens[i + 1] if next_open else closes[i]
                fill_price = base + direction * spec.slippage_per_unit
                commission = abs(delta) * spec.commission_per_unit

                cash -= delta * fill_price + commission
                qty = targets[i]

                out.fills.append((i + 1 if next_open else i, delta, fill_price))
                out.total_commission += commission
                out.total_slippage += abs(delta) * spec.slippage_per_unit

        if not next_open:
            out.equity.append(cash + qty * closes[i])
            out.positions.append(qty)

    return out


def sma_crossover_targets(closes: Sequence[float], fast: int, slow: int,
                          size: float = 1.0) -> list[float]:
    """A deterministic long/flat signal, computed once and shared by every engine.

    The signal is precomputed in plain Python on purpose. If each engine computed
    its own moving average, a disagreement in the equity curve could come from
    the indicator or from the fill, and the test could not tell them apart. One
    array of targets in, three equity curves out, and only execution differs.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be below slow ({slow})")
    targets = [0.0] * len(closes)
    for i in range(len(closes)):
        if i + 1 < slow:
            continue
        fast_ma = sum(closes[i + 1 - fast:i + 1]) / fast
        slow_ma = sum(closes[i + 1 - slow:i + 1]) / slow
        targets[i] = size if fast_ma > slow_ma else 0.0
    return targets
