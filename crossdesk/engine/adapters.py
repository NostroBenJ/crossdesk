"""Run one strategy through three engines, from one shared signal array.

Every adapter takes the SAME precomputed `targets` and returns an equity curve.
Nothing here computes an indicator: if each engine calculated its own moving
average, a disagreement could come from the indicator or from the fill, and the
agreement test could not tell them apart.

Engine imports live inside the functions, so `import crossdesk.engine.adapters`
works without vectorbt or nautilus_trader installed -- the same rule as
`plot_greeks()` in `black_scholes.py`.

**Known convention divergence, measured not assumed.** quantdesk and VectorBT
both fill at the next bar's open and reproduce the spec to floating-point
equality. NautilusTrader in bar-execution mode fills at the DECISION bar's own
close, which is optimistic. `run_nautilus` is therefore checked against the
`decision_close` spec, and `convention_gap()` measures what the difference is
worth on real data. See `scripts/engine_agreement.py`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from .spec import ExecutionSpec, run_spec

EPOCH = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)


def make_ohlc(closes: Sequence[float], opens: Sequence[float],
              pad: float = 0.003) -> tuple[list[float], list[float]]:
    """Highs and lows wide enough to contain the opens and closes.

    quantdesk's `Bar` refuses a bar whose open or close sits outside [low, high],
    which is a good validation and a nuisance for synthetic data. This keeps the
    generator honest without weakening the check.
    """
    highs = [max(o, c) * (1 + pad) for o, c in zip(opens, closes)]
    lows = [min(o, c) * (1 - pad) for o, c in zip(opens, closes)]
    return highs, lows


def run_reference(opens, closes, targets, spec: ExecutionSpec,
                  start: int = 0) -> list[float]:
    return run_spec(opens, closes, targets, spec, start).equity


def run_quantdesk(opens, closes, targets, spec: ExecutionSpec,
                  volatility_lookback: int = 20) -> list[float]:
    """quantdesk's event loop, configured to the spec's conventions."""
    from quantdesk.backtest.costs import ZeroCostModel
    from quantdesk.backtest.engine import (
        BacktestEngine, EngineConfig, FixedSizer,
    )
    from quantdesk.backtest.fills import NextBarOpenFill
    from quantdesk.backtest.view import MarketView
    from quantdesk.core.clock import Timeframe, utc
    from quantdesk.core.types import (
        Bar, BarCompleteness, Direction, InstrumentSpec, Signal,
    )
    from quantdesk.signals.base import SignalModule

    if spec.commission_per_unit or spec.slippage_per_unit:
        raise NotImplementedError(
            "the quantdesk adapter runs frictionless. Its cost model is "
            "volatility-scaled by design and cannot express a flat per-unit "
            "charge, so matching costs across engines needs a shared cost "
            "model first -- a separate piece of work from matching fills."
        )

    highs, lows = make_ohlc(closes, opens)
    timeframe = Timeframe("1d", 86_400)
    bars = []
    for i in range(len(closes)):
        t0 = utc(2020, 1, 1) + dt.timedelta(days=i)
        bars.append(Bar("TEST", t0, t0 + dt.timedelta(days=1),
                        opens[i], highs[i], lows[i], closes[i],
                        1e9, BarCompleteness.COMPLETE))
    index_of = {bar.ts_close: i for i, bar in enumerate(bars)}

    class Replay(SignalModule):
        warmup = 0

        def generate_signal(self, state):
            target = targets[index_of[state.ts]]
            return Signal(module=self.name, ts=state.ts, symbol="TEST",
                          direction=Direction.LONG if target > 0 else Direction.FLAT,
                          confidence=1.0 if target > 0 else 0.0)

    # FixedSizer must be told the same size the targets encode, or quantdesk
    # trades one share while the other engines trade the intended position.
    size = max((abs(t) for t in targets), default=1.0) or 1.0
    engine = BacktestEngine(
        InstrumentSpec("TEST"), timeframe,
        ZeroCostModel(), NextBarOpenFill(1.0), FixedSizer(size),
        EngineConfig(spec.initial_cash, volatility_lookback, None,
                     flatten_at_end=False),
    )
    return engine.run(MarketView("TEST", bars), Replay()).equity


def quantdesk_warmup(volatility_lookback: int = 20) -> int:
    """Where quantdesk's equity curve starts, so the spec can be aligned to it."""
    return volatility_lookback + 1


def mask_warmup(targets: Sequence[float], warmup: int) -> list[float]:
    """Zero every decision before `warmup` so all engines trade one window.

    Necessary, and the reason is a bug this test caught on its first run.
    quantdesk refuses to trade until it has enough history to estimate bar
    volatility, so its curve starts at bar 21. VectorBT has no such notion and
    happily trades from bar 0. On most random series that costs nothing --
    the strategy is flat early anyway -- but on a series where the crossover
    fires at bar 20, VectorBT enters a position that quantdesk never took, and
    the two curves are offset by the position's value for the rest of the run.

    The divergence looks like a fill-convention bug and is not one. Masking makes
    the comparison apples to apples by giving every engine the same decisions.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")
    return [0.0] * min(warmup, len(targets)) + list(targets[warmup:])


def run_vectorbt(opens, closes, targets, spec: ExecutionSpec) -> list[float]:
    """VectorBT, configured to fill at the next bar's open.

    The shift is the whole trick: `targets[i]` is DECIDED at bar i, so the target
    position effective from bar i+1 is `targets[i]`. Feeding the unshifted array
    fills at the decision bar and reintroduces exactly the lookahead the
    convention exists to prevent.
    """
    import numpy as np
    import pandas as pd
    import vectorbt as vbt

    if spec.fill_at != "next_open":
        raise NotImplementedError(
            f"the VectorBT adapter implements next_open only, got {spec.fill_at!r}"
        )

    n = len(closes)
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    shifted = [np.nan] + list(targets[:-1])
    portfolio = vbt.Portfolio.from_orders(
        close=pd.Series(list(closes), index=index),
        size=pd.Series(shifted, index=index),
        size_type="targetamount",
        price=pd.Series(list(opens), index=index),
        init_cash=spec.initial_cash,
        fees=0.0,
        slippage=0.0,
    )
    return [float(v) for v in portfolio.value().values]


def run_nautilus(opens, closes, targets, spec: ExecutionSpec) -> list[float]:
    """NautilusTrader in bar-execution mode.

    Returns equity marked at each bar's close. Note the convention: Nautilus
    fills market orders submitted from `on_bar` at THAT bar's close, so this is
    comparable to the `decision_close` spec, not to quantdesk.
    """
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import Bar, BarSpecification, BarType
    from nautilus_trader.model.enums import (
        AccountType, AggregationSource, BarAggregation, OmsType, OrderSide,
        PriceType,
    )
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Money, Price, Quantity
    from nautilus_trader.test_kit.providers import TestInstrumentProvider
    from nautilus_trader.trading.strategy import Strategy

    n = len(closes)
    opens = [round(float(x), 2) for x in opens]
    closes = [round(float(x), 2) for x in closes]
    highs, lows = make_ohlc(closes, opens)
    highs = [round(x, 2) for x in highs]
    lows = [round(x, 2) for x in lows]

    instrument = TestInstrumentProvider.equity(symbol="TEST", venue="SIM")
    bar_type = BarType(
        instrument_id=instrument.id,
        bar_spec=BarSpecification(1, BarAggregation.DAY, PriceType.LAST),
        aggregation_source=AggregationSource.EXTERNAL,
    )
    bars = []
    for i in range(n):
        ts = int((EPOCH + dt.timedelta(days=i + 1)).timestamp() * 1e9)
        bars.append(Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{opens[i]:.2f}"),
            high=Price.from_str(f"{highs[i]:.2f}"),
            low=Price.from_str(f"{lows[i]:.2f}"),
            close=Price.from_str(f"{closes[i]:.2f}"),
            volume=Quantity.from_str("1000000"),
            ts_event=ts, ts_init=ts,
        ))

    # ACTUAL fills, not intended ones. Recording the target as though it filled
    # hides rejections and partial fills -- and Nautilus on a CASH account WILL
    # reject an order it cannot fund, exactly like VectorBT. An adapter that
    # assumed the fill would report a position the engine never held.
    fills: list[tuple[int, float, float]] = []

    class Replay(Strategy):
        def __init__(self):
            super().__init__(StrategyConfig())
            self.i = -1
            self.intended = 0.0

        def on_start(self):
            self.subscribe_bars(bar_type)

        def on_bar(self, bar):
            self.i += 1
            delta = targets[self.i] - self.intended
            if delta != 0:
                self.submit_order(self.order_factory.market(
                    instrument_id=instrument.id,
                    order_side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                    quantity=Quantity.from_str(f"{abs(delta):.0f}"),
                ))
                self.intended = targets[self.i]

        def on_order_filled(self, event):
            signed = float(event.last_qty)
            if event.order_side == OrderSide.SELL:
                signed = -signed
            fills.append((self.i, signed, float(event.last_px)))

    engine = BacktestEngine(config=BacktestEngineConfig(trader_id="AGREE-001"))
    engine.add_venue(
        venue=Venue("SIM"), oms_type=OmsType.NETTING,
        account_type=AccountType.CASH, base_currency=USD,
        starting_balances=[Money(int(spec.initial_cash), USD)],
        bar_execution=True,
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(Replay())
    engine.run()

    # Rebuild the equity curve from the fills Nautilus ACTUALLY made, at the
    # prices it actually used. Reading it out of Nautilus's own portfolio would
    # compare the spec against Nautilus's REPORTING as well as its execution;
    # this isolates execution, which is what the test is about.
    by_bar: dict[int, list[tuple[float, float]]] = {}
    for i, signed, price in fills:
        by_bar.setdefault(i, []).append((signed, price))

    cash = spec.initial_cash
    qty = 0.0
    equity = []
    for i in range(n):
        for signed, price in by_bar.get(i, ()):
            cash -= signed * price
            qty += signed
        equity.append(cash + qty * closes[i])
    engine.dispose()
    return equity


def convention_gap(opens, closes, targets, initial_cash: float = 100_000.0
                   ) -> tuple[float, float, float]:
    """Final equity under each convention, and the gap in basis points.

    This is the number the agreement test exists to produce: what the choice of
    fill convention is worth, on this data, for this strategy -- with everything
    else held identical.
    """
    a = run_spec(opens, closes, targets,
                 ExecutionSpec(initial_cash=initial_cash, fill_at="next_open"))
    b = run_spec(opens, closes, targets,
                 ExecutionSpec(initial_cash=initial_cash, fill_at="decision_close"))
    gap_bps = (b.final_equity - a.final_equity) / initial_cash * 10_000
    return a.final_equity, b.final_equity, gap_bps
