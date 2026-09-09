"""Scratch probe: where does Nautilus actually fill a market order sent on_bar?

Not a test. It prints the fill prices next to the bar opens and closes so the
convention can be READ off the output rather than assumed from documentation.
"""

import datetime as dt
import random

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    BarAggregation,
    OmsType,
    OrderSide,
    PriceType,
)
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy

from crossdesk.engine.spec import sma_crossover_targets

N = 60
rng = random.Random(5)
closes = [100.0]
for _ in range(N - 1):
    closes.append(max(1.0, closes[-1] * (1 + rng.gauss(0, 0.01))))
opens = [round(c * (1 + rng.gauss(0, 0.002)), 2) for c in closes]
closes = [round(c, 2) for c in closes]
highs = [round(max(o, c) * 1.003, 2) for o, c in zip(opens, closes)]
lows = [round(min(o, c) * 0.997, 2) for o, c in zip(opens, closes)]
targets = sma_crossover_targets(closes, 5, 20, 1.0)

instrument = TestInstrumentProvider.equity(symbol="TEST", venue="SIM")
bar_type = BarType(
    instrument_id=instrument.id,
    bar_spec=BarSpecification(1, BarAggregation.DAY, PriceType.LAST),
    aggregation_source=AggregationSource.EXTERNAL,
)

EPOCH = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
bars = []
for i in range(N):
    ts = int((EPOCH + dt.timedelta(days=i + 1)).timestamp() * 1e9)
    bars.append(Bar(
        bar_type=bar_type,
        open=Price.from_str(f"{opens[i]:.2f}"),
        high=Price.from_str(f"{highs[i]:.2f}"),
        low=Price.from_str(f"{lows[i]:.2f}"),
        close=Price.from_str(f"{closes[i]:.2f}"),
        volume=Quantity.from_str("1000000"),
        ts_event=ts,
        ts_init=ts,
    ))


class Probe(Strategy):
    def __init__(self, config=None):
        super().__init__(config or StrategyConfig())
        self.i = -1
        self.position = 0.0
        self.log_rows = []

    def on_start(self):
        self.subscribe_bars(bar_type)

    def on_bar(self, bar):
        self.i += 1
        target = targets[self.i]
        delta = target - self.position
        if delta != 0:
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            order = self.order_factory.market(
                instrument_id=instrument.id,
                order_side=side,
                quantity=Quantity.from_str(f"{abs(delta):.0f}"),
            )
            self.submit_order(order)
            self.position = target
            self.log_rows.append(("submit", self.i, float(bar.close)))

    def on_order_filled(self, event):
        self.log_rows.append(("fill", self.i, float(event.last_px)))


engine = BacktestEngine(config=BacktestEngineConfig(trader_id="PROBE-001"))
engine.add_venue(
    venue=Venue("SIM"),
    oms_type=OmsType.NETTING,
    account_type=AccountType.CASH,
    base_currency=USD,
    starting_balances=[Money(100_000, USD)],
    bar_execution=True,
)
engine.add_instrument(instrument)
engine.add_data(bars)
strategy = Probe()
engine.add_strategy(strategy)
engine.run()

print("\n" + "=" * 70)
print("WHERE DID FILLS LAND?")
print("=" * 70)
print(f"{'bar':>4} {'open[i]':>9} {'close[i]':>9} {'open[i+1]':>10} {'event':>7} {'px':>9}")
pending = None
for kind, i, px in strategy.log_rows:
    if kind == "submit":
        pending = i
    else:
        nxt = f"{opens[pending + 1]:.2f}" if pending + 1 < N else "n/a"
        print(f"{pending:>4} {opens[pending]:>9.2f} {closes[pending]:>9.2f} "
              f"{nxt:>10} {'FILL':>7} {px:>9.2f}")

fills = [(i, px) for k, i, px in strategy.log_rows if k == "fill"]
subs = [i for k, i, px in strategy.log_rows if k == "submit"]
print(f"\nsubmitted {len(subs)}   filled {len(fills)}")
if fills:
    at_close = sum(abs(px - closes[i]) < 0.005 for i, px in zip(subs, [p for _, p in fills]))
    at_next_open = sum(
        i + 1 < N and abs(px - opens[i + 1]) < 0.005
        for i, px in zip(subs, [p for _, p in fills])
    )
    print(f"fills matching DECISION bar close : {at_close}/{len(fills)}")
    print(f"fills matching NEXT bar open      : {at_next_open}/{len(fills)}")
