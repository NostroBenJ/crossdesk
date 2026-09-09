"""Cross-engine agreement on real SPY daily bars, and what the convention costs.

Runs one precomputed signal through the reference spec, quantdesk, VectorBT and
NautilusTrader, and prints the maximum divergence of each against the reference.
Then measures what the fill convention alone is worth -- the number that decides
whether Nautilus's bar-mode results can be compared to quantdesk's at all.

**The holdout is SEALED.** In-sample bars only, matching every screen in the
quantdesk repo. This is an instrument check, not a strategy result.

Run: python scripts/engine_agreement.py
"""

import csv
import math
import os

from crossdesk.engine.adapters import (
    convention_gap,
    mask_warmup,
    quantdesk_warmup,
    run_nautilus,
    run_quantdesk,
    run_reference,
    run_vectorbt,
)
from crossdesk.engine.spec import ExecutionSpec, sma_crossover_targets

RULE = "=" * 74
CASH = 100_000.0
HOLDOUT_FRACTION = 0.30
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "..", "quantdesk", "etf_daily.csv")


def load_spy():
    opens, closes, dates = [], [], []
    with open(CSV, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["ticker"] != "SPY" or not row["close"]:
                continue
            dates.append(row["date"])
            opens.append(float(row["open"]))
            closes.append(float(row["close"]))
    return dates, opens, closes


def main() -> None:
    dates, opens, closes = load_spy()
    cut = int(len(closes) * (1 - HOLDOUT_FRACTION))
    dates, opens, closes = dates[:cut], opens[:cut], closes[:cut]

    warmup = quantdesk_warmup()
    # Size against the MAXIMUM close in the sample, not the first.
    #
    # Sizing off the first close (759 shares at $131) makes the position cost
    # $266k once SPY reaches $350 -- more than the account holds. VectorBT and
    # Nautilus then enforce the cash constraint and partially fill, while the
    # spec and quantdesk let cash go negative. The engines diverge by $11,776
    # and the cause is nothing to do with fill timing. Sizing so the position is
    # always affordable keeps this test measuring fills, which is its job. The
    # cash-constraint divergence is reported separately below.
    size = float(int(CASH / max(closes)))
    targets = mask_warmup(sma_crossover_targets(closes, 20, 100, size), warmup)
    spec = ExecutionSpec(initial_cash=CASH)

    print(RULE)
    print("CROSS-ENGINE AGREEMENT -- SPY daily, 20/100 SMA, frictionless")
    print(RULE)
    print(f"bars       {len(closes):,}  ({dates[0]} .. {dates[-1]})")
    print(f"holdout    SEALED, not read")
    print(f"position   {size:,.0f} shares (~${size*closes[0]:,.0f} at the first bar)")
    print(f"trades     {sum(a != b for a, b in zip(targets, targets[1:]))}")
    print()

    reference = run_reference(opens, closes, targets, spec, start=warmup)
    quant = run_quantdesk(opens, closes, targets, spec)
    vector = run_vectorbt(opens, closes, targets, spec)[warmup:]

    print(f"{'engine':<16} {'bars':>6} {'final equity':>15} {'max diff vs ref':>17}")
    print(f"{'-'*16} {'-'*6} {'-'*15} {'-'*17}")
    print(f"{'reference spec':<16} {len(reference):>6} {reference[-1]:>15,.4f} {'--':>17}")
    for name, curve in (("quantdesk", quant), ("vectorbt", vector)):
        diff = max(abs(a - b) for a, b in zip(curve, reference))
        print(f"{name:<16} {len(curve):>6} {curve[-1]:>15,.4f} {diff:>17.9f}")

    close_spec = ExecutionSpec(initial_cash=CASH, fill_at="decision_close")
    naut = run_nautilus(opens, closes, targets, close_spec)[warmup:]
    close_ref = run_reference(opens, closes, targets, close_spec, start=warmup)
    naut_diff = max(abs(a - b) for a, b in zip(naut, close_ref))
    print(f"{'nautilus':<16} {len(naut):>6} {naut[-1]:>15,.4f} "
          f"{naut_diff:>17.9f}  (vs decision_close ref)")
    print()

    next_open, decision_close, gap_bps = convention_gap(opens, closes, targets, CASH)
    years = len(closes) / 252
    a_cagr = (next_open / CASH) ** (1 / years) - 1
    b_cagr = (decision_close / CASH) ** (1 / years) - 1

    print(RULE)
    print("WHAT THE FILL CONVENTION IS WORTH")
    print(RULE)
    print(f"{'fill at next open (quantdesk, vectorbt)':<44} "
          f"{next_open:>13,.2f}   {a_cagr:>7.2%}/yr")
    print(f"{'fill at decision close (nautilus bar mode)':<44} "
          f"{decision_close:>13,.2f}   {b_cagr:>7.2%}/yr")
    print(f"{'gap':<44} {decision_close - next_open:>13,.2f}   "
          f"{gap_bps:>7.1f} bps")
    print(f"{'gap in CAGR':<44} {'':>13}   {b_cagr - a_cagr:>7.2%}/yr")
    print()
    print("Same strategy, same bars, same costs. The difference is entirely the")
    print("bar at which an order is assumed to fill. Nautilus's convention is the")
    print("optimistic one -- it acts on a close at that same close, which needs")
    print("the close before it prints.")
    print()
    print(RULE)
    print("SECOND DIVERGENCE: THE CASH CONSTRAINT")
    print(RULE)
    big = float(int(CASH / closes[0]))
    big_targets = mask_warmup(sma_crossover_targets(closes, 20, 100, big), warmup)
    big_ref = run_reference(opens, closes, big_targets, spec, start=warmup)
    big_vbt = run_vectorbt(opens, closes, big_targets, spec)[warmup:]
    drift = max(abs(a - b) for a, b in zip(big_vbt, big_ref))
    print(f"sizing {big:,.0f} shares off the FIRST close costs "
          f"${big * max(closes):,.0f} at the sample high")
    print(f"  reference / quantdesk  {big_ref[-1]:>13,.2f}   cash may go negative")
    print(f"  vectorbt               {big_vbt[-1]:>13,.2f}   partially fills instead")
    print(f"  max divergence         {drift:>13,.2f}")
    print()
    print("VectorBT and Nautilus refuse to spend cash they do not have. quantdesk")
    print("does not model the constraint at all, so a backtest there can hold a")
    print("position it could never have funded, financed at zero cost. That is a")
    print("gap in quantdesk, not in VectorBT, and it is invisible until position")
    print("size approaches account size.")
    print()
    print("So: quantdesk and VectorBT are interchangeable and agree to floating")
    print("point WHEN THE CASH CONSTRAINT DOES NOT BIND. Nautilus in BAR mode is")
    print("not comparable to either; use it for execution and live, or feed it")
    print("quote/trade ticks so the fill is real rather than a convention.")


if __name__ == "__main__":
    main()
