"""Run the survivorship guards against the CRSP monthly panel, and report power.

Reads only. Fits nothing, trades nothing, and touches no holdout. The
counterpart of `universe_audit.py` for the universe that was supposed to
replace the fifteen ETFs: it answers whether the data layer still holds on
tens of thousands of listings, how much of the cross-section is delisted at
any time, and what the design can now detect.

Run: python scripts/crsp_audit.py [--top 500]
"""

from __future__ import annotations

import argparse
import sys
import time

from crossdesk.data.crsp import build, load, mktcap_table, top_by_mktcap
from crossdesk.data.panel import assert_no_silent_dropout
from crossdesk.data.universe import (
    assert_delistings_priced,
    assert_membership_is_causal,
    assert_no_survivorship,
)
from crossdesk.power import Design, ic_required, sharpe_from_ic

RULE = "=" * 74
HOLDOUT_FRACTION = 0.30
#: Published gross Sharpe for cross-sectional equity momentum.
LITERATURE_SHARPE = 0.40


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=500, help="size-ranked universe width")
    ap.add_argument("--every", type=int, default=12, help="check guards every Nth month-end")
    args = ap.parse_args()

    t0 = time.time()
    data = load()
    universe, panel, exits = build(data)
    caps = mktcap_table(data)
    rebalances = list(data.dates)
    print(RULE)
    print("UNIVERSE AUDIT -- CRSP monthly, ordinary common shares on NYSE/AMEX/NASDAQ")
    print(RULE)
    print(f"PERMNOs ever listed     {len(universe.symbols):,}")
    print(f"listing windows         {len(universe.listings):,}  "
          f"({sum(1 for l in universe.listings if l.end is not None):,} ended)")
    print(f"month-ends              {len(rebalances)}  ({rebalances[0]} .. {rebalances[-1]})")
    print(f"Shumway-adjusted exits  {data.shumway_adjusted:,}  (delisting code, no delisting return)")
    codes = {}
    for code, _present in exits.values():
        band = "none (filter exit)" if code < 200 else f"{code // 100}00s"
        codes[band] = codes.get(band, 0) + 1
    print("window ends by CRSP delisting code:  " + "  ".join(f"{k}: {v:,}" for k, v in sorted(codes.items())))
    print(f"loaded in {time.time() - t0:.0f}s\n")

    print("Guards:")
    assert_delistings_priced(universe)
    print("  [ok] every ended listing carries a terminal return")

    sample = rebalances[::args.every]
    builder = top_by_mktcap(caps, args.top)
    assert_membership_is_causal(builder, universe.listings, sample)
    print(f"  [ok] top-{args.top}-by-size membership is causal on {len(sample)} sampled dates")

    members = {d: builder(universe.listings, d) for d in sample}
    assert_no_survivorship(members, universe)
    print("  [ok] no date claims a PERMNO that was not tradable")

    # Dropout on the FULL universe is the expensive check; sample it too.
    assert_no_silent_dropout(panel, sample)
    print(f"  [ok] no member leaves the cross-section unpriced ({len(sample) - 1} periods checked)\n")

    print(f"Breadth (all eligible / top-{args.top}) at year ends:")
    rows = []
    for d in rebalances:
        if d.month == 12:
            rows.append((d.year, universe.size_asof(d), len(builder(universe.listings, d))))
    for i in range(0, len(rows), 4):
        print("  " + "   ".join(f"{y}: {a:>5,}/{b:>3}" for y, a, b in rows[i:i + 4]))
    print()

    counts = {"traded": 0, "delisted": 0, "missing": 0}
    for a, b in zip(sample, sample[1:]):
        for key, value in panel.provenance(a, b).items():
            counts[key] += value
    total = sum(counts.values())
    print(f"Return provenance over the sampled {len(sample) - 1} periods (full universe):")
    for key, value in counts.items():
        print(f"  {key:<9} {value:>8,}  ({value / total:>6.2%})")
    print()

    cut = int(len(rebalances) * (1 - HOLDOUT_FRACTION))
    print(f"in-sample  {cut} month-ends ({rebalances[0]} .. {rebalances[cut - 1]})")
    print(f"holdout    {len(rebalances) - cut} month-ends SEALED ({rebalances[cut]} .. {rebalances[-1]})\n")

    print(RULE)
    print("WHAT THIS PANEL CAN DETECT")
    print(RULE)
    years = cut / 12
    print("  Two columns, because they answer different questions and only one"
          " of them has been checked:")
    print("    'law'  power against the Sharpe the Fundamental Law predicts "
          "from IC=0.03 at 0.25 haircut")
    print("    'lit'  power against a published cross-sectional Sharpe of 0.40")
    print()
    print(f"  {'universe':>10} {'IC needed':>10} {'SR (law)':>9} {'power':>7}   "
          f"{'SR (lit)':>9} {'power':>7}   verdict (lit)")
    for n in (100, 300, args.top, 1000):
        d = Design(name=f"top{n}", years=years, n_names=n, rebalances_per_year=12, n_tests=6)
        sr = sharpe_from_ic(0.03, n, 12)
        print(f"  {'top ' + str(n):>10} {ic_required(d.t_threshold, years, n, 12):>10.4f} "
              f"{sr:>9.2f} {d.power_against(sr):>7.1%}   "
              f"{LITERATURE_SHARPE:>9.2f} {d.power_against(LITERATURE_SHARPE):>7.1%}   "
              f"{d.verdict(LITERATURE_SHARPE)}")
    print("\n  Six pre-registered families, Bonferroni, in-sample years as above.")
    print("  Compare with results/universe_audit.txt: 15 names, 11.7% power.")
    print()
    print("  READ THE 'lit' COLUMN, NOT THE 'law' ONE. The law column assumes")
    print("  names x rebalances x 0.25 INDEPENDENT bets. Measured on this panel")
    print("  in results/screen_crsp_families.txt, a 30-name-per-side momentum")
    print("  spread realised an annual Sharpe of 0.36, not the 0.90 the law")
    print("  predicts -- an implied haircut near 0.04. Thirty longs picked by one")
    print("  signal are one factor bet, not thirty, so the law's breadth is the")
    print("  optimistic bound and the literature column is the honest one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
