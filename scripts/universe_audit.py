"""Run the survivorship guards against the real ETF panel, and report breadth.

Reads only. Fits nothing, trades nothing, and touches no holdout -- it describes
the data the cross-sectional programme would run on, and checks that the panel
can survive its own guards.

Run: python scripts/universe_audit.py
"""

from crossdesk.data.etf_panel import load_panel, month_ends
from crossdesk.data.panel import assert_no_silent_dropout
from crossdesk.data.universe import (
    assert_delistings_priced,
    assert_membership_is_causal,
    assert_no_survivorship,
)
from crossdesk.power import bonferroni_threshold, ic_required, power, sharpe_from_ic

RULE = "=" * 74
HOLDOUT_FRACTION = 0.30       # matches every screen in the quantdesk repo


def honest_builder(listings, asof):
    return [l.symbol for l in listings if l.is_active(asof)]


def main() -> None:
    panel = load_panel()
    universe = panel.universe
    rebalances = month_ends(panel)

    print(RULE)
    print("UNIVERSE AUDIT -- ETF daily panel")
    print(RULE)
    print(f"tickers ever listed   {len(universe.symbols)}")
    print(f"trading dates         {len(panel.dates):,}"
          f"  ({panel.dates[0]} .. {panel.dates[-1]})")
    print(f"month-end rebalances  {len(rebalances)}")
    print()

    print("Guards:")
    assert_delistings_priced(universe)
    print("  [ok] every ended listing carries a terminal return"
          "  (trivially: none has ended)")

    assert_membership_is_causal(honest_builder, universe.listings, rebalances)
    print(f"  [ok] membership is causal on all {len(rebalances)} rebalance dates")

    assert_no_survivorship({d: universe.members_asof(d) for d in rebalances},
                           universe)
    print("  [ok] no date claims a symbol that was not tradable")

    assert_no_silent_dropout(panel, rebalances)
    print("  [ok] no member leaves the cross-section unpriced")
    print()

    print("Breadth over time (members at each year end):")
    seen_years = {}
    for d in rebalances:
        seen_years[d.year] = universe.size_asof(d)
    years = sorted(seen_years)
    for i in range(0, len(years), 5):
        chunk = years[i:i + 5]
        print("  " + "   ".join(f"{y}: {seen_years[y]:>2}" for y in chunk))
    print()

    counts = {"traded": 0, "delisted": 0, "missing": 0}
    for a, b in zip(rebalances, rebalances[1:]):
        for key, value in panel.provenance(a, b).items():
            counts[key] += value
    total = sum(counts.values())
    print("Return provenance across all monthly periods:")
    for key, value in counts.items():
        print(f"  {key:<9} {value:>7,}  ({value / total:>6.2%})")
    print()

    cut = int(len(rebalances) * (1 - HOLDOUT_FRACTION))
    print(f"in-sample  {cut} rebalances ({rebalances[0]} .. {rebalances[cut - 1]})")
    print(f"holdout    {len(rebalances) - cut} rebalances SEALED "
          f"({rebalances[cut]} .. {rebalances[-1]})")
    print()

    print(RULE)
    print("WHAT THIS PANEL CAN DETECT")
    print(RULE)
    n = universe.size_asof(rebalances[cut - 1])
    years = cut / 12
    t = bonferroni_threshold(1)
    sr = sharpe_from_ic(0.03, n, 12)
    print(f"names at the in-sample cut   {n}")
    print(f"in-sample years              {years:.1f}")
    print(f"IC required to clear |t|>{t:.2f}   {ic_required(t, years, n, 12):.4f}")
    print(f"Sharpe at a realistic IC=0.03      {sr:.2f}")
    print(f"power against that Sharpe          {power(sr, years, t):.1%}")
    print()
    print("Read this as the ceiling on the current data, not a forecast. Even")
    print("uncorrected, and even before costs, fifteen names cannot support a")
    print("cross-sectional test. The data layer is now correct; it is not yet")
    print("wide enough. See results/power_audit.txt for the breadth table.")


if __name__ == "__main__":
    main()
