"""The library families re-run on CRSP, at the breadth the power audit demanded.

    python scripts/screen_crsp_families.py [--top 300] [--per-side 30]

PRE-REGISTERED. The hypotheses, the universe rule, the cost assumption and the
threshold below were fixed before any statistic was computed. Holdout: the last
30% of month-ends (2014-07-31 onward, 126 of 420) is SEALED and never read.

WHY TWO FAMILIES AND NOT SIX
----------------------------
`scripts/screen_library.py` registered six. Four of them do not transfer to a
monthly equity panel, and the reasons are two different things that both have
to be said:

  L1 overnight vs intraday   needs close-to-open and open-to-close legs
  L2 turn of the month       needs daily sessions
  L3 pre-holiday             needs daily sessions
  L4 VIX -> forward return   one market series, no cross-section

The resolution objection is obvious. The second objection matters more: none of
those four is BREADTH-limited. Each is a single time series whose power comes
from the number of sessions, and ranking three hundred names adds nothing to
it. Re-running them here would not be the same experiment at higher power, it
would be the same experiment at the same power with a longer runtime. Their
verdicts from the ETF screen stand as they were, and the honest description of
three of them remains UNDERPOWERED.

  L5 sector momentum   IS cross-sectional. It is the family the power audit
                       named at 12.7%, and top-3-of-9-sectors against bottom-3
                       generalises directly to a decile spread over 300 stocks.
  L6 trend following   is a per-name rule pooled across names. Breadth cancels
                       idiosyncratic noise; the independent observations are
                       still the MONTHS, so the gain is smaller than L5's.

THE HYPOTHESES, unchanged in form from the registered versions
--------------------------------------------------------------
L5  Rank the top-N-by-market-cap universe on total return over the previous 12
    months. Long the best `per_side`, short the worst, equal weight, held one
    month, rebalanced monthly and non-overlapping. Prior: the spread is
    POSITIVE. Published cross-sectional momentum Sharpes are 0.4-0.6 gross.

L6  Hold each name only when its total-return index is above its own 10-month
    average, against holding it unconditionally. Prior on RETURN is weak: the
    published claim for this rule is drawdown reduction, so a null on return
    does not refute it -- stated in advance, as it was the first time.

THE STATISTIC, and the mistake it exists to avoid
-------------------------------------------------
Both tests produce one number per name per month. A t-test over pooled
name-months would treat three hundred co-moving stocks as three hundred
independent draws and overstate significance by a factor of sqrt(names) --
measured exactly in `tests/test_families.py`. Everything here is therefore the
MONTHLY PORTFOLIO series: names averaged first, one observation per month, and
those observations are non-overlapping so they need no further correction.

THRESHOLD
---------
Bonferroni across the registered family of SIX: |t| > 2.64. Running two of the
six does not license the easier two-test bar of 2.24, because the six were
registered together and the choice of which two survive was made on data
availability rather than on results. Both bars are printed; the verdict uses
2.64.

COSTS
-----
ASSUMPTION (microstructure): 10bp round trip per name for top-300 US stocks --
a few bp of spread plus impact, charged on ACTUAL monthly turnover rather than
assumed to be full. The sweep reports the breakeven multiple, and the 0bp row
is the gross result, which never appears without its net beside it.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crossdesk.data.crsp import build, load, mktcap_table, top_by_mktcap  # noqa: E402
from crossdesk.overfitting import (  # noqa: E402
    deflated_sharpe_ratio,
    moments,
    probabilistic_sharpe_ratio,
)
from crossdesk.power import Design, bonferroni_threshold, sharpe_from_ic  # noqa: E402
from crossdesk.screen.families import (  # noqa: E402
    SpreadResult,
    max_drawdown,
    cross_sectional_momentum,
    monthly_stats,
    pooled_t,
    trend_following,
)

RULE = "=" * 78
HOLDOUT_FRACTION = 0.30
REGISTERED_FAMILY = 6
COST_BP = 10.0
COST_SWEEP = (0.0, 5.0, 10.0, 20.0, 30.0, 50.0)
LOOKBACK_MONTHS = 12
MA_MONTHS = 10


def report(res: SpreadResult, prior: str, years: float, n_names: int) -> dict:
    """Everything one family supports, gross beside net, with its error bar."""
    print(RULE)
    print(res.name)
    print(RULE)
    print(f"  prior (registered in advance): {prior}")

    gross = monthly_stats(res.gross_bp)
    net = monthly_stats(res.net_bp(COST_BP))
    if gross is None or net is None:
        print("  too few months to say anything.")
        return {}

    print(f"\n  months {res.n}   mean names ranked {res.mean_ranked:,.0f}"
          f"   mean cost basis {sum(res.cost_fraction) / max(res.n, 1):.2f}"
          f"   unscored name-months {res.unscored:,}")
    prov = res.provenance
    total_prov = sum(prov.values()) or 1
    print("  held-leg provenance: " + "  ".join(
        f"{k} {v:,} ({v / total_prov:.2%})" for k, v in sorted(prov.items())))
    if res.unpriced:
        print(f"  UNPRICED name-months excluded: {res.unpriced:,}")

    print(f"\n  {'':<8} {'mean bp/mo':>11} {'se':>8} {'t':>8} "
          f"{'95% CI':>22} {'ann %':>8} {'ann SR':>8}")
    for label, s in (("gross", gross), ("net", net)):
        print(f"  {label:<8} {s.mean_bp:>+11.2f} {s.se_bp:>8.2f} {s.t:>+8.2f} "
              f"  [{s.lo_bp:>+8.2f}, {s.hi_bp:>+8.2f}] {s.annual_return_pct:>+8.2f}"
              f" {s.annual_sharpe:>+8.2f}")

    print(f"\n  legs: long {sum(res.long_bp) / res.n:+.2f} bp/mo   "
          f"short {sum(res.short_bp) / res.n:+.2f} bp/mo")

    print("\n  cost sensitivity (round trip per name, charged on turnover):")
    print(f"    {'bp':>6} {'mean bp/mo':>12} {'t':>8}")
    breakeven = None
    for c in COST_SWEEP:
        st_ = monthly_stats(res.net_bp(c))
        if st_ is None:
            continue
        print(f"    {c:>6.0f} {st_.mean_bp:>+12.2f} {st_.t:>+8.2f}")
        if breakeven is None and st_.mean_bp <= 0:
            breakeven = c
    if breakeven is None:
        print(f"    survives every cost tested (up to {COST_SWEEP[-1]:.0f}bp)")
    elif breakeven == 0.0:
        print("    negative before any cost is charged -- costs are not what "
              "is wrong with it")
    else:
        print(f"    edge reaches zero between {breakeven - 5:.0f} and "
              f"{breakeven:.0f}bp  ->  breakeven multiple "
              f"{breakeven / COST_BP:.1f}x the {COST_BP:.0f}bp assumption")

    # What pooling would have claimed, for the record.
    k = max(int(round(res.mean_ranked)), 1)
    inflated = pooled_t([r for r in res.gross_bp for _ in range(k)])
    print(f"\n  the wrong statistic, for scale: pooling {k} names x {res.n} "
          f"months as independent draws would report t = {inflated:+.2f}")

    dd_long = max_drawdown(res.long_bp)
    dd_short = max_drawdown(res.short_bp)
    vol_long = monthly_stats(res.long_bp)
    vol_short = monthly_stats(res.short_bp)
    if vol_long and vol_short:
        print(f"\n  path of each leg (DESCRIPTIVE): max drawdown "
              f"{dd_long:.1%} vs {dd_short:.1%}   "
              f"monthly vol {vol_long.sd_bp / 100:.2f}% vs "
              f"{vol_short.sd_bp / 100:.2f}%")

    m = moments(res.net_bp(COST_BP))
    psr = probabilistic_sharpe_ratio(m.sharpe, m.n, m.skew, m.kurtosis)
    print(f"\n  net monthly Sharpe {m.sharpe:+.4f}  skew {m.skew:+.2f}  "
          f"kurtosis {m.kurtosis:.2f} (normal = 3)")
    print(f"  PSR  P(true Sharpe > 0) = {psr:.1%}")

    design = Design(name=res.name, years=years, n_names=n_names,
                    rebalances_per_year=12, n_tests=REGISTERED_FAMILY)
    print(f"  power against a literature Sharpe of 0.40: "
          f"{design.power_against(0.40):.1%}  -> {design.verdict(0.40)}")
    return {"gross": gross, "net": net, "moments": m, "psr": psr}


def decade_table(res: SpreadResult) -> None:
    """Descriptive only. No hypothesis is tested here and none is claimed."""
    buckets: dict[int, list[float]] = {}
    for d, v in zip(res.months, res.net_bp(COST_BP)):
        buckets.setdefault(d.year // 10 * 10, []).append(v)
    print("\n  by decade (DESCRIPTIVE -- not a test, and not corrected):")
    for decade in sorted(buckets):
        s = monthly_stats(buckets[decade])
        if s is None:
            continue
        print(f"    {decade}s  n={s.n:>3}  net {s.mean_bp:>+8.2f} bp/mo  "
              f"t={s.t:>+6.2f}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--per-side", type=int, default=30)
    args = ap.parse_args()

    data = load()
    universe, panel, _exits = build(data)
    caps = mktcap_table(data)
    all_dates = list(data.dates)
    cut = int(len(all_dates) * (1 - HOLDOUT_FRACTION))
    ins = all_dates[:cut]
    years = len(ins) / 12.0

    print(RULE)
    print("CRSP LIBRARY FAMILIES -- pre-registered, in-sample only")
    print(RULE)
    print(f"universe        top {args.top} by market cap, rebalanced monthly")
    print(f"panel           {len(universe.symbols):,} PERMNOs, "
          f"{len(universe.listings):,} listing windows")
    print(f"in-sample       {len(ins)} month-ends  ({ins[0]} .. {ins[-1]})"
          f"   {years:.1f} years")
    print(f"holdout         {len(all_dates) - cut} month-ends SEALED "
          f"({all_dates[cut]} .. {all_dates[-1]})")
    print(f"threshold       |t| > {bonferroni_threshold(REGISTERED_FAMILY):.2f} "
          f"(Bonferroni across the registered family of {REGISTERED_FAMILY})")
    print(f"                |t| > {bonferroni_threshold(2):.2f} would be the bar "
          "for the two actually run; the stricter one is used")
    print(f"costs           {COST_BP:.0f}bp round trip per name, on actual turnover")
    print()

    builder = top_by_mktcap(caps, args.top)
    out: dict[str, dict] = {}

    l5 = cross_sectional_momentum(
        panel, builder, ins, n_per_side=args.per_side,
        lookback_months=LOOKBACK_MONTHS, cost_bp=COST_BP)
    out["L5"] = report(
        l5, f"POSITIVE: top {args.per_side} minus bottom {args.per_side} "
            f"on {LOOKBACK_MONTHS}-month return", years, args.top)
    decade_table(l5)
    print()

    l6 = trend_following(panel, builder, ins, ma_months=MA_MONTHS, cost_bp=COST_BP)
    out["L6"] = report(
        l6, f"WEAK: the published claim for the {MA_MONTHS}-month rule is "
            "drawdown reduction, not return", years, args.top)
    decade_table(l6)
    print()

    # ------------------------------------------------------- the breadth check
    print(RULE)
    print("DID THE BREADTH FIX DELIVER WHAT THE POWER AUDIT PREDICTED?")
    print(RULE)
    predicted = sharpe_from_ic(0.03, args.top, 12)
    l5_gross = out["L5"]["gross"] if out.get("L5") else None
    if l5_gross is not None:
        measured = l5_gross.annual_sharpe
        implied_ic = measured / math.sqrt(args.top * 12 * 0.25)
        implied_haircut = (measured / 0.03) ** 2 / (args.top * 12)
        print(f"  predicted annual Sharpe at IC=0.03, {args.top} names, "
              f"haircut 0.25   {predicted:>6.2f}")
        print(f"  measured annual Sharpe, L5 gross                            "
              f"    {measured:>6.2f}")
        print(f"  -> implied IC at the assumed 0.25 haircut                   "
              f"    {implied_ic:>6.4f}")
        print(f"  -> implied haircut at the assumed IC of 0.03                "
              f"    {implied_haircut:>6.3f}")
        print()
        print("  The Fundamental Law counts INDEPENDENT bets. Thirty longs and "
              "thirty shorts chosen by")
        print("  one signal are not sixty bets, they are one factor bet held "
              "sixty ways, and the monthly")
        print("  series is what the design actually gets to average over. The "
              "0.25 haircut in power.py")
        print("  is far too generous for a cross-sectional factor; measured "
              "here it is nearer 0.04.")
        print()
        print(f"  Consequence for the audit: `results/crsp_audit.txt` reports "
              f"96.5% power for {args.top} names,")
        print("  which is power against a Sharpe of 0.90 that this universe "
              "does not produce. Power against")
        print("  the LITERATURE effect of 0.40 is what the family reports "
              "above, and it is far lower.")
        print(f"  Minimum detectable annual Sharpe at |t|>{bonferroni_threshold(REGISTERED_FAMILY):.2f} "
              f"over {years:.1f} years: "
              f"{bonferroni_threshold(REGISTERED_FAMILY) / math.sqrt(years):.2f}.")
    print()

    # ---------------------------------------------------------------- verdict
    thr = bonferroni_threshold(REGISTERED_FAMILY)
    print(RULE)
    print("VERDICT")
    print(RULE)
    sharpes = [o["moments"].sharpe for o in out.values() if o]
    var_sharpe = (sum((x - sum(sharpes) / len(sharpes)) ** 2 for x in sharpes)
                  / len(sharpes)) if len(sharpes) > 1 else 0.0
    survivors = []
    for key, o in out.items():
        if not o:
            continue
        net = o["net"]
        clears = abs(net.t) >= thr
        nominal = abs(net.t) >= 1.96
        dsr = deflated_sharpe_ratio(
            o["moments"].sharpe, o["moments"].n, REGISTERED_FAMILY,
            var_sharpe, o["moments"].skew, o["moments"].kurtosis)
        print(f"  {key}  net {net.mean_bp:>+8.2f} bp/mo  t={net.t:>+6.2f}  "
              f"ann SR {net.annual_sharpe:>+5.2f}  PSR {o['psr']:>6.1%}  "
              f"DSR {dsr:>6.1%}   {'CLEARS' if clears else 'fails'} "
              f"{thr:.2f}{'' if nominal else '  (and fails an uncorrected 1.96)'}")
        if clears:
            survivors.append(key)

    print("\n  DSR here deflates against the expected best of "
          f"{REGISTERED_FAMILY} trials using a Sharpe variance of "
          f"{var_sharpe:.6f}, estimated from the {len(sharpes)} families this "
          "screen ran.")
    print("  That estimate is the weakest input in the procedure and it is "
          "stated rather than buried: with two trials it is barely an estimate.")
    print("  It is also not the primary correction. Nothing was SELECTED here "
          "-- both hypotheses were")
    print("  registered before the data existed -- so Bonferroni across the "
          "family is the instrument that")
    print("  applies, and DSR is reported as a cross-check, not as the verdict.")

    if survivors:
        print(f"\n  Clears the corrected threshold: {', '.join(survivors)}")
    else:
        print("\n  Nothing clears the corrected threshold.")
    print("\n  Holdout untouched. Anything above is in-sample and stays that "
          "way until a")
    print("  pre-registered out-of-sample test is written for whatever "
          "survived, if anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
