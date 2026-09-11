"""Three published variants of momentum, pre-registered, on CRSP top-300.

    python scripts/screen_crsp_variants.py [--top 300] [--per-side 30]

PRE-REGISTERED. The hypotheses, priors, universe, cost assumption, statistic
and thresholds below were fixed before any number was computed. Holdout: the
last 30% of month-ends (2014-07-31 onward) is SEALED and never read here.

WHY THESE THREE AND NOT A SWEEP
-------------------------------
`screen_crsp_families.py` measured plain momentum (L5) on this panel: net
+88.05 bp/mo, t = +1.65, annual Sharpe +0.34, kurtosis 10.4, leg drawdowns 61%
and 86%, and a decade decay from +168 bp in the 1990s to +22 bp in the 2010s.
Those are the symptoms of the documented momentum crash, and each variant here
is a PUBLISHED response to one of them by an author who did not see this data:

V1  RESIDUAL MOMENTUM -- Blitz, Huij & Martens (2011). Beta estimated over 36
    months, residuals from the last 12, ranked on their t-statistic. Claim:
    raw momentum's ranking is contaminated by time-varying factor exposure.
    PRIOR: POSITIVE, and a HIGHER risk-adjusted spread than raw momentum.

V2  VOLATILITY-SCALED MOMENTUM -- Barroso & Santa-Clara (2015). Scale the
    spread by target / its own trailing realised volatility. Claim: momentum's
    crashes are forecastable from its own volatility even though its returns
    are not. PRIOR: HIGHER SHARPE and LOWER kurtosis; the mean return may fall,
    and a null on mean return does NOT refute it. Registered in advance.

V3  INVERSE-VOLATILITY WEIGHTING. Weight within each leg by 1/sigma over the
    trailing 36 months instead of equally. PRIOR: LOWER volatility and a
    HIGHER Sharpe at a similar or slightly lower mean.

Nothing here is tuned. Every parameter is the published one, and the two places
this deviates from a paper are forced by the panel's resolution and are stated
as assumptions below. If a variant needs a different parameter to work, that is
a finding for the holdout, not an edit to this file.

THE COMPARISON, AND THE MISTAKE IT AVOIDS
-----------------------------------------
V1 and V3 need 36 months of history where the baseline needs 12, and V2 spends
12 months estimating volatility. They therefore run over DIFFERENT months than
the +1.65 already on record. Comparing a variant's headline against that number
would credit the variant with the sample. So:

  * every variant is reported on its own months, with n printed, AND
  * every head-to-head statistic runs on the months the pair SHARES, against a
    baseline recomputed over exactly those months.

The variants are the same strategy modified, so their monthly series are
strongly correlated and an unpaired comparison would have a standard error
several times too large. Head-to-head uses the Jobson-Korkie statistic with
Memmel's correction, checked against a paired bootstrap in
`tests/test_variants.py` because an untested standard error is an assertion.

For V1 and V3 the paired MEAN DIFFERENCE is also meaningful, since both are
one unit of a long-short book in the same units as the baseline. For V2 it is
not -- scaling changes the size of the position, so only the Sharpe comparison
is reported, and the mean-difference row says why it is absent.

THRESHOLD
---------
Three new hypotheses on a panel that has already seen six. Both bars printed:

    |t| > 2.39   Bonferroni across THIS registered family of three
    |t| > 2.77   Bonferroni across all NINE hypotheses this panel has now seen

The verdict uses 2.77. The panel does not forget the earlier six because a new
script was opened, and the stricter bar is the one this project has used every
previous time the question came up.

ASSUMPTIONS, both forced by a monthly panel
-------------------------------------------
1. (resolution) Barroso & Santa-Clara estimate volatility from 126 DAILY
   returns. There are none here, so V2 uses 12 monthly observations. That
   estimate is slower and will react to a volatility spike one to two months
   later than the published rule, which biases V2 DOWNWARD against its own
   claim rather than flattering it.
2. (benchmark) V1's market proxy is the EQUAL-WEIGHTED return of the month's
   members, because both legs of every spread here are equal-weighted, so that
   is the exposure the residual is actually being purged of.

COSTS
-----
10bp round trip per name charged on ACTUAL turnover, unchanged from the
families screen so the numbers stay comparable. The 0bp row is the gross
result and never appears without its net beside it. V2's costs scale with its
position size, so a half-sized book pays half.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crossdesk.data.crsp import build, load, mktcap_table, top_by_mktcap  # noqa: E402
from crossdesk.overfitting import (  # noqa: E402
    deflated_sharpe_ratio,
    moments,
    probabilistic_sharpe_ratio,
)
from crossdesk.power import Design, bonferroni_threshold  # noqa: E402
from crossdesk.screen.families import (  # noqa: E402
    SpreadResult,
    max_drawdown,
    monthly_stats,
)
from crossdesk.screen.variants import (  # noqa: E402
    align,
    bootstrap_sharpe_difference,
    equal_weights,
    inverse_vol_weights,
    market_series,
    monthly_returns,
    raw_momentum_score,
    residual_momentum_score,
    sharpe_difference_z,
    spread,
    volatility_scaled,
)

RULE = "=" * 78
HOLDOUT_FRACTION = 0.30
COST_BP = 10.0
COST_SWEEP = (0.0, 5.0, 10.0, 20.0, 30.0, 50.0)
LOOKBACK_MONTHS = 12
ESTIMATION_MONTHS = 36
FORMATION_MONTHS = 12
VOL_LOOKBACK = 12
TARGET_ANNUAL_VOL = 0.12
LEVERAGE_CAP = 2.0
#: history the strictest variant needs before it can rank anything
COMMON_START = ESTIMATION_MONTHS
#: three registered here, six already run on this panel
FAMILY_HERE = 3
FAMILY_CUMULATIVE = 9


def report(res: SpreadResult, prior: str, years: float, n_names: int) -> dict:
    """One variant, gross beside net, with its error bar and its path."""
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
        s = monthly_stats(res.net_bp(c))
        if s is None:
            continue
        print(f"    {c:>6.0f} {s.mean_bp:>+12.2f} {s.t:>+8.2f}")
        if breakeven is None and s.mean_bp <= 0:
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

    dd_long, dd_short = max_drawdown(res.long_bp), max_drawdown(res.short_bp)
    v_long, v_short = monthly_stats(res.long_bp), monthly_stats(res.short_bp)
    if v_long and v_short:
        print(f"\n  path of each leg (DESCRIPTIVE): max drawdown "
              f"{dd_long:.1%} vs {dd_short:.1%}   monthly vol "
              f"{v_long.sd_bp / 100:.2f}% vs {v_short.sd_bp / 100:.2f}%")
    print(f"  spread max drawdown {max_drawdown(res.net_bp(COST_BP)):.1%}")

    m = moments(res.net_bp(COST_BP))
    psr = probabilistic_sharpe_ratio(m.sharpe, m.n, m.skew, m.kurtosis)
    print(f"\n  net monthly Sharpe {m.sharpe:+.4f}  skew {m.skew:+.2f}  "
          f"kurtosis {m.kurtosis:.2f} (normal = 3)")
    print(f"  PSR  P(true Sharpe > 0) = {psr:.1%}")

    design = Design(name=res.name, years=years, n_names=n_names,
                    rebalances_per_year=12, n_tests=FAMILY_CUMULATIVE)
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


def head_to_head(variant: SpreadResult, base: SpreadResult, *,
                 comparable_means: bool, note: str) -> dict:
    """The paired comparison, on the months the two actually share."""
    xa, xb, months = align(variant, base, COST_BP)
    print(f"\n  vs baseline over the {len(months)} shared months "
          f"({months[0]} .. {months[-1]}):")
    sa, sb = monthly_stats(xa), monthly_stats(xb)
    if sa is None or sb is None:
        print("    too few shared months.")
        return {}
    print(f"    variant   {sa.mean_bp:>+8.2f} bp/mo   ann SR {sa.annual_sharpe:>+5.2f}")
    print(f"    baseline  {sb.mean_bp:>+8.2f} bp/mo   ann SR {sb.annual_sharpe:>+5.2f}")

    if comparable_means:
        diff = monthly_stats([p - q for p, q in zip(xa, xb)])
        if diff is not None:
            print(f"    paired mean difference {diff.mean_bp:>+8.2f} bp/mo  "
                  f"se {diff.se_bp:.2f}  t = {diff.t:+.2f}")
    else:
        print(f"    paired mean difference: NOT REPORTED -- {note}")

    z = sharpe_difference_z(xa, xb)
    observed, p = bootstrap_sharpe_difference(xa, xb, draws=20_000)
    rho = st.correlation(xa, xb)
    print(f"    monthly Sharpe difference {observed:+.4f}   "
          f"correlation {rho:+.3f}")
    print(f"    Jobson-Korkie/Memmel  z = {z:+.2f}      "
          f"paired bootstrap  p = {p:.3f}")
    return {"z": z, "p": p, "diff": observed, "rho": rho, "n": len(months)}


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
    builder = top_by_mktcap(caps, args.top)

    print(RULE)
    print("CRSP MOMENTUM VARIANTS -- pre-registered, in-sample only")
    print(RULE)
    print(f"universe        top {args.top} by market cap, rebalanced monthly")
    print(f"panel           {len(universe.symbols):,} PERMNOs, "
          f"{len(universe.listings):,} listing windows")
    print(f"in-sample       {len(ins)} month-ends  ({ins[0]} .. {ins[-1]})"
          f"   {years:.1f} years")
    print(f"holdout         {len(all_dates) - cut} month-ends SEALED "
          f"({all_dates[cut]} .. {all_dates[-1]})")
    print(f"threshold       |t| > {bonferroni_threshold(FAMILY_HERE):.2f} "
          f"across the {FAMILY_HERE} registered here")
    print(f"                |t| > {bonferroni_threshold(FAMILY_CUMULATIVE):.2f} "
          f"across all {FAMILY_CUMULATIVE} this panel has seen  <- the verdict "
          "uses this one")
    print(f"costs           {COST_BP:.0f}bp round trip per name, on actual turnover")
    print(f"common start    index {COMMON_START} "
          f"({ins[COMMON_START]}) -- the history V1 and V3 require")
    print()

    # ---------------------------------------------------------------- inputs
    members: set[str] = set()
    for d in ins:
        members.update(builder(universe.listings, d))
    print(f"building monthly return matrix for {len(members):,} names "
          f"ever in the top {args.top} ...")
    rets = monthly_returns(panel, sorted(members), ins)
    market = market_series(rets, builder, universe.listings, ins)
    usable = sum(1 for m in market[1:] if m is not None)
    print(f"market proxy: equal-weighted members, {usable} of {len(ins) - 1} "
          f"months priced\n")

    raw_score = raw_momentum_score(panel, ins, LOOKBACK_MONTHS)
    resid_score = residual_momentum_score(rets, market,
                                          estimation=ESTIMATION_MONTHS,
                                          formation=FORMATION_MONTHS)
    ivol = inverse_vol_weights(rets, lookback=ESTIMATION_MONTHS)
    common = dict(n_per_side=args.per_side, cost_bp=COST_BP)

    # ------------------------------------------------------------- baselines
    base_full = spread(panel, builder, ins, score_fn=raw_score,
                       weight_fn=equal_weights, start=LOOKBACK_MONTHS,
                       name="L5 baseline (registered form, full window)",
                       **common)
    base = spread(panel, builder, ins, score_fn=raw_score,
                  weight_fn=equal_weights, start=COMMON_START,
                  name="L5 baseline (common window)", **common)

    print(RULE)
    print("BASELINE, for continuity with the families screen")
    print(RULE)
    b_full = monthly_stats(base_full.net_bp(COST_BP))
    b_common = monthly_stats(base.net_bp(COST_BP))
    if b_full and b_common:
        print(f"  full window    n={b_full.n:>3}  net {b_full.mean_bp:>+8.2f} "
              f"bp/mo  t={b_full.t:>+6.2f}  ann SR {b_full.annual_sharpe:>+5.2f}"
              f"   <- reproduces the +1.65 on record")
        print(f"  common window  n={b_common.n:>3}  net {b_common.mean_bp:>+8.2f} "
              f"bp/mo  t={b_common.t:>+6.2f}  ann SR "
              f"{b_common.annual_sharpe:>+5.2f}   <- what the variants are "
              "measured against")
        print(f"\n  the {b_full.n - b_common.n} months of difference are the "
              "history V1 and V3 spend before they can rank.")
        print("  Any change in the baseline between these two rows belongs to "
              "the SAMPLE, not to a variant.")

    # V2's registered claim is about the SHAPE of the distribution, not its
    # mean, so the baseline's shape has to be on the page or the claim cannot
    # be checked. V2 runs 12 months shorter than the others, so its own
    # comparison window is printed too rather than borrowed from V1's.
    bm = moments(base.net_bp(COST_BP))
    print(f"\n  baseline path on the common window: max drawdown "
          f"{max_drawdown(base.net_bp(COST_BP)):.1%}   "
          f"skew {bm.skew:+.2f}   kurtosis {bm.kurtosis:.2f}")
    v2_window = base.net_bp(COST_BP)[VOL_LOOKBACK:]
    if len(v2_window) > 3:
        vm = moments(v2_window)
        print(f"  baseline path on V2's window:      max drawdown "
              f"{max_drawdown(v2_window):.1%}   "
              f"skew {vm.skew:+.2f}   kurtosis {vm.kurtosis:.2f}")
    print()

    out: dict[str, dict] = {}

    # --------------------------------------------------------------------- V1
    v1 = spread(panel, builder, ins, score_fn=resid_score,
                weight_fn=equal_weights, start=COMMON_START,
                name="V1 residual momentum (Blitz-Huij-Martens)", **common)
    out["V1"] = report(
        v1, "POSITIVE, and a higher risk-adjusted spread than raw momentum",
        years, args.top)
    decade_table(v1)
    out["V1"]["h2h"] = head_to_head(v1, base, comparable_means=True, note="")
    print()

    # --------------------------------------------------------------------- V2
    scaled = volatility_scaled(base, target_annual_vol=TARGET_ANNUAL_VOL,
                               lookback=VOL_LOOKBACK, cap=LEVERAGE_CAP)
    v2 = scaled.result
    out["V2"] = report(
        v2, "HIGHER SHARPE and lower kurtosis; a null on mean return does "
            "not refute it", years, args.top)
    print(f"\n  scale path: mean {scaled.mean_scale:.2f}x  max "
          f"{scaled.max_scale:.2f}x  months at the {LEVERAGE_CAP:.1f}x cap "
          f"{scaled.capped_months}  months above 1.0x {scaled.levered_months}"
          f" of {len(scaled.scales)}")
    print(f"  a scale above 1.0 is BORROWING, which the cash account this "
          f"feeds cannot do.")
    decade_table(v2)
    out["V2"]["h2h"] = head_to_head(
        v2, base, comparable_means=False,
        note="scaling changes the position size, so a difference in bp/mo "
             "would measure leverage, not skill")
    print()

    # --------------------------------------------------------------------- V3
    v3 = spread(panel, builder, ins, score_fn=raw_score, weight_fn=ivol,
                start=COMMON_START,
                name="V3 inverse-volatility weighting", **common)
    out["V3"] = report(
        v3, "LOWER volatility and a higher Sharpe at a similar mean",
        years, args.top)
    decade_table(v3)
    out["V3"]["h2h"] = head_to_head(v3, base, comparable_means=True, note="")
    print()

    # ---------------------------------------------------------------- verdict
    print(RULE)
    print("VERDICT")
    print(RULE)
    sharpes = [o["moments"].sharpe for o in out.values() if o]
    if b_common:
        sharpes.append(b_common.per_period_sharpe)
    sharpe_var = st.variance(sharpes) if len(sharpes) > 1 else 0.0
    bar = bonferroni_threshold(FAMILY_CUMULATIVE)

    print(f"  {'':<34} {'net bp/mo':>10} {'t':>7} {'ann SR':>7} "
          f"{'PSR':>7} {'DSR':>7}  verdict")
    if b_common:
        print(f"  {'L5 baseline (common window)':<34} "
              f"{b_common.mean_bp:>+10.2f} {b_common.t:>+7.2f} "
              f"{b_common.annual_sharpe:>+7.2f} {'':>7} {'':>7}  reference, "
              "not a new trial")
    for key, o in out.items():
        if not o:
            continue
        m, net = o["moments"], o["net"]
        dsr = deflated_sharpe_ratio(m.sharpe, m.n, FAMILY_CUMULATIVE,
                                    sharpe_var, m.skew, m.kurtosis)
        verdict = "CLEARS" if abs(net.t) > bar else f"fails {bar:.2f}"
        name = {"V1": "V1 residual momentum",
                "V2": "V2 volatility-scaled",
                "V3": "V3 inverse-vol weighting"}[key]
        print(f"  {name:<34} {net.mean_bp:>+10.2f} {net.t:>+7.2f} "
              f"{net.annual_sharpe:>+7.2f} {o['psr']:>6.1%} {dsr:>6.1%}  "
              f"{verdict}")

    print(f"\n  DSR deflates against the expected best of {FAMILY_CUMULATIVE} "
          f"trials using a Sharpe variance of {sharpe_var:.6f}, estimated from "
          f"the {len(sharpes)} series here.")
    print("  That estimate is the weakest input in the procedure, and these "
          "series are strongly")
    print("  correlated modifications of one strategy, so their disagreement "
          "understates the")
    print("  disagreement across genuinely independent trials. DSR is a "
          "cross-check here, not the verdict.")
    print("\n  The head-to-head z above is the statistic that answers the "
          "question actually asked:")
    print("  not 'does this variant make money' but 'is it BETTER than the "
          "momentum already measured'.")
    print("\n  Holdout untouched. Everything above is in-sample and stays that "
          "way until a")
    print("  pre-registered out-of-sample test is written for whatever "
          "survived, if anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
