"""What could the six-family library screen actually have detected?

Reads nothing and fits nothing. It re-describes the SHAPE of a screen that has
already been run -- 3,521 in-sample sessions, 15 ETFs, six pre-registered
families, |t| > 2.64 -- and asks what effect size that shape could have found.

Run: python scripts/power_audit.py
"""

from crossdesk.power import (
    Design, bonferroni_threshold, ic_required, min_detectable_sharpe,
    names_required, power, sharpe_from_ic, years_required,
)

IN_SAMPLE_SESSIONS = 3_521
YEARS = IN_SAMPLE_SESSIONS / 252
N_TESTS = 6

# Published cross-sectional/sector momentum Sharpes cluster here, GROSS.
# Sourced from the strategy's own literature, not from this data.
LITERATURE = {
    "sector momentum (cross-sectional)": 0.40,
    "time-series momentum / trend": 0.50,
    "turn of the month": 0.30,
    "overnight vs intraday": 0.60,
}

RULE = "=" * 74


def main() -> None:
    t = bonferroni_threshold(N_TESTS)
    print(RULE)
    print("POWER AUDIT -- the six-family library screen")
    print(RULE)
    print(f"in-sample sessions   {IN_SAMPLE_SESSIONS:,}  ({YEARS:.1f} years)")
    print(f"families tested      {N_TESTS}")
    print(f"corrected threshold  |t| > {t:.2f}")
    print(f"uncorrected          |t| > {bonferroni_threshold(1):.2f}")
    print()
    print(f"minimum detectable Sharpe (corrected)    {min_detectable_sharpe(YEARS, t):.3f}")
    print(f"minimum detectable Sharpe (uncorrected)  "
          f"{min_detectable_sharpe(YEARS, bonferroni_threshold(1)):.3f}")
    print()
    print("A design's 'minimum detectable' effect is one it finds HALF the time.")
    print("Power against a literature-strength effect is the honest number:")
    print()
    print(f"  {'hypothesis':<36} {'lit SR':>7} {'power':>7}  verdict")
    print(f"  {'-'*36} {'-'*7} {'-'*7}  {'-'*12}")
    for name, sr in LITERATURE.items():
        p = power(sr, YEARS, t)
        d = Design(name, YEARS, 15, 12, N_TESTS)
        print(f"  {name:<36} {sr:>7.2f} {p:>7.1%}  {d.verdict(sr)}")
    print()
    print("Years of data needed for 80% power at |t| > 2.64:")
    for name, sr in LITERATURE.items():
        print(f"  {name:<36} {years_required(sr, t):>6.0f} years")
    print()
    print(RULE)
    print("THE BREADTH FIX")
    print(RULE)
    print("More years are unavailable. More names are not.")
    print("IR = IC * sqrt(breadth), breadth = names * rebalances * haircut(0.25).")
    print()
    print(f"  {'universe':>10} {'IC needed':>11} {'SR at IC=0.03':>15} {'power':>8}")
    print(f"  {'-'*10} {'-'*11} {'-'*15} {'-'*8}")
    for n in (15, 50, 100, 250, 500, 1000):
        ic = ic_required(t, YEARS, n, 12)
        sr = sharpe_from_ic(0.03, n, 12)
        print(f"  {n:>10} {ic:>11.4f} {sr:>15.2f} {power(sr, YEARS, t):>8.1%}")
    print()
    for ic in (0.02, 0.03, 0.05):
        n = names_required(ic, YEARS, t, rebalances_per_year=12)
        print(f"  names needed for 80% power at IC={ic:.2f}:  {n:>7.0f}")
    print()
    print("Read the IC column as: the per-bet skill this design demands before")
    print("it can see anything. At 15 names it demands more skill per bet than")
    print("any published cross-sectional signal has ever shown.")


if __name__ == "__main__":
    main()
