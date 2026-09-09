# CLAUDE.md — crossdesk

Cross-sectional alpha on US equities and ETFs. Four tools, one referee, and a
data layer that makes the cross-sectional leaks impossible to introduce quietly.

Inherits `Downloads/CLAUDE.md` (verify numerically, do not assert) and
`quantdesk/CLAUDE.md` (the causality rule, UTC, gross beside net). This file
adds what is specific to ranking many names at once, and to running three
third-party engines against one another.

## Why four tools and not one

| Tool | Job | Explicitly NOT its job |
|---|---|---|
| **Qlib** (3.12 venv) | cross-sectional alpha: rank N names, rolling retrain | costs, execution, the verdict |
| **VectorBT** | parameter screening, in-sample only | producing any number that gets reported |
| **quantdesk** | the referee: pre-registration, cost sensitivity, walk-forward, `grade_strategy()` | speed |
| **NautilusTrader** | event-driven revalidation, paper, live | alpha discovery |

**quantdesk is the only thing allowed to issue a verdict.** VectorBT's
`total_return` and Qlib's IC are diagnostics for narrowing a hypothesis. They
are not results. A number leaves this project through `grade_strategy()` or it
does not leave.

## Power before hypotheses

`crossdesk/power.py` exists because the six-family library screen in quantdesk
returned six nulls at **12.7% power** against a literature-strength effect. A
test that weak cannot distinguish "does not work" from "cannot tell", and only
one of those should end a research programme.

**Every screen states its power before it runs.** `Design.verdict(literature_sharpe)`
returns ADEQUATE / WEAK / UNDERPOWERED, and an UNDERPOWERED null is reported as
UNDERPOWERED — never as evidence of absence. The arithmetic:

    minimum detectable Sharpe = t_threshold / sqrt(years)
    IR = IC * sqrt(breadth),  breadth = names * rebalances * haircut

At 15 names over 14 years the design demands a per-bet IC of 0.105, higher than
any published cross-sectional signal. **The fix is breadth, not years** — more
years do not exist. Roughly 300 names is where a realistic IC of 0.03 becomes
detectable. That number is soft (the Fundamental Law assumes independent bets,
which correlated equities are not); the order of magnitude is not.

## The two survivorship leaks

quantdesk's `assert_causal` guards a feature against reading the future. It
cannot guard the **universe**, which is chosen before any feature exists — and
choosing it wrong biases everything downstream in the same optimistic direction.
The two leaks are separated in `data/universe.py` because they have different
causes and different fixes, and calling both "survivorship bias" is how one of
them goes unfixed.

**Selection leak** — using today's membership for a past date. Guarded by
`assert_membership_is_causal(build, listings, dates)`, which takes the universe
*builder*, not a finished universe. This matters: an earlier version took a
`PointInTimeUniverse` and could never fail, because `members_asof` consults only
start and end against the date it was handed. It was a tautology in the costume
of a test. The real bug is upstream, in filters like "names with enough history"
or "tickers the API returns today", so the guard runs the builder twice — once
with the full record, once with only what was knowable — and demands the same
answer.

**Delisting leak** — a name that stops trading vanishes from the panel, so a
bankruptcy becomes an *absence* rather than a -100% return and the cross-section
silently loses its worst outcomes. `Listing` refuses to construct an ended
listing without a terminal return, and `PricePanel.period_returns` raises rather
than skipping a member it cannot price. **Never soften that to a skip.** The
measured cost, on a 10-name panel with survivors at +10%: **0.11 of return per
deleted name**, so one unpriced delisting in ten moves the mean by 11 points.
`tests/test_panel.py` pins that number.

Report `PricePanel.provenance` beside any cross-sectional result. A period where
5% of the cross-section came from `delisted` is a period whose ranking rests on
an assumption rather than on prices.

## CRSP — the universe that replaced the fifteen ETFs (2026-09-08)

`data/crsp_monthly.csv`: CRSP monthly, 1990-01 .. 2024-12 (CRSP's annual
release ends there), ordinary common shares on NYSE/AMEX/NASDAQ, 2.1M rows,
18.7k PERMNOs, delisting returns included. Pulled by
`dev/cisd-bot/research/wrds_pull.py`, converted by `scripts/crsp_import.py`,
loaded by `crossdesk/data/crsp.py`, audited by `scripts/crsp_audit.py`
(`results/crsp_audit.txt`). Every guard passes on it: causal top-N-by-size
membership, no unpriced dropout, 2.24% of period returns from delistings.

**Power now:** top-300 names, 24.5 in-sample years, six Bonferroni families:
**96.5%** against a Sharpe from IC 0.03; top-500 99.9%. Fifteen ETFs were 11.7%.
The breadth argument in "Power before hypotheses" is settled.

Conventions the loader enforces, each pinned in `tests/test_crsp.py`:

- **PERMNO is the key.** Tickers are labels. Two businesses that reused a
  ticker are two series.
- **A delisted window ends at the last FULL month.** The delisting-month row
  says the position ended during that month; the name is holdable through the
  previous month end, and that month's `ret_adj` (CRSP `dlret` folded in) is
  the listing's explicit `delisting_return`, applied by
  `PricePanel.period_return` on top of the previous index level. A -100%
  delisting is a -100% return, never a zero price and never a hole.
- **Shumway (1997) only where CRSP is silent:** a delisting code with no
  return gets -30% for codes 400-599 and 0% for 200-399; the count is
  reported by the audit (28 across the whole file).
- **Rows are re-dated to the common month end.** CRSP dates a halted name's
  row on its own last trade; without this a member had no value on the
  rebalance date.
- Total-return index (dividends in), not raw prices; splits are therefore
  handled, which the ETF panel section below still says they are not.

Four traps that cost the evening, so they are not re-learned:

1. **WRDS on Python 3.14 segfaults on any query result** (exit 139, no
   output) while `list_libraries` works. Pull with `.venv-qlib` (3.12).
2. **The classic msf x msenames join drops the delisting month.** msf rows
   are month-end dated; the last name record ends on the delisting day. On
   1990 that kept 49 of 515 delisting returns. `MONTHLY_SQL` extends
   `nameendt` to its month end and de-duplicates (permno, date).
3. `wrds.Connection()` without `wrds_username` uses the OS user and never
   consults the pgpass entry. See the `wrds-crsp-access` memory.
4. `PricePanel` used to scan every listing per return; on 30k listings that
   is the whole runtime. Ended windows are now indexed per symbol.

Holdout on this panel: last 30% = **2014-07-31 onward, 126 month-ends,
SEALED**. Nothing reads it yet.

## What the ETF panel can and cannot support (superseded by CRSP above)

`etf_daily.csv` is 15 tickers, 2006-2026, and every guard passes on it: 100%
`traded` provenance, no dropouts, causal membership on all 241 month-ends. It is
**correct and too narrow** — 11.7% power at a realistic IC, uncorrected.

The delisting leak is absent here by construction: none of the fifteen has ever
stopped trading. The selection leak is **not** absent — those tickers were
chosen in 2026 from funds that still exist in 2026, and over a thousand US ETFs
have closed since 2006. The bias is small for large sector SPDRs and it is
**unmeasured**, because measuring it needs prices for funds that closed, which
no free source provides.

ASSUMPTION (data availability): widening past this universe requires a
delisted-inclusive vendor — Sharadar SEP, Norgate, or CRSP. Build
`PointInTimeUniverse` from that vendor's listing table, never from whatever
tickers an API returns today.

Splits are **not** handled. Raw series across a 2-for-1 book a -50% return that
never happened. The ETF set is split-free; a wider equity universe is not.

## The agreement test — `engine/spec.py`, `engine/adapters.py`

`engine/spec.py` is the canonical reference: forty lines of arithmetic, and
every engine is measured against IT rather than against each other, so a
disagreement has an owner. Signals are precomputed once and shared — if each
engine built its own moving average, an indicator difference and a fill
difference would be indistinguishable.

Run `scripts/engine_agreement.py` after every version bump. A dependency upgrade
that silently changes a fill convention is indistinguishable from alpha.

### Result: they do not all agree, and the reasons are known

```
quantdesk   max diff vs reference   0.000000000
vectorbt    max diff vs reference   0.000000000
nautilus    max diff vs reference   0.015 (vs decision_close ref)
```

**Divergence 1 — fill convention.** Nautilus in bar-execution mode fills a market
order at the DECISION bar's own close, measured in `scripts/_nautilus_probe.py`
(3 of 3 fills) rather than assumed. quantdesk and VectorBT fill at the next
bar's open. On SPY 2006-2020 that convention alone is worth **30.5 bps**, and it
is optimistic — acting on a close at that close needs the close before it prints.

**Nautilus bar-mode numbers must never be placed beside quantdesk's.** Use
Nautilus for execution and live, or feed it quote/trade ticks so the fill is
real rather than a convention.

**Divergence 2 — the cash constraint.** VectorBT and Nautilus refuse to spend
cash they do not have and partially fill. **quantdesk does not model the
constraint at all**, so a backtest there can hold a position it could never have
funded, financed at zero cost. Sizing off the first close of SPY and letting the
price triple produces an **$11,776** gap with no fill-timing involved. Invisible
until position size approaches account size, which is exactly where a small
account operates. This is a gap in quantdesk; the correct policy is quantdesk's
to decide.

### Two traps this test caught on itself

- **Warmup alignment.** quantdesk refuses to trade for 21 bars; VectorBT has no
  such notion. On a series whose signal fires at bar 20, VectorBT takes a
  position quantdesk never took and the curves stay offset forever — which looks
  exactly like a fill bug. `mask_warmup` gives every engine the same decisions.
- **Position size hides everything.** At 1 share on $100k both divergences above
  round to nothing. Test at realistic notional or the test proves nothing.

## Multiple testing — `overfitting.py`

`grade_strategy()` is built for one pre-registered hypothesis. Alpha158 is 158
features fed to a model that selects among them, and Bonferroni does not
describe that. **Every selected result reports DSR and PBO**, not just costs and
walk-forward.

- **Deflated Sharpe** judges one number given a trial count. It deflates against
  `expected_max_sharpe`, the Sharpe the best of N noise trials would show anyway,
  and corrects for the skew and fat tails that make the ordinary Sharpe t-test
  optimistic. Report the undeflated PSR beside it — the gap is the cost of having
  searched, the same way gross sits beside net.
- **PBO** judges the SELECTION PROCEDURE. A strategy can pass DSR and still come
  from a procedure with PBO of 0.6, which means the next strategy that procedure
  selects should not be believed.

**Sharpes here are PER OBSERVATION, never annualised.** Feeding an annualised
Sharpe with a daily `n_obs` overstates significance by ~sqrt(252). Use
`annual_to_per_period`. `kurtosis` is NON-excess (normal = 3.0), the opposite of
scipy's default; values below 1 are refused.

**PBO's pure-noise baseline is ~0.60, not 0.50.** Measured over 40 seeds: 0.603,
SE 0.025. The in-sample winner's mean out-of-sample rank is 0.43, so selecting on
noise is *worse* than picking at random — the Sharpe criterion rewards low
in-sample volatility, and that half is a fact about the split, not the strategy.
**Compare a result against 0.60.** Reading 0.55 as "slightly overfit" would be
reading a family with no skill at all as promising.

`performance_degradation` is **descriptive only**. Its slope is negatively biased
by construction — in-sample and out-of-sample partition one fixed sample — and it
measures -0.79 on a family containing four genuine signals. Never use it as a
second opinion on PBO.

PBO on one family has a standard deviation of ~0.19. A single run is an
observation, not a measurement; average over seeds when the answer matters.

### What it caught on the current data

`scripts/sweep_deflation.py`, 48 momentum configurations, in-sample only:

```
PSR (one hypothesis)      95.6%     <- clears an uncorrected 95% bar
DSR (deflated, 48 trials) 82.8%     <- does not
PBO                       69.0%     SELECTION IS NOISE
```

Worse than the 0.60 noise baseline. That gap between 95.6% and 82.8% is the
number `grade_strategy()` structurally could not produce.

## The families re-run on CRSP (2026-09-09) — and what breadth actually bought

`scripts/screen_crsp_families.py`, output in `results/screen_crsp_families.txt`.
Pre-registered, in-sample only (1990-01 .. 2014-06, 294 month-ends), holdout
untouched. Top 300 by market cap, 30 names per side, 10bp round trip charged on
actual turnover, Bonferroni across the registered family of six: |t| > 2.64.

**Two of the six transfer, not six.** L1 overnight, L2 turn-of-month and L3
pre-holiday need daily or intraday bars a monthly panel does not have — and,
more importantly, none of them is breadth-limited: each is one time series whose
power comes from the number of SESSIONS. L4 is market-level timing. Only L5 is
cross-sectional; L6 is a per-name rule that breadth helps indirectly. Saying
"we ran the six families on 300 names" would have been false, so the script says
so in its own docstring.

| family | gross | net | t (net) | ann SR | verdict |
|---|---:|---:|---:|---:|---|
| L5 cross-sectional momentum | +94.1 bp/mo | +88.1 | **+1.65** | +0.34 | fails 2.64, fails 1.96 |
| L6 trend following | -26.0 bp/mo | -33.0 | **-2.12** | -0.43 | fails 2.64 |

L5 has the registered sign and a plausible size, and it is not significant.
Turnover is only 0.61/month so it survives 50bp costs; the problem is the error
bar, not the cost. It decays hard by decade (1990s +168bp t=+2.65, 2000s +46bp
t=+0.43, 2010s +22bp t=+0.37) with kurtosis 10.4 — the momentum-crash signature.

L6's negative return was pre-registered as not refuting it, and the descriptive
path numbers show why that clause mattered: max drawdown **23.8% vs 52.3%** and
monthly vol **2.58% vs 4.34%**. The rule does exactly what it is published to
do. It buys that with return, and the return cost is real before any cost is
charged.

### THE BREADTH ARGUMENT WAS TOO OPTIMISTIC, and by how much

This is the finding that outranks both families.

```
predicted annual Sharpe at IC=0.03, 300 names, haircut 0.25    0.90
measured annual Sharpe, L5 gross                               0.36
-> implied IC at the assumed 0.25 haircut                      0.0122
-> implied haircut at the assumed IC of 0.03                   0.041
```

The Fundamental Law counts INDEPENDENT bets. Thirty longs and thirty shorts
picked by one signal are one factor bet held sixty ways, not sixty bets. The
0.25 haircut in `power.py` is roughly six times too generous for a
cross-sectional factor; measured here it is nearer **0.04**.

So `results/crsp_audit.txt`'s "96.5% power at 300 names" was power against a
Sharpe of 0.90 that this universe does not produce. Power against the
*literature* effect of 0.40 is **25.5%**, and the minimum detectable annual
Sharpe at |t|>2.64 over 24.5 years is **0.53** — above most of the published
range. The audit now prints both columns and says to read the literature one.

**The breadth fix was still worth doing** — 11.7% to 25.5% is real, and the
delisting-inclusive panel removed a bias worth 0.11 of return per deleted name.
But breadth alone does not rescue a 0.40-Sharpe effect over this sample, and
"300 names is where it becomes detectable" is now known to be wrong. What would
change the answer is a HIGHER-SHARPE signal or MORE REBALANCES, not more names.

### The aggregation rule, pinned in `tests/test_families.py`

Both families produce one number per name per month. A t-test over pooled
name-months treats co-moving stocks as independent draws and inflates the
t-statistic by

    sqrt(k) * sqrt( (k*M - 1) / (k * (M - 1)) )

which the test asserts exactly rather than approximately. On this data pooling
would have reported **t = +30.4** for L5 instead of +1.65, and **-28.8** for L6
instead of -2.12. Every statistic in `screen/families.py` is therefore the
monthly portfolio series: names averaged first, one observation per month,
non-overlapping so no further correction is needed.

## Holdout

The last **30%** is SEALED, matching every screen in quantdesk. On the ETF panel
that is 2020-09-30 onward, 73 of 241 month-ends. Scripts that touch it say so in
their module docstring. Nothing in this repo reads it yet.

## Environments

Two, because Qlib cannot share the others' interpreter.

```
.venv        Python 3.14   vectorbt, nautilus_trader, quantdesk
.venv-qlib   Python 3.12   qlib (+ LightGBM), quantdesk, wrds (CRSP pulls)
```

- **plotly is pinned `<7`.** vectorbt 1.1.0 declares `plotly>=4.12.0` with no
  upper bound; plotly 7.0 removed the `scattermapbox` trace its theme registers
  at import, so an unpinned install fails at `import vectorbt` while reporting a
  successful install.
- **Qlib is installed `--no-deps`.** It requires `gym`, the archived OpenAI
  package, which is sdist-only and unresolvable. `gym` is imported only by
  `qlib.rl`, which this project does not use.
