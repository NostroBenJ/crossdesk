# crossdesk

Cross-sectional equity research on survivorship-free CRSP data, built instruments first: the checks for
statistical power, survivorship and overfitting exist before any strategy gets judged.

## Why it exists

My first stock-momentum backtest returned **+26.75% a year at a Sharpe of 1.11**, on 100 large caps that
all still trade today. Every one of them survived, by construction. This repo exists to find out how much
of that number was survivorship. On CRSP, with dead companies and their delisting returns included, the
same idea has a **Sharpe of 0.34**.

## The instruments

| module | what it answers |
|---|---|
| `power.py` | Before running a test: could it detect a literature-sized effect at all? An earlier 15-ETF screen had ~13% power, so its six nulls meant "can't tell", not "doesn't work". |
| `data/universe.py`, `data/panel.py` | Point-in-time membership. A delisted name can't leave the panel without a terminal return; missing ones cost 0.11 of return each on a test panel. |
| `data/crsp.py` | CRSP loader: PERMNO-keyed total returns, delisting returns folded into the final month, and a guard for the standard name-history join, which silently drops delisting months. |
| `overfitting.py` | Deflated Sharpe ratio and probability of backtest overfitting. PBO's pure-noise baseline measures ~0.60, not the textbook 0.50. |
| `engine/spec.py`, `engine/adapters.py` | Three backtest engines checked against a 40-line reference, so a disagreement has an owner. VectorBT matches the reference exactly; NautilusTrader's bar mode is 30.5bp optimistic; the in-house engine has no cash constraint. |
| `screen/families.py`, `screen/variants.py` | The pre-registered screens. |

## Results

Top 300 US stocks by market cap, 30 names per side, in-sample 1990-01 to 2014-06 (294 month-ends),
costs on actual turnover. The holdout is sealed.

| strategy | net bp/month | t | annual Sharpe |
|---|---:|---:|---:|
| cross-sectional momentum | +88 | +1.65 | 0.34 |
| time-series trend | −33 | −2.12 | — (max drawdown 23.8% vs 52.3%) |
| momentum, common window | +81.82 | +1.42 | 0.31 |
| residual momentum | +62.19 | +1.60 | 0.35 |
| volatility-scaled momentum | +45.64 | +1.73 | 0.38 (max drawdown 70.8% → 27.7%) |
| inverse-vol weighted momentum | +57.74 | +1.01 | 0.22 (**significantly worse**, paired t = −3.83) |

- **Nothing clears the multiple-testing bar** (|t| > 2.77 across nine hypotheses). Momentum decays by
  decade, from t = +2.65 in the 1990s to +0.37 in the 2010s.
- **The breadth argument was ~6x too optimistic.** The Fundamental Law promised a Sharpe of 0.90 from
  300 names; the spread delivered 0.36 gross. Sixty names chosen by one signal are one bet, not sixty,
  so effective breadth tracks the number of months. Honest power is 21–26%, and the smallest detectable
  annual Sharpe is 0.53, above most published effects. **No further in-sample test on this panel can
  settle the question.**
- Pooling name-months as if they were independent would have printed t = +30.4 instead of +1.65. Every
  statistic here comes from the monthly portfolio series.
- The holdout (2014-07-31 onward, 126 month-ends) stays sealed, with three candidates registered.

The full reasoning behind each choice is in [`CLAUDE.md`](CLAUDE.md). Output of every run is in `results/`.

## Running it

Python 3.12+.

```
pip install -e ".[dev]"          # add [screen] for VectorBT, [engine] for NautilusTrader
pytest
python scripts/screen_crsp_families.py
python scripts/screen_crsp_variants.py
```

Two things this repo can't include:
- **CRSP data** is licensed through WRDS and never committed. `scripts/crsp_import.py` builds the panel
  from your own WRDS extract; see [`data/README.md`](data/README.md).
- **`quantdesk`**, the backtesting package it depends on, is a separate private project, so the engine
  comparison and the crypto conventions tests won't import without it.

Research code for learning, not investment advice.
