"""What a parameter sweep on the real ETF panel looks like after deflation.

Sweeps cross-sectional momentum over lookback x skip x quantile, takes the
winner the way a VectorBT sweep would, and then asks the two questions
`grade_strategy()` cannot:

  * Deflated Sharpe -- is the winner's Sharpe credible GIVEN how many
    configurations were tried?
  * PBO -- does the selection procedure itself carry any skill, or would
    picking the in-sample best have been a coin flip?

**The holdout is SEALED.** This reads only the first 70% of month-ends, matching
every screen in the quantdesk repo. Nothing here is a verdict on momentum; it is
a demonstration that the referee now catches what it previously could not.

Run: python scripts/sweep_deflation.py
"""

import math
from statistics import fmean, pstdev

from crossdesk.data.etf_panel import load_panel, month_ends
from crossdesk.data.panel import equal_weight_long_short, momentum_scores
from crossdesk.overfitting import (
    deflated_sharpe_ratio,
    moments,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)

RULE = "=" * 74
HOLDOUT_FRACTION = 0.30

LOOKBACKS = (63, 126, 189, 252)
SKIPS = (0, 5, 10, 21)
QUANTILES = (0.2, 0.3, 0.4)


def run_config(panel, rebalances, lookback, skip, quantile):
    """Monthly long-short returns for one configuration."""
    out = []
    for start, end in zip(rebalances, rebalances[1:]):
        scores = momentum_scores(panel, start, lookback, skip)
        if len(scores) < 4:
            out.append(0.0)
            continue
        forward, _ = panel.period_returns(start, end)
        out.append(equal_weight_long_short(scores, forward, quantile))
    return out


def main() -> None:
    panel = load_panel()
    rebalances = month_ends(panel)
    cut = int(len(rebalances) * (1 - HOLDOUT_FRACTION))
    in_sample = rebalances[:cut]

    print(RULE)
    print("SWEEP DEFLATION -- cross-sectional momentum, 15 ETFs")
    print(RULE)
    print(f"in-sample  {len(in_sample)} month-ends "
          f"({in_sample[0]} .. {in_sample[-1]})")
    print(f"holdout    {len(rebalances) - cut} SEALED, not read by this script")

    configs, series = [], []
    for lb in LOOKBACKS:
        for sk in SKIPS:
            for q in QUANTILES:
                returns = run_config(panel, in_sample, lb, sk, q)
                if pstdev(returns) > 0:
                    configs.append((lb, sk, q))
                    series.append(returns)
    n_trials = len(configs)
    print(f"configurations tried  {n_trials}")
    print()

    sharpes = [fmean(s) / pstdev(s) for s in series]
    best = max(range(n_trials), key=lambda i: sharpes[i])
    lb, sk, q = configs[best]
    winner = series[best]
    m = moments(winner)
    annual = m.sharpe * math.sqrt(12)

    print(f"{'WINNER':<22} lookback={lb}d skip={sk}d quantile={q}")
    print(f"{'monthly Sharpe':<22} {m.sharpe:>8.4f}")
    print(f"{'annualised Sharpe':<22} {annual:>8.2f}")
    print(f"{'skew':<22} {m.skew:>8.2f}")
    print(f"{'kurtosis (non-excess)':<22} {m.kurtosis:>8.2f}")
    print(f"{'observations':<22} {m.n:>8}")
    print()

    naive = probabilistic_sharpe_ratio(m.sharpe, m.n, m.skew, m.kurtosis)
    sharpe_var = pstdev(sharpes) ** 2
    dsr = deflated_sharpe_ratio(m.sharpe, m.n, n_trials, sharpe_var,
                                m.skew, m.kurtosis)
    print(f"{'PSR (one hypothesis)':<34} {naive:>7.1%}")
    print(f"{'spread of trial Sharpes (var)':<34} {sharpe_var:>7.5f}")
    print(f"{'DSR (deflated for ' + str(n_trials) + ' trials)':<34} {dsr:>7.1%}")
    print()

    result = probability_of_backtest_overfitting(series, n_splits=10)
    slope, _ = result.performance_degradation
    print(f"{'PBO':<34} {result.pbo:>7.1%}   {result.verdict()}")
    print(f"{'  partitions':<34} {result.n_combinations:>7,}")
    print(f"{'  winner mean OOS rank':<34} {fmean(result.oos_ranks):>7.3f}")
    print(f"{'  P(winner loses OOS)':<34} {result.prob_oos_loss:>7.1%}")
    print(f"{'  IS->OOS slope (biased, see doc)':<34} {slope:>7.3f}")
    print()

    print(RULE)
    print("READING THIS")
    print(RULE)
    print("PBO's pure-noise baseline is ~0.60, not 0.50 -- selecting the best")
    print("in-sample Sharpe is worse than random because the criterion rewards")
    print("low in-sample volatility, which does not persist. Compare against")
    print("0.60, not against 0.50.")
    print()
    print("The gap between PSR and DSR is the cost of having searched. It is")
    print("the number `grade_strategy()` could not produce, because it assumes")
    print("one pre-registered hypothesis and this was", n_trials, "of them.")


if __name__ == "__main__":
    main()
