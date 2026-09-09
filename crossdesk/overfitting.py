"""Deflated Sharpe ratio and probability of backtest overfitting.

`quantdesk.grade_strategy()` answers "did this hypothesis survive costs and
walk-forward?" That is the right question when there is ONE hypothesis, declared
in advance. It is the wrong question the moment a model selects among 158
features, or a sweep tries 10,000 parameter combinations, because then the
reported Sharpe is a MAXIMUM over many trials and the maximum of many noise
draws is reliably positive.

Two instruments, answering two different questions:

**Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014). Given the number of
trials that produced this winner, how likely is its Sharpe to reflect real skill
rather than the luck of the best draw? It deflates the significance threshold by
the expected maximum Sharpe under a null of zero skill, and corrects for the
non-normality that makes the ordinary Sharpe t-test optimistic on the exact
return distributions strategies tend to have -- negative skew and fat tails.

**Probability of Backtest Overfitting** (Bailey, Borwein, Lopez de Prado & Zhu
2017). Given a whole family of trials, how often does the in-sample winner
underperform the median out-of-sample? Answered by combinatorially symmetric
cross-validation, which re-splits the data many ways rather than trusting one
train/test cut. PBO near 0.5 means the selection procedure has no skill at all:
picking the in-sample best is a coin flip out of sample.

They are complements. DSR judges one reported number given a trial count; PBO
judges the SELECTION PROCEDURE that produced it. A strategy can pass DSR and
still come from a procedure with PBO of 0.6, which means the next strategy that
procedure selects should not be believed.

Standard library only. `statistics.NormalDist` supplies the normal CDF and its
inverse; the rest is arithmetic.

**Units.** Every Sharpe here is PER OBSERVATION, not annualised, and `n_obs` is
the number of observations behind it. Feeding an annualised Sharpe with a daily
`n_obs` overstates significance by roughly sqrt(252). `annual_to_per_period`
exists so the conversion is explicit at the call site.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import NormalDist, fmean

_N = NormalDist()

#: Euler-Mascheroni constant, from the Gumbel limit of the maximum of N normals.
EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Moments:
    """The four moments a deflated Sharpe needs.

    `kurtosis` is NON-EXCESS: a normal distribution has 3.0, not 0.0. This is the
    convention in the DSR literature and the opposite of `scipy.stats.kurtosis`'s
    default, which is exactly the kind of mismatch that silently changes an
    answer, so it is named and asserted rather than assumed.
    """

    n: int
    mean: float
    stdev: float
    skew: float
    kurtosis: float

    @property
    def sharpe(self) -> float:
        """Per-observation Sharpe. Not annualised."""
        if self.stdev == 0:
            raise ZeroDivisionError("zero volatility: Sharpe is undefined")
        return self.mean / self.stdev


def moments(returns: Sequence[float]) -> Moments:
    """Sample moments, with the population (biased) skew and kurtosis.

    The biased estimators are what the DSR papers use. For n in the hundreds the
    difference from the bias-corrected versions is immaterial; for n below ~30
    nothing in this module should be trusted anyway.
    """
    n = len(returns)
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}")
    mu = fmean(returns)
    devs = [r - mu for r in returns]
    m2 = sum(d * d for d in devs) / n
    if m2 <= 0:
        raise ValueError("zero variance: every return is identical")
    m3 = sum(d ** 3 for d in devs) / n
    m4 = sum(d ** 4 for d in devs) / n
    sd = math.sqrt(m2)
    return Moments(n=n, mean=mu, stdev=sd,
                   skew=m3 / m2 ** 1.5, kurtosis=m4 / m2 ** 2)


def annual_to_per_period(annual_sharpe: float, periods_per_year: float) -> float:
    """Convert an annualised Sharpe to the per-observation units used here."""
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    return annual_sharpe / math.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Probabilistic and deflated Sharpe
# ---------------------------------------------------------------------------

def sharpe_standard_error(sharpe: float, n_obs: int, skew: float = 0.0,
                          kurtosis: float = 3.0) -> float:
    """Standard error of the Sharpe estimator under non-normal returns.

        SE = sqrt( (1 - g3*SR + (g4-1)/4 * SR^2) / (n-1) )

    Negative skew and fat tails INFLATE this, which is why strategies with those
    features -- short volatility, carry, anything selling insurance -- need a
    higher observed Sharpe to clear the same bar. Under normality it reduces to
    sqrt((1 + SR^2/2)/(n-1)).
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be at least 2, got {n_obs}")
    if kurtosis < 1:
        raise ValueError(
            f"kurtosis is NON-excess and cannot be below 1, got {kurtosis}. "
            "If this came from scipy.stats.kurtosis, add 3."
        )
    variance = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2
    if variance <= 0:
        raise ValueError(
            f"non-positive Sharpe variance ({variance:.4g}) for SR={sharpe:.4g}, "
            f"skew={skew:.4g}, kurtosis={kurtosis:.4g}. The moment estimates are "
            "mutually inconsistent; check the return series."
        )
    return math.sqrt(variance / (n_obs - 1))


def probabilistic_sharpe_ratio(sharpe: float, n_obs: int, skew: float = 0.0,
                               kurtosis: float = 3.0,
                               benchmark: float = 0.0) -> float:
    """P(true Sharpe > `benchmark`) given the observed Sharpe and its moments.

    >>> round(probabilistic_sharpe_ratio(0.1, 1000), 4)
    0.9993
    """
    se = sharpe_standard_error(sharpe, n_obs, skew, kurtosis)
    return _N.cdf((sharpe - benchmark) / se)


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe across `n_trials` trials of ZERO true skill.

    The Gumbel approximation to the expected maximum of N iid normals:

        E[max] = sqrt(V) * [ (1-g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N*e)) ]

    This is the number that makes DSR bite. Run 1,000 parameter combinations on
    pure noise and the best of them will show a respectably positive Sharpe --
    not because anything works, but because you took a maximum. That expected
    maximum, not zero, is the honest benchmark for the winner.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if sharpe_variance < 0:
        raise ValueError(f"sharpe_variance must be non-negative, got {sharpe_variance}")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * ((1.0 - g) * a + g * b)


def deflated_sharpe_ratio(sharpe: float, n_obs: int, n_trials: int,
                          sharpe_variance: float, skew: float = 0.0,
                          kurtosis: float = 3.0) -> float:
    """P(true Sharpe > 0) after deflating for `n_trials` and non-normality.

    `sharpe_variance` is the variance of the Sharpe ratios ACROSS the trials --
    how much the trials disagreed. It is not the variance of returns and it is
    not the squared standard error. When the trials are unavailable, estimating
    it is the weakest step in the whole procedure; say so in the report rather
    than quietly picking a number.

    A DSR below 0.95 on a strategy selected from many trials is the expected
    result, not a surprise. Treat it as the default outcome to be argued against.
    """
    benchmark = expected_max_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurtosis, benchmark)


def min_track_record_length(sharpe: float, skew: float = 0.0,
                            kurtosis: float = 3.0, benchmark: float = 0.0,
                            confidence: float = 0.95) -> float:
    """Observations needed before this Sharpe would be significant.

    Answers "how long must the track record be?" rather than "is it significant
    yet?" -- often the more useful question, because the answer is frequently
    "longer than the strategy will exist".
    """
    if sharpe <= benchmark:
        raise ValueError(
            f"sharpe {sharpe} does not exceed benchmark {benchmark}; no track "
            "record length can make it significant"
        )
    z = _N.inv_cdf(confidence)
    variance = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe ** 2
    return 1.0 + variance * (z / (sharpe - benchmark)) ** 2


# ---------------------------------------------------------------------------
# Probability of backtest overfitting (CSCV)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PBOResult:
    """The output of combinatorially symmetric cross-validation."""

    pbo: float
    n_splits: int
    n_combinations: int
    n_trials: int
    logits: tuple[float, ...]
    oos_ranks: tuple[float, ...]
    is_sharpes: tuple[float, ...]
    oos_sharpes: tuple[float, ...]

    @property
    def median_oos_rank(self) -> float:
        return sorted(self.oos_ranks)[len(self.oos_ranks) // 2]

    @property
    def performance_degradation(self) -> tuple[float, float]:
        """OLS slope and intercept of OOS Sharpe on IS Sharpe for the winners.

        **Read this carefully: the slope is negatively biased by construction.**
        In-sample and out-of-sample partition one fixed sample, so a partition
        that hands a trial an unusually good in-sample half has necessarily left
        it a worse out-of-sample half. That mechanism produces a negative slope
        even for a strategy with a large, entirely real edge -- measured at about
        -0.75 on a trial family containing four genuine signals.

        So a negative slope is NOT by itself evidence of overfitting, and this
        property must not be used as a second opinion on `pbo`. It is descriptive
        only. `pbo` is unaffected by the artifact, because it compares the winner
        against the OTHER trials on the same out-of-sample blocks rather than
        against its own in-sample half.
        """
        xs, ys = self.is_sharpes, self.oos_sharpes
        n = len(xs)
        if n < 2:
            return 0.0, 0.0
        mx, my = fmean(xs), fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx == 0:
            return 0.0, my
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx
        return slope, my - slope * mx

    @property
    def prob_oos_loss(self) -> float:
        """Fraction of splits where the in-sample winner lost money out of sample."""
        return sum(s < 0 for s in self.oos_sharpes) / len(self.oos_sharpes)

    def verdict(self) -> str:
        if self.pbo <= 0.10:
            return "ROBUST"
        if self.pbo <= 0.25:
            return "ACCEPTABLE"
        if self.pbo <= 0.50:
            return "OVERFIT"
        return "SELECTION IS NOISE"


def _block_stats(column: Sequence[float], blocks: Sequence[slice]
                 ) -> list[tuple[int, float, float]]:
    """(count, sum, sum of squares) per block, so any union is an O(S) merge.

    Without this the CSCV loop recomputes every Sharpe from scratch for each of
    C(S, S/2) combinations, which is the difference between seconds and hours.
    """
    out = []
    for blk in blocks:
        chunk = column[blk]
        out.append((len(chunk), math.fsum(chunk), math.fsum(x * x for x in chunk)))
    return out


def _sharpe_from_stats(parts: list[tuple[int, float, float]]) -> float:
    n = sum(p[0] for p in parts)
    if n < 2:
        return 0.0
    total = math.fsum(p[1] for p in parts)
    total_sq = math.fsum(p[2] for p in parts)
    mean = total / n
    var = total_sq / n - mean * mean
    if var <= 0:
        return 0.0
    return mean / math.sqrt(var)


def probability_of_backtest_overfitting(
    trials: Sequence[Sequence[float]], n_splits: int = 16
) -> PBOResult:
    """CSCV probability of backtest overfitting over a family of trials.

    `trials` is one return series per configuration -- every parameter set the
    sweep tried, not just the winner. All must be the same length. Handing this
    only the survivors defeats the purpose: PBO measures the SELECTION, so it
    needs the things that were rejected.

    The data is cut into `n_splits` contiguous blocks. For each of the
    C(S, S/2) balanced train/test partitions, the in-sample best configuration is
    found and its out-of-sample rank recorded. PBO is the fraction of partitions
    where that winner landed below the out-of-sample median.

    **Calibration: the pure-noise baseline is about 0.60, not 0.50.** Measured on
    families of identical noise (24 trials, 1,200 observations, S=10) the winner's
    mean out-of-sample rank is 0.43 and PBO averages 0.60. Selecting the best
    in-sample Sharpe is therefore WORSE than picking at random, because the
    criterion rewards low in-sample volatility as well as high in-sample mean, and
    the volatility half of that does not persist -- it is a property of the split,
    not of the strategy.

    So do not read 0.55 as "slightly overfit". Read it as indistinguishable from
    a family with no skill whatsoever. The bands in `verdict()` are set against
    this baseline rather than against 0.5.

    Blocks are CONTIGUOUS, not shuffled, so serial correlation stays inside a
    block rather than being spread across the train/test boundary. Shuffling here
    would leak, and would make PBO look better than it is.
    """
    n_trials = len(trials)
    if n_trials < 2:
        raise ValueError(f"need at least 2 trials to measure selection, got {n_trials}")
    lengths = {len(t) for t in trials}
    if len(lengths) != 1:
        raise ValueError(f"all trials must have equal length, got lengths {sorted(lengths)}")
    n_obs = lengths.pop()
    if n_splits < 4 or n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even and at least 4, got {n_splits}")
    if n_obs < n_splits * 2:
        raise ValueError(
            f"need at least {n_splits * 2} observations for {n_splits} splits, "
            f"got {n_obs}"
        )

    edges = [round(i * n_obs / n_splits) for i in range(n_splits + 1)]
    blocks = [slice(edges[i], edges[i + 1]) for i in range(n_splits)]
    stats = [_block_stats(t, blocks) for t in trials]

    half = n_splits // 2
    all_blocks = set(range(n_splits))
    logits: list[float] = []
    ranks: list[float] = []
    is_best: list[float] = []
    oos_best: list[float] = []

    for train in combinations(range(n_splits), half):
        test = sorted(all_blocks - set(train))
        is_sr = [_sharpe_from_stats([s[i] for i in train]) for s in stats]
        oos_sr = [_sharpe_from_stats([s[i] for i in test]) for s in stats]

        winner = max(range(n_trials), key=lambda k: is_sr[k])
        # Relative rank of the winner's OOS performance among all trials, in
        # (0,1). Ties count as half, so a family of identical trials lands at
        # 0.5 rather than at an arbitrary extreme.
        below = sum(oos_sr[k] < oos_sr[winner] for k in range(n_trials))
        equal = sum(oos_sr[k] == oos_sr[winner] for k in range(n_trials))
        omega = (below + 0.5 * equal) / n_trials
        omega = min(max(omega, 1.0 / (2 * n_trials)), 1.0 - 1.0 / (2 * n_trials))

        logits.append(math.log(omega / (1.0 - omega)))
        ranks.append(omega)
        is_best.append(is_sr[winner])
        oos_best.append(oos_sr[winner])

    pbo = sum(x < 0 for x in logits) / len(logits)
    return PBOResult(
        pbo=pbo,
        n_splits=n_splits,
        n_combinations=len(logits),
        n_trials=n_trials,
        logits=tuple(logits),
        oos_ranks=tuple(ranks),
        is_sharpes=tuple(is_best),
        oos_sharpes=tuple(oos_best),
    )
