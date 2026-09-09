"""Deflated Sharpe and PBO, checked against simulation and against two controls.

The file is built around the same discipline as quantdesk's coin-flip pipeline
test: an instrument is trusted only after it has been shown to give the right
answer in two cases whose answers are known in advance.

  * `test_pbo_on_pure_noise_is_at_least_a_coin_flip` -- selection among identical
    noise has no skill, so the in-sample winner must land below the out-of-sample
    median at least half the time. The measured baseline is ~0.60 rather than the
    naive 0.50, for a reason `test_selection_on_noise_is_worse_than_random`
    isolates.
  * `test_pbo_is_near_zero_with_one_genuinely_better_strategy` -- when one trial
    really is better, the procedure must find it and keep it. An instrument that
    reports "overfit" for everything is as useless as one that never does.

Several tests here average over twelve seeds. PBO on a single family has a
standard deviation of about 0.19, so a one-seed assertion tests the seed rather
than the code -- a lesson learned by writing exactly that test first and watching
it fail on a value that turned out to be perfectly healthy.
"""

import math
import random

import pytest

from crossdesk.overfitting import (
    EULER_MASCHERONI,
    Moments,
    annual_to_per_period,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_standard_error,
)

from statistics import NormalDist, fmean

_N = NormalDist()


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def test_moments_of_a_symmetric_sample():
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    m = moments(xs)
    assert m.n == 5
    assert m.mean == pytest.approx(0.0)
    assert m.skew == pytest.approx(0.0, abs=1e-12)
    assert m.stdev == pytest.approx(math.sqrt(2.0))


def test_kurtosis_is_non_excess():
    """A normal sample must land near 3.0, not near 0.0."""
    rng = random.Random(11)
    xs = [rng.gauss(0, 1) for _ in range(200_000)]
    assert moments(xs).kurtosis == pytest.approx(3.0, abs=0.05)


def test_moments_reject_degenerate_input():
    with pytest.raises(ValueError, match="at least 3"):
        moments([1.0, 2.0])
    with pytest.raises(ValueError, match="zero variance"):
        moments([1.0, 1.0, 1.0])


def test_sharpe_is_per_observation():
    m = Moments(n=100, mean=0.001, stdev=0.01, skew=0.0, kurtosis=3.0)
    assert m.sharpe == pytest.approx(0.1)


def test_annual_to_per_period_round_trip():
    per = annual_to_per_period(1.5, 252)
    assert per * math.sqrt(252) == pytest.approx(1.5)
    assert per == pytest.approx(1.5 / math.sqrt(252))


# ---------------------------------------------------------------------------
# Standard error and PSR
# ---------------------------------------------------------------------------

def test_standard_error_reduces_to_the_normal_case():
    """Under normality SE = sqrt((1 + SR^2/2)/(n-1))."""
    for sr in (0.0, 0.05, 0.2):
        for n in (100, 1000):
            expected = math.sqrt((1 + sr ** 2 / 2) / (n - 1))
            assert sharpe_standard_error(sr, n) == pytest.approx(expected, rel=1e-12)


def test_negative_skew_and_fat_tails_widen_the_error():
    """The correction that matters for anything selling insurance."""
    base = sharpe_standard_error(0.15, 1000, skew=0.0, kurtosis=3.0)
    skewed = sharpe_standard_error(0.15, 1000, skew=-1.5, kurtosis=3.0)
    fat = sharpe_standard_error(0.15, 1000, skew=0.0, kurtosis=9.0)
    assert skewed > base
    assert fat > base


def test_excess_kurtosis_passed_by_mistake_is_refused():
    """scipy returns EXCESS kurtosis; 0.0 here would silently flatter the result."""
    with pytest.raises(ValueError, match="NON-excess"):
        sharpe_standard_error(0.1, 1000, kurtosis=0.0)


def test_psr_matches_the_normal_cdf_of_the_t_statistic():
    for sr in (0.02, 0.1):
        for n in (250, 2000):
            se = math.sqrt((1 + sr ** 2 / 2) / (n - 1))
            assert probabilistic_sharpe_ratio(sr, n) == pytest.approx(
                _N.cdf(sr / se), rel=1e-12
            )


def test_psr_is_one_half_at_the_benchmark():
    assert probabilistic_sharpe_ratio(0.1, 500, benchmark=0.1) == pytest.approx(0.5)


def test_psr_is_monotone_in_sharpe_and_in_sample_size():
    assert (probabilistic_sharpe_ratio(0.05, 500)
            < probabilistic_sharpe_ratio(0.10, 500))
    assert (probabilistic_sharpe_ratio(0.05, 500)
            < probabilistic_sharpe_ratio(0.05, 5000))


def test_psr_is_calibrated_under_the_null():
    """With no true edge, PSR must be uniform -- so its mean is 0.5.

    This is the calibration check. A PSR that averaged 0.7 on strategies with no
    skill would certify noise 70% of the time.
    """
    rng = random.Random(4242)
    values = []
    for _ in range(3000):
        xs = [rng.gauss(0.0, 1.0) for _ in range(500)]
        m = moments(xs)
        values.append(probabilistic_sharpe_ratio(m.sharpe, m.n, m.skew, m.kurtosis))
    assert sum(values) / len(values) == pytest.approx(0.5, abs=0.02)
    # Uniform, not merely centred: about a fifth should sit below 0.2.
    assert sum(v < 0.2 for v in values) / len(values) == pytest.approx(0.2, abs=0.03)


# ---------------------------------------------------------------------------
# Expected maximum Sharpe
# ---------------------------------------------------------------------------

def test_expected_max_sharpe_is_zero_for_a_single_trial():
    """One trial is not a selection, so nothing is deflated."""
    assert expected_max_sharpe(1, 1.0) == 0.0


@pytest.mark.parametrize("n_trials", [10, 100, 1000])
def test_expected_max_sharpe_matches_monte_carlo(n_trials):
    """The Gumbel approximation, checked against the actual maximum of N draws."""
    rng = random.Random(909 + n_trials)
    variance = 0.04
    sd = math.sqrt(variance)
    maxima = [max(rng.gauss(0.0, sd) for _ in range(n_trials))
              for _ in range(4000)]
    simulated = sum(maxima) / len(maxima)
    analytic = expected_max_sharpe(n_trials, variance)
    assert analytic == pytest.approx(simulated, abs=0.01)


def test_expected_max_sharpe_grows_with_trial_count():
    vals = [expected_max_sharpe(n, 0.04) for n in (2, 10, 100, 1000, 10000)]
    assert vals == sorted(vals)


def test_expected_max_sharpe_scales_with_the_spread_of_trials():
    """Doubling the standard deviation of trial Sharpes doubles the benchmark."""
    a = expected_max_sharpe(500, 0.01)
    b = expected_max_sharpe(500, 0.04)
    assert b == pytest.approx(2 * a, rel=1e-12)


def test_expected_max_sharpe_rejects_bad_input():
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe(0, 0.04)
    with pytest.raises(ValueError, match="non-negative"):
        expected_max_sharpe(10, -0.01)


def test_euler_constant_is_right():
    assert EULER_MASCHERONI == pytest.approx(0.577215664901, abs=1e-11)


# ---------------------------------------------------------------------------
# Deflated Sharpe
# ---------------------------------------------------------------------------

def test_dsr_equals_psr_when_there_was_only_one_trial():
    assert (deflated_sharpe_ratio(0.1, 1000, n_trials=1, sharpe_variance=0.04)
            == pytest.approx(probabilistic_sharpe_ratio(0.1, 1000)))


def test_dsr_falls_as_the_trial_count_rises():
    """The whole point: the same Sharpe means less after more searching."""
    values = [deflated_sharpe_ratio(0.12, 1000, n_trials=n, sharpe_variance=0.02)
              for n in (1, 10, 100, 1000, 10000)]
    assert values == sorted(values, reverse=True)
    assert values[0] > 0.99
    assert values[-1] < values[0]


def test_a_sweep_winner_that_would_pass_undeflated_fails_deflated():
    """The concrete failure DSR exists to catch.

    A per-period Sharpe of 0.09 over 1,000 observations looks decisive on its
    own -- until you say it was the best of 5,000 parameter combinations.
    """
    sharpe, n_obs = 0.09, 1000
    assert probabilistic_sharpe_ratio(sharpe, n_obs) > 0.99
    deflated = deflated_sharpe_ratio(sharpe, n_obs, n_trials=5000,
                                     sharpe_variance=0.0025)
    assert deflated < 0.95


def test_dsr_penalises_negative_skew_when_the_edge_clears_the_benchmark():
    """Fat left tails cost you -- but only when you are ABOVE the benchmark."""
    plain = deflated_sharpe_ratio(0.1, 1000, 10, 0.0002, skew=0.0, kurtosis=3.0)
    ugly = deflated_sharpe_ratio(0.1, 1000, 10, 0.0002, skew=-2.0, kurtosis=12.0)
    assert plain > 0.9                      # comfortably above its benchmark
    assert ugly < plain


def test_below_the_benchmark_extra_uncertainty_helps_and_that_is_correct():
    """The direction reverses below the benchmark, and it is not a bug.

    Negative skew and fat tails widen the standard error. When the observed
    Sharpe EXCEEDS the benchmark, a wider error makes the excess less convincing.
    When it FALLS SHORT, the same widening makes it more plausible that the true
    Sharpe is above after all. The statistic is symmetric; only the intuition
    that "ugly moments are always penalised" is not.

    Worth pinning because it looks like a sign error the first time it is seen,
    and both DSRs here are near zero anyway -- neither strategy is credible.
    """
    below_plain = deflated_sharpe_ratio(0.1, 1000, 100, 0.01,
                                        skew=0.0, kurtosis=3.0)
    below_ugly = deflated_sharpe_ratio(0.1, 1000, 100, 0.01,
                                       skew=-2.0, kurtosis=12.0)
    assert below_ugly > below_plain
    assert below_plain < 0.01 and below_ugly < 0.01


# ---------------------------------------------------------------------------
# Minimum track record length
# ---------------------------------------------------------------------------

def test_min_track_record_length_agrees_with_psr():
    """At exactly the minimum length, PSR must equal the confidence level."""
    sr = 0.08
    n = min_track_record_length(sr, confidence=0.95)
    assert probabilistic_sharpe_ratio(sr, math.ceil(n) + 1) >= 0.95
    assert probabilistic_sharpe_ratio(sr, int(n)) == pytest.approx(0.95, abs=0.02)


def test_min_track_record_grows_as_the_edge_shrinks():
    assert min_track_record_length(0.02) > min_track_record_length(0.10)


def test_min_track_record_refuses_an_edge_below_the_benchmark():
    with pytest.raises(ValueError, match="does not exceed"):
        min_track_record_length(0.05, benchmark=0.05)


# ---------------------------------------------------------------------------
# PBO -- the two controls
# ---------------------------------------------------------------------------

def _noise_trials(n_trials, n_obs, rng, drift=0.0):
    return [[rng.gauss(drift, 1.0) for _ in range(n_obs)] for _ in range(n_trials)]


SEEDS = range(12)


def _mean_pbo(build, n_splits=10):
    """Average PBO over several seeds.

    PBO on a single family is a NOISY estimate: measured across seeds on pure
    noise it has a standard deviation of roughly 0.19, so any single run can land
    anywhere from 0.39 to 0.83 with nothing wrong. Testing a stochastic estimator
    against one seed tests the seed. Averaging twelve cuts the standard error to
    about 0.06, which is tight enough to catch a real defect.
    """
    values = [probability_of_backtest_overfitting(build(random.Random(s)),
                                                  n_splits).pbo
              for s in SEEDS]
    return fmean(values)


def test_pbo_on_pure_noise_is_at_least_a_coin_flip():
    """NULL CONTROL. Identical noise trials: selection carries no information.

    The naive expectation is 0.5. The measured baseline is about **0.60**, and
    the gap is real rather than sampling error -- over 40 seeds the mean is 0.603
    with a standard error of 0.025, four standard errors above 0.5.

    The cause is visible in `test_selection_on_noise_is_worse_than_random`: the
    winner's mean out-of-sample rank is 0.43, not 0.50. Selecting on the Sharpe
    ratio rewards low in-sample volatility as well as high in-sample mean, and
    only the mean has any chance of persisting. The volatility half is a fact
    about the split, so selecting on it actively hurts out of sample.

    A PBO reporting 0.0 here would certify every sweep it ever saw.
    """
    mean_pbo = _mean_pbo(lambda rng: _noise_trials(24, 1200, rng))
    assert 0.45 < mean_pbo < 0.80
    assert mean_pbo == pytest.approx(0.60, abs=0.15)


def test_selection_on_noise_is_worse_than_random():
    """The mechanism behind the 0.60 baseline, measured directly.

    A random pick would rank 0.5 out of sample on average. The in-sample winner
    ranks 0.43. That is the anti-persistent half of the Sharpe criterion, and it
    is why PBO's null baseline sits above one half.
    """
    ranks = []
    for seed in SEEDS:
        result = probability_of_backtest_overfitting(
            _noise_trials(24, 1200, random.Random(seed)), 10)
        ranks.extend(result.oos_ranks)
    assert fmean(ranks) < 0.48
    assert fmean(ranks) == pytest.approx(0.43, abs=0.06)


def test_pbo_is_near_zero_with_one_genuinely_better_strategy():
    """POSITIVE CONTROL. One real edge among noise must be found and kept.

    The strong trial has a per-observation Sharpe of ~0.15, comfortably visible
    in half the sample. If PBO cannot tell this apart from the null case above,
    it is not measuring anything.
    """
    def build(rng):
        trials = _noise_trials(23, 1200, rng)
        trials.append([rng.gauss(0.15, 1.0) for _ in range(1200)])
        return trials

    assert _mean_pbo(build) < 0.10


def test_the_two_controls_are_far_apart():
    """The instrument must SEPARATE the cases, not merely produce numbers."""
    null = _mean_pbo(lambda rng: _noise_trials(24, 1200, rng))

    def with_signal(rng):
        trials = _noise_trials(23, 1200, rng)
        trials.append([rng.gauss(0.18, 1.0) for _ in range(1200)])
        return trials

    assert null - _mean_pbo(with_signal) > 0.3


def test_degradation_slope_is_negatively_biased_even_on_a_real_edge():
    """The artifact documented on `performance_degradation`, pinned.

    Four genuinely profitable trials among noise. The slope still comes out
    firmly negative, because in-sample and out-of-sample split one fixed sample:
    a partition generous to a trial in-sample has left it a poorer out-of-sample
    half. Anyone reading a negative slope as evidence of overfitting would
    condemn this family, which contains four real signals.
    """
    rng = random.Random(777)
    trials = _noise_trials(15, 1500, rng)
    for drift in (0.05, 0.10, 0.15, 0.20):
        trials.append([rng.gauss(drift, 1.0) for _ in range(1500)])
    result = probability_of_backtest_overfitting(trials, n_splits=10)
    slope, _ = result.performance_degradation
    assert slope < 0
    # ...while PBO, which is immune to the artifact, correctly clears it.
    assert result.pbo < 0.10


def test_the_winner_actually_makes_money_when_the_edge_is_real():
    """The meaningful level check, in place of the slope."""
    rng = random.Random(777)
    trials = _noise_trials(15, 1500, rng)
    trials.append([rng.gauss(0.20, 1.0) for _ in range(1500)])
    result = probability_of_backtest_overfitting(trials, n_splits=10)
    assert fmean(result.oos_sharpes) > 0.05
    assert result.prob_oos_loss < 0.05


def test_the_noise_winner_usually_loses_money_out_of_sample():
    """Consistent with the 0.60 baseline: selection on noise is actively harmful.

    The in-sample winner posts a negative out-of-sample Sharpe on roughly 60% of
    partitions -- more often than the 50% a random pick would give, for the same
    reason its rank sits at 0.43.
    """
    values = [probability_of_backtest_overfitting(
        _noise_trials(24, 1200, random.Random(s)), 10).prob_oos_loss
        for s in SEEDS]
    assert fmean(values) > 0.45


def test_default_sixteen_splits_runs_and_agrees_with_ten():
    """C(16,8) = 12,870 partitions. Slower, and must not change the conclusion."""
    rng = random.Random(8080)
    trials = _noise_trials(12, 800, rng)
    coarse = probability_of_backtest_overfitting(trials, n_splits=10)
    fine = probability_of_backtest_overfitting(trials, n_splits=16)
    assert fine.n_combinations == 12_870
    assert abs(fine.pbo - coarse.pbo) < 0.25


def test_ranks_and_logits_agree_in_sign():
    rng = random.Random(1)
    result = probability_of_backtest_overfitting(_noise_trials(10, 600, rng), 10)
    for omega, lam in zip(result.oos_ranks, result.logits):
        assert (omega > 0.5) == (lam > 0)


def test_pbo_rejects_bad_input():
    rng = random.Random(2)
    good = _noise_trials(4, 400, rng)
    with pytest.raises(ValueError, match="at least 2 trials"):
        probability_of_backtest_overfitting(good[:1], 10)
    with pytest.raises(ValueError, match="equal length"):
        probability_of_backtest_overfitting([good[0], good[1][:100]], 10)
    with pytest.raises(ValueError, match="even and at least 4"):
        probability_of_backtest_overfitting(good, n_splits=7)
    with pytest.raises(ValueError, match="at least"):
        probability_of_backtest_overfitting([[0.1] * 10, [0.2] * 10], n_splits=10)
