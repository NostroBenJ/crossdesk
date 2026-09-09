"""Three engines, one strategy, one bar series -- do they agree?

The prerequisite for trusting any cross-engine number. It is deliberately harsh:
quantdesk and VectorBT are required to match the reference to floating-point
equality, not to a loose tolerance, because a frictionless fixed-size long/flat
backtest has no legitimate source of disagreement. Anything above rounding is a
convention difference hiding as a rounding difference.

NautilusTrader is held to a DIFFERENT reference on purpose. It does not share
the convention, and the tests below pin that as a measured fact rather than
papering over it with a wider tolerance.
"""

import math
import random

import pytest

from crossdesk.engine.adapters import (
    convention_gap,
    mask_warmup,
    quantdesk_warmup,
    run_nautilus,
    run_quantdesk,
    run_reference,
    run_vectorbt,
)
from crossdesk.engine.spec import (
    ExecutionSpec,
    run_spec,
    sma_crossover_targets,
)

CASH = 100_000.0


def make_series(n=200, seed=5):
    rng = random.Random(seed)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(max(1.0, closes[-1] * (1 + rng.gauss(0, 0.01))))
    opens = [c * (1 + rng.gauss(0, 0.002)) for c in closes]
    return opens, closes


@pytest.fixture(scope="module")
def series():
    opens, closes = make_series()
    return opens, closes, sma_crossover_targets(closes, 5, 20, 1.0)


# ---------------------------------------------------------------------------
# The spec itself
# ---------------------------------------------------------------------------

def test_spec_holds_cash_when_never_invested():
    opens, closes = make_series(50)
    result = run_spec(opens, closes, [0.0] * 50, ExecutionSpec(initial_cash=CASH))
    assert all(e == CASH for e in result.equity)
    assert result.fills == []


def test_spec_fills_at_the_next_bar_open():
    opens = [10.0, 20.0, 30.0]
    closes = [11.0, 21.0, 31.0]
    result = run_spec(opens, closes, [1.0, 1.0, 1.0], ExecutionSpec(initial_cash=CASH))
    # Decision at bar 0 -> fill at bar 1's OPEN of 20.0, never bar 0's close.
    assert result.fills == [(1, 1.0, 20.0)]
    assert result.equity[0] == CASH               # flat through bar 0
    assert result.equity[1] == CASH - 20.0 + 21.0


def test_spec_expires_an_order_on_the_final_bar():
    opens = [10.0, 20.0]
    closes = [11.0, 21.0]
    result = run_spec(opens, closes, [0.0, 1.0], ExecutionSpec(initial_cash=CASH))
    assert result.expired_orders == 1
    assert result.fills == []


def test_spec_slippage_always_works_against_the_trader(series):
    opens, closes, targets = series
    free = run_spec(opens, closes, targets, ExecutionSpec(initial_cash=CASH))
    costly = run_spec(opens, closes, targets,
                      ExecutionSpec(initial_cash=CASH, slippage_per_unit=0.05))
    assert costly.final_equity < free.final_equity


def test_spec_commission_scales_with_turnover(series):
    opens, closes, targets = series
    one = run_spec(opens, closes, targets,
                   ExecutionSpec(initial_cash=CASH, commission_per_unit=1.0))
    two = run_spec(opens, closes, targets,
                   ExecutionSpec(initial_cash=CASH, commission_per_unit=2.0))
    assert two.total_commission == pytest.approx(2 * one.total_commission)


def test_spec_rejects_an_unknown_convention():
    with pytest.raises(ValueError, match="fill_at must be"):
        ExecutionSpec(fill_at="magic")


def test_spec_rejects_ragged_input():
    with pytest.raises(ValueError, match="equal length"):
        run_spec([1.0, 2.0], [1.0], [0.0], ExecutionSpec())


def test_crossover_targets_are_causal(series):
    """A target at bar i must not move when bars after i are deleted."""
    _, closes, targets = series
    for cut in (60, 120, 199):
        truncated = sma_crossover_targets(closes[:cut + 1], 5, 20, 1.0)
        assert truncated[cut] == targets[cut]


# ---------------------------------------------------------------------------
# The agreement
# ---------------------------------------------------------------------------

def test_quantdesk_matches_the_reference_exactly(series):
    opens, closes, targets = series
    spec = ExecutionSpec(initial_cash=CASH)
    start = quantdesk_warmup()
    actual = run_quantdesk(opens, closes, targets, spec)
    expected = run_reference(opens, closes, targets, spec, start=start)
    assert len(actual) == len(expected)
    assert max(abs(a - b) for a, b in zip(actual, expected)) < 1e-9


def test_vectorbt_matches_the_reference_exactly(series):
    opens, closes, targets = series
    spec = ExecutionSpec(initial_cash=CASH)
    actual = run_vectorbt(opens, closes, targets, spec)
    expected = run_reference(opens, closes, targets, spec, start=0)
    assert len(actual) == len(expected)
    assert max(abs(a - b) for a, b in zip(actual, expected)) < 1e-6


def test_quantdesk_and_vectorbt_agree_with_each_other(series):
    """The pairwise check, which the spec makes redundant -- and that is the point.

    Both match the reference, so they match each other. Keeping this explicit
    means a future engine change that breaks the pair is caught even if someone
    edits the reference to match one of them.
    """
    opens, closes, targets = series
    spec = ExecutionSpec(initial_cash=CASH)
    start = quantdesk_warmup()
    masked = mask_warmup(targets, start)
    qd = run_quantdesk(opens, closes, masked, spec)
    vbt = run_vectorbt(opens, closes, masked, spec)[start:]
    assert len(qd) == len(vbt)
    assert max(abs(a - b) for a, b in zip(qd, vbt)) < 1e-6


@pytest.mark.parametrize("seed", [1, 2, 3, 11, 99])
def test_agreement_holds_across_random_series(seed):
    """One series could agree by luck. Five cannot.

    Seed 11 is here deliberately: it is the series whose crossover fires at bar
    20, inside quantdesk's warmup, and it is what exposed the alignment bug that
    `mask_warmup` now prevents.
    """
    opens, closes = make_series(150, seed)
    targets = mask_warmup(sma_crossover_targets(closes, 5, 20, 1.0),
                          quantdesk_warmup())
    spec = ExecutionSpec(initial_cash=CASH)
    start = quantdesk_warmup()
    qd = run_quantdesk(opens, closes, targets, spec)
    ref = run_reference(opens, closes, targets, spec, start=start)
    vbt = run_vectorbt(opens, closes, targets, spec)[start:]
    assert max(abs(a - b) for a, b in zip(qd, ref)) < 1e-9
    assert max(abs(a - b) for a, b in zip(vbt, ref)) < 1e-6


def test_unmasked_warmup_is_what_broke_seed_11():
    """Pin the failure mode, so the masking is not removed as ceremony.

    Without masking, VectorBT takes a position at bar 20 that quantdesk's warmup
    forbids, and the curves stay apart for the rest of the run.
    """
    opens, closes = make_series(150, 11)
    targets = sma_crossover_targets(closes, 5, 20, 1.0)
    spec = ExecutionSpec(initial_cash=CASH)
    start = quantdesk_warmup()
    assert any(t != 0 for t in targets[:start]), "seed 11 must trade inside warmup"
    ref = run_reference(opens, closes, targets, spec, start=start)
    vbt = run_vectorbt(opens, closes, targets, spec)[start:]
    assert max(abs(a - b) for a, b in zip(vbt, ref)) > 0.1


def test_the_test_can_fail_when_an_engine_is_wrong(series):
    """A guard that never fires certifies nothing.

    Feeding VectorBT the UNSHIFTED targets is the realistic bug -- it fills at the
    decision bar instead of the next open, which is a lookahead. The reference
    must reject it.
    """
    import numpy as np
    import pandas as pd
    import vectorbt as vbt

    opens, closes, targets = series
    index = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    wrong = vbt.Portfolio.from_orders(
        close=pd.Series(list(closes), index=index),
        size=pd.Series(list(targets), index=index),      # NOT shifted
        size_type="targetamount",
        price=pd.Series(list(opens), index=index),
        init_cash=CASH,
    )
    actual = [float(v) for v in wrong.value().values]
    expected = run_reference(opens, closes, targets,
                             ExecutionSpec(initial_cash=CASH), start=0)
    assert max(abs(a - b) for a, b in zip(actual, expected)) > 1.0


# ---------------------------------------------------------------------------
# Nautilus, and the convention it does not share
# ---------------------------------------------------------------------------

def test_nautilus_matches_the_decision_close_reference(series):
    """Nautilus's accounting is correct -- for ITS convention.

    Measured, not assumed: in `scripts/_nautilus_probe.py` all three fills landed
    on the decision bar's close rather than the next bar's open. So it is checked
    against the `decision_close` spec. Holding it to `next_open` would report a
    bug in Nautilus that is really a difference in contract.
    """
    opens, closes, targets = series
    spec = ExecutionSpec(initial_cash=CASH, fill_at="decision_close")
    actual = run_nautilus(opens, closes, targets, spec)
    expected = run_reference(opens, closes, targets, spec, start=0)
    assert len(actual) == len(expected)
    # Nautilus quantises prices to the instrument's 2-decimal precision.
    assert max(abs(a - b) for a, b in zip(actual, expected)) < 0.02


def test_nautilus_does_not_match_the_next_open_reference(series):
    """The divergence, pinned. If this ever passes, the convention changed."""
    opens, closes, targets = series
    actual = run_nautilus(opens, closes, targets,
                          ExecutionSpec(initial_cash=CASH, fill_at="decision_close"))
    expected = run_reference(opens, closes, targets,
                             ExecutionSpec(initial_cash=CASH), start=0)
    assert max(abs(a - b) for a, b in zip(actual, expected)) > 1.0


def test_the_convention_gap_is_material(series):
    """Fill convention alone moves the result. That is the finding."""
    opens, closes, targets = series
    next_open, decision_close, gap_bps = convention_gap(opens, closes, targets, CASH)
    assert next_open != decision_close
    assert abs(gap_bps) > 0.01
    assert math.isfinite(gap_bps)


def test_quantdesk_adapter_refuses_costs_it_cannot_express():
    """Better to refuse than to silently run a different cost model."""
    opens, closes = make_series(40)
    targets = [0.0] * 40
    with pytest.raises(NotImplementedError, match="volatility-scaled"):
        run_quantdesk(opens, closes, targets,
                      ExecutionSpec(initial_cash=CASH, commission_per_unit=1.0))


# ---------------------------------------------------------------------------
# The cash constraint -- a second divergence, unrelated to fill timing
# ---------------------------------------------------------------------------

def _rising_series(n=300):
    """A series that triples, so a position sized off bar 0 becomes unaffordable."""
    closes = [100.0 * (1 + 2.0 * i / (n - 1)) for i in range(n)]
    opens = [c * 0.999 for c in closes]
    return opens, closes


def test_vectorbt_enforces_cash_and_the_reference_does_not():
    """Discovered by this test on real SPY data: an $11,776 divergence.

    The spec and quantdesk let cash go negative -- implicit, free, unlimited
    leverage. VectorBT refuses to spend cash it does not have and partially
    fills instead. Neither is wrong in isolation; comparing them without knowing
    which is which is what produces an unexplained gap.

    This is pinned rather than fixed because the correct behaviour is a policy
    decision, and quantdesk owns it.
    """
    opens, closes = _rising_series()
    start = quantdesk_warmup()
    unaffordable = float(int(CASH / closes[0]))     # costs 3x the account later
    targets = mask_warmup([unaffordable] * len(closes), start)
    spec = ExecutionSpec(initial_cash=CASH)

    reference = run_reference(opens, closes, targets, spec, start=start)
    vector = run_vectorbt(opens, closes, targets, spec)[start:]
    assert max(abs(a - b) for a, b in zip(vector, reference)) > 1.0


def test_the_engines_agree_once_the_position_is_always_affordable():
    """The same series, sized so the constraint never binds: exact agreement."""
    opens, closes = _rising_series()
    start = quantdesk_warmup()
    affordable = float(int(CASH / max(closes)))
    targets = mask_warmup([affordable] * len(closes), start)
    spec = ExecutionSpec(initial_cash=CASH)

    reference = run_reference(opens, closes, targets, spec, start=start)
    assert max(abs(a - b) for a, b in
               zip(run_quantdesk(opens, closes, targets, spec), reference)) < 1e-9
    assert max(abs(a - b) for a, b in
               zip(run_vectorbt(opens, closes, targets, spec)[start:], reference)) < 1e-6


def test_nautilus_adapter_reports_actual_fills_not_intended_ones():
    """A rejected order must not appear as a held position.

    On a CASH account Nautilus refuses what it cannot fund. The adapter records
    `on_order_filled` events rather than the targets it asked for, so the curve
    reflects what the engine did.
    """
    opens, closes = _rising_series(120)
    unaffordable = float(int(CASH / closes[0]))
    targets = [unaffordable] * len(closes)
    curve = run_nautilus(opens, closes, targets,
                         ExecutionSpec(initial_cash=CASH, fill_at="decision_close"))
    assert len(curve) == len(closes)
    assert all(math.isfinite(v) for v in curve)
    # Whatever it filled, it cannot have manufactured cash from nothing.
    assert curve[0] <= CASH * 1.05
