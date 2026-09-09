"""Funding scale, sign, and timing.

Sign errors are the dangerous ones: they invert the strategy while leaving an
equity curve that still looks like a strategy. So the sign is tested in all four
combinations of position side and rate side, plus a conservation identity.
"""

from datetime import datetime, timedelta, timezone

import pytest

from crossdesk.crypto.funding import (
    FundingEvent,
    annualise,
    cashflow,
    intervals_per_day,
    is_settlement,
)

UTC = timezone.utc


def test_intervals_per_day():
    assert intervals_per_day(8) == 3
    assert intervals_per_day(1) == 24
    assert intervals_per_day(4) == 6


@pytest.mark.parametrize("bad", [0, -8, 5, 7, 24 * 2])
def test_interval_must_divide_the_day(bad):
    """A 5-hour funding interval does not tile a UTC day. Refuse it."""
    with pytest.raises(ValueError):
        intervals_per_day(bad)


def test_annualise_the_headline_number():
    """0.01% per 8 hours is 10.95%/yr -- the number that makes carry visible."""
    assert annualise(0.0001) == pytest.approx(0.1095, abs=1e-12)


def test_annualise_is_linear_and_signed():
    assert annualise(-0.0001) == pytest.approx(-0.1095, abs=1e-12)
    assert annualise(0.0) == 0.0
    assert annualise(0.0002) == pytest.approx(2 * annualise(0.0001), rel=1e-15)


def test_annualise_respects_interval_length():
    """The same quoted rate is worth 8x more at a 1h interval than at 8h."""
    assert annualise(0.0001, 1) == pytest.approx(8 * annualise(0.0001, 8), rel=1e-15)


@pytest.mark.parametrize("qty, rate, expected_sign, who", [
    (+1.0, +0.0001, -1, "long pays when funding is positive"),
    (-1.0, +0.0001, +1, "short receives when funding is positive"),
    (+1.0, -0.0001, +1, "long receives when funding is negative"),
    (-1.0, -0.0001, -1, "short pays when funding is negative"),
])
def test_sign_convention(qty, rate, expected_sign, who):
    pnl = cashflow(qty, 50_000.0, rate)
    assert pnl != 0
    assert (pnl > 0) == (expected_sign > 0), who


def test_funding_is_conserved_between_the_two_sides():
    """Funding is a transfer, not a source of money.

    Equal and opposite positions must net to exactly zero at any rate. This is
    the identity that catches a sign or scale bug applied to only one side.
    """
    for rate in (0.0001, -0.0003, 0.0, 0.0075):
        long = cashflow(+2.5, 41_234.5, rate)
        short = cashflow(-2.5, 41_234.5, rate)
        assert long + short == pytest.approx(0.0, abs=1e-12)


def test_cashflow_scales_with_notional():
    base = cashflow(1.0, 50_000.0, 0.0001)
    assert cashflow(2.0, 50_000.0, 0.0001) == pytest.approx(2 * base, rel=1e-15)
    assert cashflow(1.0, 100_000.0, 0.0001) == pytest.approx(2 * base, rel=1e-15)


def test_flat_position_never_pays_or_receives():
    assert cashflow(0.0, 50_000.0, 0.05) == 0.0


@pytest.mark.parametrize("hour, ok", [(0, True), (8, True), (16, True),
                                      (1, False), (7, False), (23, False)])
def test_settlement_hours(hour, ok):
    assert is_settlement(datetime(2026, 3, 1, hour, tzinfo=UTC)) is ok


def test_settlement_requires_the_exact_instant():
    """One second past the boundary is not a settlement.

    A bar CONTAINING the boundary is not the same as the boundary. Treating it
    as one accrues funding to positions opened after the payment.
    """
    assert is_settlement(datetime(2026, 3, 1, 8, 0, 0, tzinfo=UTC)) is True
    assert is_settlement(datetime(2026, 3, 1, 8, 0, 1, tzinfo=UTC)) is False
    assert is_settlement(datetime(2026, 3, 1, 7, 59, 59, tzinfo=UTC)) is False
    assert is_settlement(datetime(2026, 3, 1, 8, 0, 0, 1, tzinfo=UTC)) is False


def test_settlement_rejects_naive_datetimes():
    """Inherited from quantdesk's ensure_utc: naive timestamps are refused."""
    with pytest.raises(ValueError):
        is_settlement(datetime(2026, 3, 1, 8, 0, 0))


def test_settlement_is_utc_not_local():
    """16:00 UTC is a settlement; 16:00 in a +05:00 zone is 11:00 UTC and is not."""
    plus5 = timezone(timedelta(hours=5))
    assert is_settlement(datetime(2026, 3, 1, 16, tzinfo=UTC)) is True
    assert is_settlement(datetime(2026, 3, 1, 16, tzinfo=plus5)) is False
    # ...and the instant that IS 16:00 UTC, expressed in +05:00, still counts.
    assert is_settlement(datetime(2026, 3, 1, 21, tzinfo=plus5)) is True


def test_funding_event_pnl_and_annualised():
    ev = FundingEvent(datetime(2026, 3, 1, 8, tzinfo=UTC), "BTCUSDT", 0.0001, 50_000.0)
    assert ev.pnl(+1.0) == pytest.approx(-5.0)
    assert ev.pnl(-1.0) == pytest.approx(+5.0)
    assert ev.annualised == pytest.approx(0.1095, abs=1e-12)


def test_funding_event_rejects_bad_input():
    with pytest.raises(ValueError):
        FundingEvent(datetime(2026, 3, 1, 8, tzinfo=UTC), "BTCUSDT", 0.0001, 0.0)
    with pytest.raises(ValueError):
        FundingEvent(datetime(2026, 3, 1, 8), "BTCUSDT", 0.0001, 50_000.0)
