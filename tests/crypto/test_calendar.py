"""The annualisation constants, pinned to exact values.

These are the numbers that silently corrupt every volatility estimate and every
position size downstream, so they are asserted exactly rather than approximately.
"""

import math

import pytest

from crossdesk.crypto.calendar import DAY_1, HOUR_1, MINUTE_1, MINUTE_5, crypto
from quantdesk.core.clock import MINUTE_5 as EQUITY_MINUTE_5


@pytest.mark.parametrize("tf, expected", [
    (MINUTE_1, 525_600),
    (MINUTE_5, 105_120),
    (HOUR_1, 8_760),
    (DAY_1, 365),
])
def test_periods_per_year_exact(tf, expected):
    assert tf.periods_per_year == pytest.approx(expected, rel=0, abs=1e-9)


def test_daily_is_365_not_252():
    """The equity default is 252. Perps do not close, so the daily count is 365.

    Using 252 on a 24/7 series understates annual vol by sqrt(365/252) = 1.20x.
    """
    assert DAY_1.periods_per_year == 365
    assert DAY_1.periods_per_year != 252


def test_the_5m_trap_is_2_31x():
    """The specific error this module exists to prevent.

    quantdesk's equity 5-minute Timeframe annualises off a 6.5h session and 252
    days. Applying it to 24/7 bars understates vol by this factor.
    """
    assert EQUITY_MINUTE_5.periods_per_year == pytest.approx(19_656)
    ratio = MINUTE_5.periods_per_year / EQUITY_MINUTE_5.periods_per_year
    understated_by = math.sqrt(ratio)
    assert understated_by == pytest.approx(2.312, abs=0.001)


def test_annualisation_factor_is_sqrt_of_periods():
    """Structural identity: the vol multiplier is sqrt(periods_per_year)."""
    for tf in (MINUTE_1, MINUTE_5, HOUR_1, DAY_1):
        assert tf.annualisation_factor == pytest.approx(
            math.sqrt(tf.periods_per_year), rel=1e-15
        )


def test_bars_tile_the_year_consistently():
    """Every timeframe must describe the same year: bars * seconds = 365 days."""
    seconds_in_year = 365 * 86_400
    for tf in (MINUTE_1, MINUTE_5, HOUR_1, DAY_1, crypto("15m"), crypto("4h")):
        assert tf.periods_per_year * tf.seconds == pytest.approx(seconds_in_year)


def test_crypto_calendar_does_not_mutate_the_equity_default():
    """Building a crypto Timeframe must not disturb quantdesk's shared constants."""
    assert EQUITY_MINUTE_5.sessions_per_year == 252
    assert EQUITY_MINUTE_5.seconds_per_session == 23_400
    assert MINUTE_5.sessions_per_year == 365
    assert MINUTE_5.seconds_per_session == 86_400
