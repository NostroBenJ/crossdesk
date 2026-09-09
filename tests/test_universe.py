"""Point-in-time membership and the guards over it.

The test that matters most here is `test_causality_guard_catches_a_real_leak`.
A guard that passes everything is indistinguishable from no guard, so the suite
proves this one FAILS on a builder with a known leak before it is allowed to
certify anything as clean.
"""

from datetime import date

import pytest

from crossdesk.data.universe import (
    DELISTING_RETURNS,
    Listing,
    PointInTimeUniverse,
    SurvivorshipError,
    as_known_on,
    assert_delistings_priced,
    assert_membership_is_causal,
    assert_no_survivorship,
    survivorship_leak,
)

D = date


def _listings():
    return [
        Listing("ALWAYS", D(2000, 1, 1)),
        Listing("LATE", D(2015, 6, 1)),
        Listing("GONE", D(2000, 1, 1), D(2010, 3, 15), reason="bankrupt"),
        Listing("BOUGHT", D(2000, 1, 1), D(2012, 8, 1), reason="acquired"),
    ]


# --------------------------------------------------------------------------
# Listing validation
# --------------------------------------------------------------------------

def test_ended_listing_must_state_its_terminal_outcome():
    """The core invariant: you cannot record a delisting without pricing it."""
    with pytest.raises(ValueError, match="terminal outcome"):
        Listing("X", D(2000, 1, 1), D(2005, 1, 1))


def test_reason_supplies_the_delisting_return():
    assert Listing("X", D(2000, 1, 1), D(2005, 1, 1),
                   reason="bankrupt").delisting_return == -1.0
    assert Listing("X", D(2000, 1, 1), D(2005, 1, 1),
                   reason="acquired").delisting_return == 0.0


def test_explicit_delisting_return_overrides_the_default():
    lst = Listing("X", D(2000, 1, 1), D(2005, 1, 1), reason="bankrupt",
                  delisting_return=-0.85)
    assert lst.delisting_return == -0.85


def test_unknown_reason_is_refused():
    with pytest.raises(ValueError, match="unknown delisting reason"):
        Listing("X", D(2000, 1, 1), D(2005, 1, 1), reason="vanished")


def test_live_listing_cannot_carry_a_delisting_return():
    with pytest.raises(ValueError, match="still listed"):
        Listing("X", D(2000, 1, 1), None, delisting_return=-1.0)


def test_end_before_start_is_refused():
    with pytest.raises(ValueError, match="precedes start"):
        Listing("X", D(2005, 1, 1), D(2000, 1, 1), reason="acquired")


def test_bankruptcy_is_a_total_loss_and_acquisition_is_not():
    assert DELISTING_RETURNS["bankrupt"] == -1.0
    assert DELISTING_RETURNS["acquired"] == 0.0


# --------------------------------------------------------------------------
# Membership
# --------------------------------------------------------------------------

def test_members_asof_respects_both_ends_of_the_window():
    u = PointInTimeUniverse(_listings())
    assert u.members_asof(D(2005, 1, 1)) == {"ALWAYS", "GONE", "BOUGHT"}
    assert u.members_asof(D(2011, 1, 1)) == {"ALWAYS", "BOUGHT"}
    assert u.members_asof(D(2016, 1, 1)) == {"ALWAYS", "LATE"}
    assert u.members_asof(D(1999, 1, 1)) == frozenset()


def test_membership_is_inclusive_of_the_end_date():
    u = PointInTimeUniverse(_listings())
    assert "GONE" in u.members_asof(D(2010, 3, 15))
    assert "GONE" not in u.members_asof(D(2010, 3, 16))


def test_symbols_is_every_name_ever_and_not_a_tradable_set():
    u = PointInTimeUniverse(_listings())
    assert u.symbols == {"ALWAYS", "LATE", "GONE", "BOUGHT"}
    assert u.members_asof(D(2016, 1, 1)) != u.symbols


def test_overlapping_windows_for_one_symbol_are_refused():
    with pytest.raises(ValueError, match="overlapping listing windows"):
        PointInTimeUniverse([
            Listing("X", D(2000, 1, 1), D(2010, 1, 1), reason="acquired"),
            Listing("X", D(2005, 1, 1), D(2015, 1, 1), reason="acquired"),
        ])


def test_relisting_after_a_gap_is_allowed():
    u = PointInTimeUniverse([
        Listing("X", D(2000, 1, 1), D(2010, 1, 1), reason="acquired"),
        Listing("X", D(2012, 1, 1)),
    ])
    assert u.members_asof(D(2011, 1, 1)) == frozenset()
    assert u.members_asof(D(2013, 1, 1)) == {"X"}


def test_delisting_return_is_found_only_on_the_exact_end_date():
    u = PointInTimeUniverse(_listings())
    assert u.delisting_return("GONE", D(2010, 3, 15)) == -1.0
    assert u.delisting_return("GONE", D(2010, 3, 14)) is None


# --------------------------------------------------------------------------
# Leak detection
# --------------------------------------------------------------------------

def test_survivorship_leak_names_the_offending_symbols():
    u = PointInTimeUniverse(_listings())
    # In 2005 "LATE" had not listed and everything else had.
    leaked = survivorship_leak(["ALWAYS", "LATE", "GONE"], u, D(2005, 1, 1))
    assert leaked == {"LATE"}
    # In 2016 "GONE" and "BOUGHT" are long dead.
    leaked = survivorship_leak(["ALWAYS", "GONE", "BOUGHT"], u, D(2016, 1, 1))
    assert leaked == {"GONE", "BOUGHT"}


def test_survivors_only_is_the_wrong_universe_and_says_so():
    """`survivors_only` at the END of the sample is the classic biased set."""
    u = PointInTimeUniverse(_listings())
    assert u.survivors_only(D(2026, 1, 1)) == {"ALWAYS", "LATE"}
    # Applied to 2005 it would trade LATE, which did not exist yet.
    assert survivorship_leak(u.survivors_only(D(2026, 1, 1)), u, D(2005, 1, 1))


def test_assert_no_survivorship_passes_on_correct_membership():
    u = PointInTimeUniverse(_listings())
    dates = [D(2005, 1, 1), D(2011, 1, 1), D(2016, 1, 1)]
    assert_no_survivorship({d: u.members_asof(d) for d in dates}, u)


def test_assert_no_survivorship_fails_and_reports_the_dates():
    u = PointInTimeUniverse(_listings())
    survivors = u.survivors_only(D(2026, 1, 1))
    with pytest.raises(SurvivorshipError, match="universe leak"):
        assert_no_survivorship(
            {D(2005, 1, 1): survivors, D(2011, 1, 1): survivors}, u
        )


def test_assert_delistings_priced_accepts_a_clean_universe():
    assert_delistings_priced(PointInTimeUniverse(_listings()))


# --------------------------------------------------------------------------
# The causality guard -- proven to be able to fail
# --------------------------------------------------------------------------

def _honest_builder(listings, asof):
    """Uses only start/end against the as-of date."""
    return [l.symbol for l in listings if l.is_active(asof)]


def _survivor_builder(listings, asof):
    """THE BUG: 'only trade names that never delisted.'

    Reads every listing's end date -- including ends far in the future of `asof`
    -- to decide what to hold today. Looks like a quality filter. Is a leak.
    """
    return [l.symbol for l in listings if l.is_active(asof) and l.end is None]


def _min_history_builder(listings, asof):
    """THE OTHER BUG: 'only names with a long enough total history.'

    Total history includes history that has not happened yet on `asof`.
    """
    out = []
    for l in listings:
        if not l.is_active(asof):
            continue
        end = l.end or D(2026, 9, 1)
        if (end - l.start).days > 365 * 8:
            out.append(l.symbol)
    return out


DATES = [D(2005, 1, 1), D(2009, 1, 1), D(2011, 1, 1), D(2016, 1, 1)]


def test_causality_guard_passes_an_honest_builder():
    assert_membership_is_causal(_honest_builder, _listings(), DATES)


def test_causality_guard_catches_a_real_leak():
    """The guard must FAIL here, or it certifies nothing.

    In 2005 both GONE and BOUGHT were trading and nobody knew they would end.
    The survivor builder drops them anyway, because it can see 2010 and 2012.
    """
    with pytest.raises(SurvivorshipError, match="depends on post-"):
        assert_membership_is_causal(_survivor_builder, _listings(), DATES)


def test_causality_guard_catches_the_min_history_filter():
    """A 'needs 8 years of history' filter reads the future for young names.

    Needs a genuinely short-lived name to trip on. SHORTLIVED trades from 2009
    to 2011. On 2010-01-01 nobody knew it had only two years in it, so an honest
    filter must include it; the min-history filter sees the 2011 end and drops
    it. Every name in `_listings()` happens to live longer than eight years
    either way, which is why the leak does not show there.
    """
    listings = _listings() + [
        Listing("SHORTLIVED", D(2009, 1, 1), D(2011, 1, 1), reason="bankrupt")
    ]
    with pytest.raises(SurvivorshipError, match="depends on post-"):
        assert_membership_is_causal(_min_history_builder, listings,
                                    [D(2010, 1, 1)])


def test_min_history_filter_is_clean_when_no_name_is_short_lived():
    """The same builder passes when the filter never binds differently.

    Kept to make the previous test meaningful: it shows the guard is responding
    to the leak itself, not merely to the builder's identity.
    """
    assert_membership_is_causal(_min_history_builder, _listings(), DATES)


def test_the_leak_report_names_the_symbols():
    with pytest.raises(SurvivorshipError) as exc:
        assert_membership_is_causal(_survivor_builder, _listings(),
                                    [D(2005, 1, 1)])
    message = str(exc.value)
    assert "GONE" in message and "BOUGHT" in message


def test_as_known_on_opens_ends_that_had_not_happened():
    known = {l.symbol: l for l in as_known_on(_listings(), D(2005, 1, 1))}
    assert set(known) == {"ALWAYS", "GONE", "BOUGHT"}   # LATE not yet listed
    assert known["GONE"].end is None                     # its 2010 end is unknowable
    assert known["BOUGHT"].end is None


def test_as_known_on_keeps_ends_that_already_happened():
    known = {l.symbol: l for l in as_known_on(_listings(), D(2011, 1, 1))}
    assert known["GONE"].end == D(2010, 3, 15)
    assert known["GONE"].delisting_return == -1.0
    assert known["BOUGHT"].end is None                   # 2012 is still the future
