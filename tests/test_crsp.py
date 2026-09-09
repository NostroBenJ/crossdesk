"""The CRSP loader: windows, delisting folds, ticker reuse, and causal size ranking.

Fixtures are tiny CSVs written on the fly, so every assertion is about a row
whose answer is known by construction.
"""

from datetime import date

import pytest

from crossdesk.data.crsp import build, load, mktcap_table, top_by_mktcap
from crossdesk.data.panel import PricePanel
from crossdesk.data.universe import (
    Listing,
    PointInTimeUniverse,
    SurvivorshipError,
    assert_membership_is_causal,
)

D = date
HEADER = "permno,date,ret_adj,mktcap,dlstcd,dlret,ticker,comnam\n"


def _write(tmp_path, rows: str):
    p = tmp_path / "crsp.csv"
    p.write_text(HEADER + rows, encoding="utf-8")
    return str(p)


def _base_rows():
    # 10001: lives the whole sample.
    # 10002: acquired in 2020-03 with dlret recorded (+0.10 folded into ret_adj).
    # 10003: delisted for performance in 2020-03 with NO dlret -> Shumway -30%.
    # 10004: leaves the filter after 2020-02 (no code), returns in 2020-04.
    return (
        "10001,2020-01-31,,100,,,AAA,Alpha\n"
        "10001,2020-02-28,0.10,110,,,AAA,Alpha\n"
        "10001,2020-03-31,0.10,121,,,AAA,Alpha\n"
        "10001,2020-04-30,0.10,133,,,AAA,Alpha\n"
        "10002,2020-01-31,,50,,,BBB,Beta\n"
        "10002,2020-02-28,0.00,50,,,BBB,Beta\n"
        "10002,2020-03-31,0.10,55,231,0.10,BBB,Beta\n"
        "10003,2020-01-31,,20,,,CCC,Gamma\n"
        "10003,2020-02-28,-0.50,10,,,CCC,Gamma\n"
        "10003,2020-03-31,0.00,10,552,,CCC,Gamma\n"
        "10004,2020-01-31,,30,,,DDD,Delta\n"
        "10004,2020-02-28,0.00,30,,,DDD,Delta\n"
        "10004,2020-04-30,,30,,,DDD,Delta\n"
    )


def test_windows_and_open_ends(tmp_path):
    data = load(_write(tmp_path, _base_rows()))
    universe, panel, exits = build(data)
    by = {}
    for l in universe.listings:
        by.setdefault(l.symbol, []).append(l)
    assert by["10001"] == [Listing("10001", D(2020, 1, 31), None)]
    # Acquired during March: holdable through February, terminal = March's ret_adj.
    assert by["10002"][0].end == D(2020, 2, 28) and by["10002"][0].reason == "priced"
    assert by["10002"][0].delisting_return == pytest.approx(0.10)
    assert by["10003"][0].end == D(2020, 2, 28)
    assert by["10003"][0].delisting_return == pytest.approx(-0.30)
    # A gap splits a PERMNO into two windows; the second is open at the as-of date.
    assert [(l.start, l.end) for l in by["10004"]] == [
        (D(2020, 1, 31), D(2020, 2, 28)), (D(2020, 4, 30), None)
    ]
    assert exits["10002@2020-02-28"] == (231, True)
    assert exits["10004@2020-02-28"] == (0, False)


def test_total_return_index_compounds_ret_adj(tmp_path):
    data = load(_write(tmp_path, _base_rows()))
    _, panel, _ = build(data)
    assert panel.price("10001", D(2020, 1, 31)) == pytest.approx(100.0)
    assert panel.price("10001", D(2020, 4, 30)) == pytest.approx(133.1)
    value, how = panel.period_return("10001", D(2020, 1, 31), D(2020, 4, 30))
    assert how == "traded" and value == pytest.approx(0.331)


def test_acquired_name_pays_the_path_to_the_takeover(tmp_path):
    """dlret is already in ret_adj, so the exit value is the whole outcome."""
    data = load(_write(tmp_path, _base_rows()))
    _, panel, _ = build(data)
    value, how = panel.period_return("10002", D(2020, 1, 31), D(2020, 4, 30))
    assert how == "delisted"
    assert value == pytest.approx(0.10)     # 1.00 * 1.10 - 1


def test_missing_delisting_return_gets_the_shumway_haircut(tmp_path):
    data = load(_write(tmp_path, _base_rows()))
    _, panel, _ = build(data)
    assert data.shumway_adjusted == 1
    value, how = panel.period_return("10003", D(2020, 1, 31), D(2020, 4, 30))
    assert how == "delisted"
    assert value == pytest.approx(0.5 * 0.7 - 1.0)   # -50% then -30%


def test_merger_without_return_is_not_haircut(tmp_path):
    rows = _base_rows().replace("10003,2020-03-31,0.00,10,552,,", "10003,2020-03-31,0.00,10,231,,")
    data = load(_write(tmp_path, rows))
    _, panel, _ = build(data)
    assert data.shumway_adjusted == 1
    value, _ = panel.period_return("10003", D(2020, 1, 31), D(2020, 4, 30))
    assert value == pytest.approx(-0.5)


def test_period_returns_keep_every_member_including_the_dead(tmp_path):
    data = load(_write(tmp_path, _base_rows()))
    universe, panel, _ = build(data)
    returns, holes = panel.period_returns(D(2020, 1, 31), D(2020, 4, 30))
    assert holes == []
    assert set(returns) == {"10001", "10002", "10003", "10004"}
    # 10004 left the filter in March: exit at its last index value (100 -> 100).
    assert returns["10004"] == pytest.approx(0.0)


def test_same_ticker_two_permnos_are_two_series(tmp_path):
    rows = (
        "20001,2020-01-31,,10,,,XONE,Old Co\n"
        "20001,2020-02-28,-0.90,1,552,-0.5,XONE,Old Co\n"
        "20002,2020-03-31,,10,,,XONE,New Co\n"
        "20002,2020-04-30,0.50,15,,,XONE,New Co\n"
    )
    data = load(_write(tmp_path, rows))
    universe, panel, _ = build(data)
    assert universe.members_asof(D(2020, 1, 31)) == {"20001"}
    assert universe.members_asof(D(2020, 2, 28)) == set()     # delisted during Feb
    assert universe.members_asof(D(2020, 4, 30)) == {"20002"}
    assert data.labels["20001"][0] == data.labels["20002"][0] == "XONE"


def test_top_by_mktcap_is_causal_and_size_ranked(tmp_path):
    data = load(_write(tmp_path, _base_rows()))
    universe, _, _ = build(data)
    caps = mktcap_table(data)
    builder = top_by_mktcap(caps, 2)
    assert builder(universe.listings, D(2020, 1, 31)) == ["10001", "10002"]
    # In March 10004 is out of the filter and 10002/10003 delisted during the
    # month, so neither is holdable at the March month end.
    assert builder(universe.listings, D(2020, 3, 31)) == ["10001"]
    assert_membership_is_causal(builder, universe.listings, list(data.dates))


def test_survivor_filter_on_this_data_is_caught():
    """The guard must still bite on CRSP-shaped listings."""
    listings = [
        Listing("1", D(2020, 1, 31), None),
        Listing("2", D(2020, 1, 31), D(2020, 3, 31), reason="priced"),
    ]

    def survivors(ls, asof):
        return [l.symbol for l in ls if l.is_active(asof) and l.end is None]

    with pytest.raises(SurvivorshipError):
        assert_membership_is_causal(survivors, listings, [D(2020, 2, 28)])


# --------------------------------------------------------------------------
# the refined delisting arithmetic in PricePanel, both branches
# --------------------------------------------------------------------------

def _refined_panel():
    universe = PointInTimeUniverse([
        Listing("ACQ", D(2019, 1, 1), D(2020, 1, 5), reason="acquired"),
        Listing("BKR", D(2019, 1, 1), D(2020, 1, 5), reason="bankrupt"),
        Listing("PRC", D(2019, 1, 1), D(2020, 1, 5), reason="priced"),
    ])
    prices = {
        "ACQ": {D(2020, 1, 1): 100.0, D(2020, 1, 5): 120.0},
        "BKR": {D(2020, 1, 1): 100.0, D(2020, 1, 5): 20.0},
        "PRC": {D(2020, 1, 1): 100.0, D(2020, 1, 5): 20.0},
    }
    return PricePanel(prices, universe)


def test_acquisition_returns_the_path_to_the_last_price():
    value, how = _refined_panel().period_return("ACQ", D(2020, 1, 1), D(2020, 1, 10))
    assert how == "delisted" and value == pytest.approx(0.20)


def test_bankruptcy_is_a_total_loss_whatever_the_path():
    value, _ = _refined_panel().period_return("BKR", D(2020, 1, 1), D(2020, 1, 10))
    assert value == pytest.approx(-1.0)


def test_priced_exit_is_the_last_observed_value():
    value, _ = _refined_panel().period_return("PRC", D(2020, 1, 1), D(2020, 1, 10))
    assert value == pytest.approx(-0.80)


def test_exit_price_is_never_read_after_the_end_date():
    """A price dated after the listing ended (stale vendor row) must not be used."""
    universe = PointInTimeUniverse([Listing("X", D(2019, 1, 1), D(2020, 1, 5), reason="priced")])
    panel = PricePanel({"X": {D(2020, 1, 1): 100.0, D(2020, 1, 5): 50.0, D(2020, 1, 8): 500.0}}, universe)
    value, _ = panel.period_return("X", D(2020, 1, 1), D(2020, 1, 10))
    assert value == pytest.approx(-0.5)


def test_off_date_monthly_row_is_redated_to_the_common_month_end(tmp_path):
    """A halted name dated 03-27 must still be priced on the 03-31 rebalance."""
    rows = "\n".join([
        "30001,2020-02-28,,100,,,AAA,Alpha",
        "30001,2020-03-31,0.10,110,,,AAA,Alpha",
        "30002,2020-02-28,,100,,,HLT,Halted",
        "30002,2020-03-27,-0.20,80,,,HLT,Halted",
    ]) + "\n"
    data = load(_write(tmp_path, rows))
    assert data.dates == (D(2020, 2, 28), D(2020, 3, 31))
    universe, panel, _ = build(data)
    returns, holes = panel.period_returns(D(2020, 2, 28), D(2020, 3, 31))
    assert holes == []
    assert returns["30002"] == pytest.approx(-0.20)


def test_total_loss_delisting_is_minus_one_not_missing(tmp_path):
    """dlret = -1.0 must produce a -100% return, never a hole."""
    rows = "\n".join([
        "40001,2020-01-31,,100,,,ZZZ,Zero",
        "40001,2020-02-28,0.0,100,,,ZZZ,Zero",
        "40001,2020-03-31,-1.0,,574,-1.0,ZZZ,Zero",
    ]) + "\n"
    data = load(_write(tmp_path, rows))
    universe, panel, _ = build(data)
    assert universe.members_asof(D(2020, 2, 28)) == {"40001"}
    assert universe.members_asof(D(2020, 3, 31)) == set()
    value, how = panel.period_return("40001", D(2020, 2, 28), D(2020, 4, 30))
    assert how == "delisted" and value == pytest.approx(-1.0)
    returns, holes = panel.period_returns(D(2020, 1, 31), D(2020, 3, 31))
    assert holes == [] and returns["40001"] == pytest.approx(-1.0)


def test_listed_and_delisted_within_one_month_is_never_a_member(tmp_path):
    rows = "50001,2020-03-31,-0.4,10,552,-0.4,ONE,Oneshot\n"
    data = load(_write(tmp_path, rows))
    universe, panel, _ = build(data)
    assert universe.symbols == frozenset()
