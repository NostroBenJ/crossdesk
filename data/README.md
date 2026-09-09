# data/

Nothing here is committed. Both files are vendor extracts: one is licensed and
must not be redistributed, the other is a free cache that is cheaper to refetch
than to version.

## crsp_monthly.csv — CRSP monthly stock file, 1990-01 .. 2024-12

135MB, ~2.1M rows, 18,695 PERMNOs. Ordinary common shares (SHRCD 10/11) on
NYSE, AMEX and NASDAQ (EXCHCD 1-3), with delisting returns folded into
`ret_adj`. **Licensed through WRDS — do not commit it, and do not send it
anywhere.**

To rebuild, with a WRDS account whose credentials are in
`%APPDATA%\postgresql\pgpass.conf`:

```
cd ../../dev/cisd-bot
../../Downloads/crossdesk/.venv-qlib/Scripts/python -m research.wrds_pull --pull --start 1990-01-01
cd ../../Downloads/crossdesk
.venv/Scripts/python scripts/crsp_import.py
```

Two traps that cost an evening and are documented in `CLAUDE.md`:

* the pull must run on **Python 3.12** (`.venv-qlib`) — on 3.14 the driver
  segfaults on any query result while `list_libraries()` still works, so the
  connection looks healthy right up until it returns nothing;
* the classic `msf` x `msenames` join **drops the delisting month**, which is
  the one row that carries `dlret`. `MONTHLY_SQL` extends `nameendt` to its
  month end and de-duplicates. On 1990 alone the naive join kept 49 of 515
  delisting returns.

## tiingo_tickers.csv — Tiingo's listing table

5MB, free, no key: `https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip`.
Point-in-time membership for US stocks and ETFs including delisted ones. Kept
from the survey that preceded the CRSP pull; CRSP supersedes it for the
equity universe, but it remains the only free source that knows which ETFs
have closed.
