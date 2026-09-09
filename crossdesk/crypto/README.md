# Dormant: crypto conventions

`calendar.py` (24/7 annualisation) and `funding.py` (perp funding sign, scale,
timing) were written on 2026-09-08 for a crypto-perpetuals thesis that was
abandoned the same day: the only reachable venue is Robinhood spot, which has
no perpetuals and therefore no funding rate.

They are correct and tested (36 tests) and cost nothing to keep. They become
live again the moment a CFTC-regulated perp or dated-futures account exists.
Until then nothing in the main package imports them.
