"""Convert the WRDS pull (parquet) into the CSV `crossdesk.data.crsp` reads.

    python scripts/crsp_import.py [path/to/crsp_monthly.parquet]

The pull lives in the sibling repo (`dev/cisd-bot/data/crsp/`), written by
`research.wrds_pull --pull`. This keeps only the columns the panel needs and
writes `data/crsp_monthly.csv` here. pandas is imported inside the function so
the package stays importable without it.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_IN = os.path.join(ROOT, "..", "..", "dev", "cisd-bot", "data", "crsp", "crsp_monthly.parquet")
OUT = os.path.join(ROOT, "data", "crsp_monthly.csv")

COLUMNS = ["permno", "date", "ret_adj", "mktcap", "dlstcd", "dlret", "ticker", "comnam"]


def convert(src: str, dst: str = OUT) -> int:
    import pandas as pd

    if not os.path.exists(src):
        alt = src[:-len(".parquet")] + ".csv"
        if os.path.exists(alt):
            src = alt
        else:
            raise SystemExit(f"{src} not found -- run the pull first")
    df = pd.read_parquet(src) if src.endswith(".parquet") else pd.read_csv(src)
    df = df[COLUMNS].copy()
    df["permno"] = df["permno"].astype(int).astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["permno", "date"])
    df.to_csv(dst, index=False)
    print(f"{len(df):,} rows, {df.permno.nunique():,} PERMNOs, "
          f"{df.date.min()} .. {df.date.max()} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(convert(sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(DEFAULT_IN)))
