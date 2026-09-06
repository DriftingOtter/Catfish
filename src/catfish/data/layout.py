import re
from pathlib import Path

from catfish.paths import PROJECT_ROOT

OHLCV_INTERVALS = ("per_second", "per_minute", "per_hour", "per_day", "per_month")

FILING_PATTERN = re.compile(
    r"^(.+)-(10-K|10-Q|8-K)-(\d{4})-(\d{2})-(\d{2})\.txt$"
)


def ohlcv_root():
    return PROJECT_ROOT / "datasets" / "OHLCV"


def ohlcv_path(ticker, interval="per_day", session_date=None):
    ticker = ticker.lower().strip()
    if interval not in OHLCV_INTERVALS:
        raise ValueError(f"interval must be one of {OHLCV_INTERVALS}")
    if session_date:
        name = f"{ticker}__{session_date}.csv"
    else:
        name = f"{ticker}__.csv"
    return ohlcv_root() / interval / name


def parse_ticker(path):
    stem = Path(path).stem
    return stem.split("__")[0].upper()


def sec_dir():
    return PROJECT_ROOT / "datasets" / "SEC"


def sec_filings_for(ticker):
    ticker = ticker.upper().strip()
    root   = sec_dir()
    if not root.is_dir():
        return []

    filings = []
    for path in sorted(root.glob("*.txt")):
        match = FILING_PATTERN.match(path.name)
        if match and match.group(1) == ticker:
            filings.append(path)
    return filings
