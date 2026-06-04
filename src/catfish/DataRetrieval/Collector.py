#!/usr/bin/env python3

import argparse
import enum
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ET:       final = ZoneInfo("America/New_York")
FMT:      final = "%Y-%m-%d %H:%M:%S"
PRICE_DP: final = 6
LOOKBACK: final = 30
INTERVAL: final = 62
OPEN:     final = (9, 30)
CLOSE:    final = (16, 0)
COLUMNS:  final = ["Datetime", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


class Feed(enum.Enum):
    SIP = "sip"
    IEX = "iex"


class Adjust(enum.Enum):
    # Raw:   unadjusted prices. Adj Close is undefined — written as NaN.
    # Split: split-adjusted prices. Adj Close = Close (split factor already applied).
    # All:   split + dividend adjusted. Adj Close = Close (both factors applied).
    #        This matches Yahoo Finance's convention for Adj Close.
    Raw   = "raw"
    Split = "split"
    All   = "all"


class BarCollector:

    def __init__(self, symbol, feed=Feed.SIP, adjust=Adjust.All, extended_hours=False, path=None):
        if not isinstance(feed, Feed):
            raise ValueError("feed must be a Feed enum member.")
        if not isinstance(adjust, Adjust):
            raise ValueError("adjust must be an Adjust enum member.")

        self.symbol = symbol.upper().strip()
        self.path   = Path(path) if path else Path(f"{self.symbol}-1min.csv")

        self.data   = pd.DataFrame()
        self.client = None

        self.feed           = feed
        self.adjust         = adjust
        self.extended_hours = extended_hours

    def connect(self):
        key, secret = BarCollector._credentials()
        self.client = StockHistoricalDataClient(key, secret)
        return True

    def load(self):
        if not self.path.exists():
            self.data = pd.DataFrame(columns=COLUMNS)
            return True
        try:
            self.data = pd.read_csv(self.path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            raise ValueError(f"Cannot parse {self.path}: {e}")
        for col in COLUMNS:
            if col not in self.data.columns:
                self.data[col] = float("nan")
        self.data = self.data[COLUMNS]
        return True

    def run(self, start_dt=None):
        if self.client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        end    = BarCollector._now()
        start  = self._start(start_dt)
        resume = self._last_timestamp() is not None

        if start >= end:
            print(f"[{self.symbol}]  Already up to date.")
            return True

        print(
            f"[{self.symbol}]  {'Resuming' if resume else 'Starting'}  "
            f"{start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} ET  "
            f"(adjust={self.adjust.name})"
        )

        new = self._fetch(start, end)

        if new.empty:
            print(f"[{self.symbol}]  No bars returned.")
            return True

        self.data = BarCollector._merge(self.data, new)
        BarCollector._write(self.data, self.path)

        print(f"[{self.symbol}]  +{len(new)} bars  |  {len(self.data)} total  |  → {self.path}")
        return True

    def run_live(self):
        if self.client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        print(f"[{self.symbol}]  Live mode — polling every {INTERVAL}s.  Ctrl-C to stop.\n")

        while True:
            end   = BarCollector._now()
            start = self._start(None)
            ts    = end.strftime("%H:%M:%S")

            if start < end:
                new = self._fetch(start, end)
                if not new.empty:
                    self.data = BarCollector._merge(self.data, new)
                    BarCollector._write(self.data, self.path)
                    print(
                        f"  [{ts} ET]  +{len(new)} bars  |  "
                        f"{len(self.data)} total  |  last: {self.data['Datetime'].iloc[-1]}"
                    )
                else:
                    print(f"  [{ts} ET]  No new bars.")
            else:
                print(f"  [{ts} ET]  Up to date.")

            time.sleep(INTERVAL)

    def _last_timestamp(self):
        if self.data.empty:
            return None
        valid = self.data["Datetime"].dropna()
        if valid.empty:
            return None
        return datetime.strptime(valid.iloc[-1], FMT).replace(tzinfo=ET)

    def _start(self, start_dt):
        ts = self._last_timestamp()
        if ts is not None:
            return ts + timedelta(minutes=1)
        if start_dt is not None:
            return start_dt
        base = BarCollector._now() - timedelta(days=LOOKBACK)
        return base.replace(hour=OPEN[0], minute=OPEN[1], second=0, microsecond=0)

    def _fetch(self, start, end):
        if self.client is None:
            raise RuntimeError("Client not connected.")

        alpaca_feed   = DataFeed.SIP if self.feed == Feed.SIP else DataFeed.IEX
        alpaca_adjust = Adjustment(self.adjust.value)

        request = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            adjustment=alpaca_adjust,
            feed=alpaca_feed,
        )

        try:
            bars = self.client.get_stock_bars(request)
        except Exception as e:
            if self.feed != Feed.SIP or "422" not in str(e):
                raise
            # SIP requires paid subscription; retry on the free feed
            print("  WARNING: SIP subscription required. Retrying with IEX.")
            request = StockBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                adjustment=alpaca_adjust,
                feed=DataFeed.IEX,
            )
            bars = self.client.get_stock_bars(request)

        data = bars.df
        if data.empty:
            return pd.DataFrame(columns=COLUMNS)

        data = data.reset_index()
        data = self._shape_(data)
        data = self._validate_(data)
        data = self._round_(data)

        if not self.extended_hours:
            data = BarCollector._filter_session(data)

        return data[COLUMNS].sort_values("Datetime").reset_index(drop=True)

    def _shape_(self, data):
        data.rename(columns={
            "timestamp": "Datetime",
            "open":      "Open",
            "high":      "High",
            "low":       "Low",
            "close":     "Close",
            "volume":    "Volume",
        }, inplace=True)
        data["Datetime"] = data["Datetime"].dt.tz_convert(ET).dt.strftime(FMT)
        # Adj Close intentionally not assigned here — set in _validate_ based on adjust mode
        return data

    def _validate_(self, data):
        n_orig   = len(data)
        repaired = 0
        dropped  = 0

        present  = data.columns.tolist()
        expected = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
        absent   = [f for f in expected if f not in present]

        if absent:
            for f in absent:
                data[f] = float("nan")
            print(f"  [{self.symbol}]  Fields absent from API response: {absent}")

        # High recoverable as lower bound from open-close range
        hi_null = data["High"].isna()
        if hi_null.any():
            data.loc[hi_null, "High"] = np.maximum(
                data.loc[hi_null, "Open"], data.loc[hi_null, "Close"]
            )
            repaired += int(hi_null.sum())

        # Low recoverable as upper bound from open-close range
        lo_null = data["Low"].isna()
        if lo_null.any():
            data.loc[lo_null, "Low"] = np.minimum(
                data.loc[lo_null, "Open"], data.loc[lo_null, "Close"]
            )
            repaired += int(lo_null.sum())

        # Volume defaults to 0: bar existed but no trades were recorded
        vl_null = data["Volume"].isna()
        if vl_null.any():
            data.loc[vl_null, "Volume"] = 0
            repaired += int(vl_null.sum())

        # Open and Close have no intra-bar recovery path
        core_null = data[["Open", "Close"]].isna().any(axis=1)
        if core_null.any():
            dropped += int(core_null.sum())
            data     = data[~core_null].reset_index(drop=True)

        # High must be >= max(open, close); repair upward if violated
        oc_max    = np.maximum(data["Open"], data["Close"])
        hi_breach = data["High"] < oc_max
        if hi_breach.any():
            data.loc[hi_breach, "High"] = oc_max[hi_breach]
            repaired += int(hi_breach.sum())

        # Low must be <= min(open, close); repair downward if violated
        oc_min    = np.minimum(data["Open"], data["Close"])
        lo_breach = data["Low"] > oc_min
        if lo_breach.any():
            data.loc[lo_breach, "Low"] = oc_min[lo_breach]
            repaired += int(lo_breach.sum())

        # Negative volume is physically impossible
        neg_vol = data["Volume"] < 0
        if neg_vol.any():
            data.loc[neg_vol, "Volume"] = 0
            repaired += int(neg_vol.sum())

        # Adj Close: Raw prices have no intraday adjustment factor available from Alpaca.
        # Split and All modes return prices that already carry the requested correction,
        # so Close == Adj Close in those cases.
        if self.adjust == Adjust.Raw:
            data["Adj Close"] = float("nan")
        else:
            data["Adj Close"] = data["Close"]

        if repaired or dropped:
            print(
                f"  [{self.symbol}]  Validation:  "
                f"{repaired} field(s) repaired  |  {dropped} bar(s) dropped  "
                f"({n_orig} → {len(data)} bars)"
            )

        return data

    def _round_(self, data):
        for col in ("Open", "High", "Low", "Close", "Adj Close"):
            data[col] = data[col].round(PRICE_DP)
        data["Volume"] = data["Volume"].astype(int)
        return data

    @staticmethod
    def _credentials():
        key    = os.environ.get("APCA_API_KEY_ID",    "")
        secret = os.environ.get("APCA_API_SECRET_KEY", "")
        if not key or not secret:
            raise Exception(
                "Alpaca credentials not found in environment.\n"
                "  export APCA_API_KEY_ID=your_key\n"
                "  export APCA_API_SECRET_KEY=your_secret"
            )
        return key, secret

    @staticmethod
    def _now():
        return datetime.now(tz=ET)

    @staticmethod
    def _write(df, path):
        df.to_csv(path, index=False, float_format=f"%.{PRICE_DP}f")

    @staticmethod
    def _merge(existing, new):
        combined = (
            pd.concat([existing, new], ignore_index=True)
            .drop_duplicates(subset=["Datetime"])
            .sort_values("Datetime")
            .reset_index(drop=True)
        )
        for col in ("Open", "High", "Low", "Close", "Adj Close"):
            combined[col] = combined[col].astype(float).round(PRICE_DP)
        combined["Volume"] = combined["Volume"].fillna(0).astype(int)
        return combined

    @staticmethod
    def _filter_session(data):
        times   = pd.to_datetime(data["Datetime"]).dt.time
        open_t  = datetime(1900, 1, 1, OPEN[0],  OPEN[1]).time()
        close_t = datetime(1900, 1, 1, CLOSE[0], CLOSE[1]).time()
        return data[(times >= open_t) & (times < close_t)].reset_index(drop=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog="catfish_collector.py",
        description="Catfish — Alpaca 1-minute OHLCV bar collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "adjustment modes:\n"
            "  raw    Unadjusted prices. Adj Close written as NaN (undefined for intraday).\n"
            "  split  Split-adjusted prices. Adj Close = Close.\n"
            "  all    Split + dividend adjusted. Adj Close = Close. Matches Yahoo Finance.\n\n"
            "environment variables:\n"
            "  APCA_API_KEY_ID       Alpaca API key (required)\n"
            "  APCA_API_SECRET_KEY   Alpaca secret key (required)\n\n"
            "examples:\n"
            "  python catfish_collector.py QQQ\n"
            "  python catfish_collector.py QQQ --start 2024-01-02\n"
            "  python catfish_collector.py QQQ --adjust split\n"
            "  python catfish_collector.py QQQ --file QQQ-custom.csv\n"
            "  python catfish_collector.py QQQ --live\n"
            "  python catfish_collector.py QQQ --feed iex"
        ),
    )
    parser.add_argument("symbol",                                      help="Ticker symbol, e.g. QQQ")
    parser.add_argument("--file",           "-f", default=None,        help="Output CSV path (default: <SYMBOL>-1min.csv)")
    parser.add_argument("--start",          "-s", default=None,        help="Start date YYYY-MM-DD (ignored when resuming existing file)")
    parser.add_argument("--live",           "-l", action="store_true", help="Poll Alpaca every ~60 s and append new bars continuously")
    parser.add_argument("--feed",                 default="sip",       choices=["sip", "iex"],        help="Data feed: 'sip' (default, paid) or 'iex' (free)")
    parser.add_argument("--adjust",               default="all",       choices=["raw", "split", "all"], help="Price adjustment mode (default: all)")
    parser.add_argument("--extended-hours",       action="store_true", help="Include pre/after-market bars (default: regular session only)")
    args = parser.parse_args()

    start_dt = None
    if args.start:
        try:
            start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(
                hour=OPEN[0], minute=OPEN[1], second=0, microsecond=0, tzinfo=ET
            )
        except ValueError:
            raise Exception("--start must be YYYY-MM-DD, e.g. 2024-01-02")

    _adjust_map = {"raw": Adjust.Raw, "split": Adjust.Split, "all": Adjust.All}

    Collector = BarCollector(
        args.symbol,
        feed=Feed.SIP if args.feed == "sip" else Feed.IEX,
        adjust=_adjust_map[args.adjust],
        extended_hours=args.extended_hours,
        path=args.file,
    )
    Collector.connect()
    Collector.load()

    try:
        if args.live:
            Collector.run_live()
        else:
            _ = Collector.run(start_dt=start_dt)
            if _ is False:
                raise Exception("Fetch failed")
    except KeyboardInterrupt:
        print("\nInterrupted — data on disk is intact.")
