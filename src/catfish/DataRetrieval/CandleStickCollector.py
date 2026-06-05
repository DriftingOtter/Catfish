import asyncio
import enum
from datetime import datetime, timedelta
from pathlib import Path
from typing import final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from catfish.paths import PROJECT_ROOT

ET: final = ZoneInfo("America/New_York")
FMT: final = "%Y-%m-%d %H:%M:%S"
DATE_FMT: final = "%Y-%m-%d"
PRICE_DP: final = 6
COLUMNS: final = ["Datetime", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
DAILY_COLUMNS: final = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
CORE: final = ["Open", "High", "Low", "Close", "Volume"]


# session windows in ET as (open_hm, close_hm, lunch_start, lunch_end); overnight sessions have close_hm < open_hm
class Region(enum.Enum):
    NewYork  = ((9, 30),  (16,  0),  None,      None)
    London   = ((3,  0),  (11, 30),  None,      None)
    Tokyo    = ((20, 0),  (2,  30),  (22, 30),  (23, 30))
    HongKong = ((21, 30), (4,   0),  (0,   0),  (1,   0))


class CandleStickCollector:

    def __init__(self, symbol, region=Region.NewYork):
        if not isinstance(region, Region):
            raise ValueError("region must be a Region enum member.")

        today = datetime.now(tz=ET).strftime("%Y-%m-%d")

        self.data            = pd.DataFrame()
        self.historical_data = pd.DataFrame()

        self.symbol = symbol.upper().strip()
        self.region = region
        self.live_path = Path(f"{region.name}-{today}.csv")

        self._bar      = None
        self._bar_ts   = None
        self._prev_vol = 0

    def run(self):
        open_hm, close_hm = self.region.value[0], self.region.value[1]
        end   = CandleStickCollector._now()
        start = end.replace(hour=open_hm[0], minute=open_hm[1], second=0, microsecond=0)

        # overnight session opened on the previous calendar day
        if open_hm > close_hm:
            start -= timedelta(days=1)

        if start >= end:
            print(f"[{self.symbol}]  Session has not started yet.")
            return True

        print(
            f"[{self.symbol}]  Fetching  "
            f"{start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} ET  "
            f"({self.region.name})"
        )

        new = self._fetch(start, end)

        if new.empty:
            print(f"[{self.symbol}]  No bars returned.")
            return True

        self.data = new
        CandleStickCollector._write(self.data, self.live_path)

        print(f"[{self.symbol}]  {len(self.data)} bars → {self.live_path}")
        return True

    def fetch_historical(self, start=None, end=None, path=None):
        if start is not None and end is not None:
            start_ts = pd.Timestamp(start)
            end_ts   = pd.Timestamp(end)
            if start_ts > end_ts:
                raise Exception(f"start ({start}) must not exceed end ({end}).")
            range_label = (
                f"{start_ts.strftime(DATE_FMT)} → {end_ts.strftime(DATE_FMT)}"
            )
        elif start is None and end is None:
            start_ts    = None
            end_ts      = None
            range_label = "all available history"
        else:
            raise Exception("start and end must both be provided, or both omitted.")

        if path is None:
            path = PROJECT_ROOT / "datasets" / self.symbol / f"{self.symbol}.csv"
        else:
            path = Path(path)

        print(f"[{self.symbol}]  Fetching daily  {range_label}")

        new = self._fetch_daily(start_ts, end_ts)

        if new.empty:
            print(f"[{self.symbol}]  No daily bars returned.")
            return True

        self.historical_data = new
        path.parent.mkdir(parents=True, exist_ok=True)
        CandleStickCollector._write(self.historical_data, path)

        print(f"[{self.symbol}]  {len(self.historical_data)} days → {path}")
        return True

    async def run_live(self):
        print(f"[{self.symbol}]  WebSocket streaming active.  Ctrl-C to stop.\n")

        ws = yf.AsyncWebSocket()
        ws.subscribe([self.symbol])
        await ws.start(callback=self._on_tick)

    def _fetch(self, start, end):
        raw = yf.Ticker(self.symbol).history(
            start=start,
            end=end,
            interval="1m",
            auto_adjust=True,
            prepost=False,
        )

        if raw.empty:
            return pd.DataFrame(columns=COLUMNS)

        data = self._shape_(raw)
        data = self._validate_(data)
        data = self._round_(data)
        data = self._filter_session(data)

        return data[COLUMNS].sort_values("Datetime").reset_index(drop=True)

    def _fetch_daily(self, start, end):
        ticker = yf.Ticker(self.symbol)

        if start is None and end is None:
            raw = ticker.history(
                period="max",
                interval="1d",
                auto_adjust=True,
                prepost=False,
            )
        else:
            # yfinance end is exclusive — advance one day so end date is included
            raw = ticker.history(
                start=start,
                end=end + pd.Timedelta(days=1),
                interval="1d",
                auto_adjust=True,
                prepost=False,
            )

        if raw.empty:
            return pd.DataFrame(columns=DAILY_COLUMNS)

        data = self._shape_daily_(raw)
        data = self._validate_(data)
        data = self._round_(data)

        return data[DAILY_COLUMNS].sort_values("Date").reset_index(drop=True)

    def _filter_session(self, data):
        if data.empty:
            return data

        open_hm, close_hm, lunch_s, lunch_e = self.region.value

        times   = pd.to_datetime(data["Datetime"]).dt.time
        open_t  = datetime(1900, 1, 1, *open_hm).time()
        close_t = datetime(1900, 1, 1, *close_hm).time()

        # overnight session crosses midnight in ET
        if open_hm > close_hm:
            in_session = (times >= open_t) | (times < close_t)
        else:
            in_session = (times >= open_t) & (times < close_t)

        if lunch_s is not None:
            l_open  = datetime(1900, 1, 1, *lunch_s).time()
            l_close = datetime(1900, 1, 1, *lunch_e).time()
            in_session = in_session & ~((times >= l_open) & (times < l_close))

        return data[in_session].reset_index(drop=True)

    async def _on_tick(self, tick):
        price = getattr(tick, "price", None)
        if not price or price <= 0:
            return

        raw_t = getattr(tick, "time", None)
        if raw_t is None:
            return

        # yfinance may encode timestamps in milliseconds
        if raw_t > 1e12:
            raw_t /= 1000

        ts     = datetime.fromtimestamp(raw_t, tz=ET)
        bar_ts = ts.replace(second=0, microsecond=0)

        day_vol = getattr(tick, "dayVolume", 0) or 0

        # new calendar date in ET resets the cumulative volume baseline
        if self._bar_ts is not None and bar_ts.date() != self._bar_ts.date():
            self._prev_vol = day_vol

        vol_delta      = max(0, day_vol - self._prev_vol)
        self._prev_vol = day_vol

        if self._bar_ts is not None and bar_ts != self._bar_ts:
            row = pd.DataFrame([{
                "Datetime":  self._bar_ts.strftime(FMT),
                "Open":      self._bar["o"],
                "High":      self._bar["h"],
                "Low":       self._bar["l"],
                "Close":     self._bar["c"],
                "Adj Close": self._bar["c"],
                "Volume":    self._bar["v"],
            }])
            row = self._validate_(row)
            row = self._round_(row)
            row = self._filter_session(row)

            if not row.empty:
                self.data = pd.concat([self.data, row], ignore_index=True)
                CandleStickCollector._write(self.data, self.live_path)
                b = self._bar
                print(
                    f"  [{ts.strftime('%H:%M:%S')} ET]  "
                    f"{self._bar_ts.strftime('%H:%M')}  "
                    f"O:{b['o']:.4f}  H:{b['h']:.4f}  "
                    f"L:{b['l']:.4f}  C:{b['c']:.4f}  V:{b['v']}"
                )

        if bar_ts != self._bar_ts:
            self._bar    = {"o": price, "h": price, "l": price, "c": price, "v": vol_delta}
            self._bar_ts = bar_ts
        else:
            if self._bar is None:
                self._bar = {"o": price, "h": price, "l": price, "c": price, "v": 0}
            self._bar["h"]  = max(self._bar["h"], price)
            self._bar["l"]  = min(self._bar["l"], price)
            self._bar["c"]  = price
            self._bar["v"] += vol_delta

    def _shape_(self, data):
        # yfinance >= 0.2 wraps single-ticker columns in a MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index.name = "Datetime"
        data = data.reset_index()
        ts = pd.to_datetime(data["Datetime"])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(ET)
        data["Datetime"] = ts.dt.tz_convert(ET).dt.strftime(FMT)
        return data

    def _shape_daily_(self, data):
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index.name = "Date"
        data = data.reset_index()
        ts = pd.to_datetime(data["Date"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(ET)
        data["Date"] = ts.dt.strftime(DATE_FMT)
        return data

    def _validate_(self, data):
        absent = [f for f in CORE if f not in data.columns]
        if absent:
            raise ValueError(f"Fields absent from API response: {absent}")

        oc_max = np.maximum(data["Open"], data["Close"])
        oc_min = np.minimum(data["Open"], data["Close"])

        data["High"]   = np.where(data["High"].isna(), oc_max, data["High"])
        data["Low"]    = np.where(data["Low"].isna(), oc_min, data["Low"])
        data["Volume"] = np.where(data["Volume"].isna(), 0, data["Volume"])

        core_null = data[["Open", "Close"]].isna().any(axis=1)
        if core_null.any():
            raise ValueError(
                f"{int(core_null.sum())} bar(s) with unrecoverable Open/Close "
                f"in {len(data)}-bar response."
            )

        data["High"]   = np.where(data["High"] < oc_max, oc_max, data["High"])
        data["Low"]    = np.where(data["Low"] > oc_min, oc_min, data["Low"])
        data["Volume"] = np.where(data["Volume"] < 0, 0, data["Volume"])

        # auto_adjust=True folds split/dividend correction into OHLC
        data["Adj Close"] = data["Close"]

        return data

    def _round_(self, data):
        for col in ("Open", "High", "Low", "Close", "Adj Close"):
            data[col] = data[col].round(PRICE_DP)
        data["Volume"] = data["Volume"].astype(int)
        return data

    @staticmethod
    def _now():
        return datetime.now(tz=ET)

    @staticmethod
    def _write(df, path):
        df.to_csv(path, index=False, float_format=f"%.{PRICE_DP}f")


if __name__ == '__main__':

    QQQCollector = CandleStickCollector("QQQ", region=Region.NewYork)

    _ = QQQCollector.fetch_historical()
    if _ is False:
        raise Exception("Historical fetch failed")

    # _ = QQQCollector.fetch_historical("2020-01-01", "2025-12-31")

    _ = QQQCollector.run()
    if _ is False:
        raise Exception("Session fetch failed")

    try:
        asyncio.run(QQQCollector.run_live())
    except KeyboardInterrupt:
        print("\nInterrupted — data on disk is intact.")
