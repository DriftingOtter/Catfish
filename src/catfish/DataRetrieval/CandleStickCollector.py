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
FLUSH_INTERVAL: final = timedelta(minutes=10)


# session windows in ET as (open_hm, close_hm, lunch_start, lunch_end); overnight sessions have close_hm < open_hm
class Region(enum.Enum):
    NewYork  = ((9, 30),  (16,  0),  None,      None)
    London   = ((3,  0),  (11, 30),  None,      None)
    Tokyo    = ((20, 0),  (2,  30),  (22, 30),  (23, 30))
    HongKong = ((21, 30), (4,   0),  (0,   0),  (1,   0))


class BarInterval(enum.Enum):
    OneSecond  = 1
    OneMinute  = 60
    FiveMinute = 300
    TenMinute  = 600


class CandleStickCollector:

    def __init__(self, symbols, region=Region.NewYork, path=None, interval=BarInterval.OneMinute):
        if not isinstance(region, Region):
            raise ValueError("region must be a Region enum member.")
        if not isinstance(interval, BarInterval):
            raise ValueError("interval must be a BarInterval enum member.")

        if isinstance(symbols, str):
            symbols = [symbols]
        if not symbols:
            raise ValueError("symbols must be a non-empty ticker string or list.")

        self.symbols = [s.upper().strip() for s in symbols]
        if any(not s for s in self.symbols):
            raise ValueError("symbols must be a non-empty ticker string or list.")

        self.region   = region
        self.interval = interval
        self.paths    = CandleStickCollector._resolve_paths(self.symbols, path)

        self.data            = {s: pd.DataFrame() for s in self.symbols}
        self.historical_data = {s: pd.DataFrame() for s in self.symbols}

        self._bar          = {s: None for s in self.symbols}
        self._bar_ts       = {s: None for s in self.symbols}
        self._prev_vol     = {s: 0 for s in self.symbols}
        self._rows_written = {s: 0 for s in self.symbols}
        self._last_flush   = {s: None for s in self.symbols}
        self._flush_task   = {s: None for s in self.symbols}

    def run(self):
        open_hm, close_hm = self.region.value[0], self.region.value[1]
        end   = CandleStickCollector._now()
        start = end.replace(hour=open_hm[0], minute=open_hm[1], second=0, microsecond=0)

        # overnight session opened on the previous calendar day
        if open_hm > close_hm:
            start -= timedelta(days=1)

        if start >= end:
            print(f"[{', '.join(self.symbols)}]  Session has not started yet.")
            return True

        print(
            f"[{', '.join(self.symbols)}]  Fetching  "
            f"{start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} ET  "
            f"({self.region.name}, {self.interval.name})"
        )

        for symbol in self.symbols:
            new = self._fetch(symbol, start, end)

            if new.empty:
                if self.interval == BarInterval.OneSecond:
                    print(f"[{symbol}]  1s interval — session prefetch skipped; live will aggregate ticks.")
                else:
                    print(f"[{symbol}]  No bars returned.")
                continue

            self.data[symbol] = new
            path = self.paths[symbol]
            path.parent.mkdir(parents=True, exist_ok=True)
            CandleStickCollector._write(self.data[symbol], path)
            self._rows_written[symbol] = len(self.data[symbol])

            print(f"[{symbol}]  {len(self.data[symbol])} bars → {path}")

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

        paths = CandleStickCollector._resolve_paths(self.symbols, path, daily=True)

        print(f"[{', '.join(self.symbols)}]  Fetching daily  {range_label}")

        for symbol in self.symbols:
            new = self._fetch_daily(symbol, start_ts, end_ts)

            if new.empty:
                print(f"[{symbol}]  No daily bars returned.")
                continue

            out_path = paths[symbol]
            self.historical_data[symbol] = new
            out_path.parent.mkdir(parents=True, exist_ok=True)
            CandleStickCollector._write(self.historical_data[symbol], out_path)

            print(f"[{symbol}]  {len(self.historical_data[symbol])} days → {out_path}")

        return True

    def finalize_current_bars(self):
        for symbol in self.symbols:
            if self._bar[symbol] is None or self._bar_ts[symbol] is None:
                continue

            self._append_completed_bar_(symbol)
            self._bar[symbol]    = None
            self._bar_ts[symbol] = None

        return True

    def flush(self):
        self.finalize_current_bars()

        for symbol in self.symbols:
            pending = self.data[symbol].iloc[self._rows_written[symbol]:]
            if pending.empty:
                continue

            path = self.paths[symbol]
            path.parent.mkdir(parents=True, exist_ok=True)
            CandleStickCollector._append(
                pending, path, self._rows_written[symbol] == 0,
            )
            rows = len(pending)
            self._rows_written[symbol] = len(self.data[symbol])
            CandleStickCollector._print_flush(symbol, rows, path)

        return True

    async def run_live(self):
        now = CandleStickCollector._now()
        for symbol in self.symbols:
            self._last_flush[symbol] = now

        print(
            f"[{', '.join(self.symbols)}]  WebSocket streaming active  "
            f"({self.interval.name}).  Ctrl-C to stop.\n"
        )

        await asyncio.gather(*[
            self._run_live_symbol(symbol) for symbol in self.symbols
        ])

    async def _run_live_symbol(self, symbol):
        ws = yf.AsyncWebSocket(verbose=False)
        await ws.subscribe(symbol)
        await ws.listen(message_handler=self._on_tick)

    def _fetch(self, symbol, start, end):
        if self.interval == BarInterval.OneSecond:
            return pd.DataFrame(columns=COLUMNS)

        yf_interval = CandleStickCollector._yfinance_interval(self.interval)
        raw = yf.Ticker(symbol).history(
            start=start,
            end=end,
            interval=yf_interval,
            auto_adjust=True,
            prepost=False,
        )

        if raw.empty:
            return pd.DataFrame(columns=COLUMNS)

        data = self._shape_(raw)
        data = self._validate_(data)
        data = self._round_(data)
        data = self._filter_session(data)

        if self.interval == BarInterval.TenMinute:
            data = self._resample_(data)

        return data[COLUMNS].sort_values("Datetime").reset_index(drop=True)

    def _fetch_daily(self, symbol, start, end):
        ticker = yf.Ticker(symbol)

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
        symbol = CandleStickCollector._tick_val(tick, "id")
        if symbol not in self.symbols:
            return

        price = CandleStickCollector._tick_num(tick, "price")
        if price is None or price <= 0:
            return

        raw_t = CandleStickCollector._tick_num(tick, "time")
        if raw_t is None:
            return

        # yfinance may encode timestamps in milliseconds
        if raw_t > 1e12:
            raw_t /= 1000

        ts     = datetime.fromtimestamp(raw_t, tz=ET)
        bar_ts = CandleStickCollector._floor_ts(ts, self.interval)

        day_vol = CandleStickCollector._tick_num(tick, "day_volume", "dayVolume") or 0

        # new calendar date in ET resets the cumulative volume baseline
        if self._bar_ts[symbol] is not None and bar_ts.date() != self._bar_ts[symbol].date():
            self._prev_vol[symbol] = day_vol

        # day_volume is session-cumulative; stale ticks are ignored, baseline never regresses
        if day_vol >= self._prev_vol[symbol]:
            vol_delta                 = day_vol - self._prev_vol[symbol]
            self._prev_vol[symbol] = day_vol
        else:
            vol_delta = 0

        if self._bar_ts[symbol] is not None and bar_ts != self._bar_ts[symbol]:
            if self._append_completed_bar_(symbol):
                now = CandleStickCollector._now()
                if (now - self._last_flush[symbol]) >= FLUSH_INTERVAL:
                    self._schedule_flush(symbol)

                b = self._bar[symbol]
                bar_label = CandleStickCollector._bar_label(self._bar_ts[symbol], self.interval)
                print(
                    f"  [{symbol}]  [{ts.strftime('%H:%M:%S')} ET]  "
                    f"{bar_label}  "
                    f"O:{b['o']:.4f}  H:{b['h']:.4f}  "
                    f"L:{b['l']:.4f}  C:{b['c']:.4f}  V:{b['v']}"
                )

        if bar_ts != self._bar_ts[symbol]:
            self._bar[symbol]    = {"o": price, "h": price, "l": price, "c": price, "v": vol_delta}
            self._bar_ts[symbol] = bar_ts
        else:
            if self._bar[symbol] is None:
                self._bar[symbol] = {"o": price, "h": price, "l": price, "c": price, "v": 0}
            self._bar[symbol]["h"]  = max(self._bar[symbol]["h"], price)
            self._bar[symbol]["l"]  = min(self._bar[symbol]["l"], price)
            self._bar[symbol]["c"]  = price
            self._bar[symbol]["v"] += vol_delta

    def _append_completed_bar_(self, symbol):
        row = pd.DataFrame([{
            "Datetime":  self._bar_ts[symbol].strftime(FMT),
            "Open":      self._bar[symbol]["o"],
            "High":      self._bar[symbol]["h"],
            "Low":       self._bar[symbol]["l"],
            "Close":     self._bar[symbol]["c"],
            "Adj Close": self._bar[symbol]["c"],
            "Volume":    self._bar[symbol]["v"],
        }])
        row = self._validate_(row)
        row = self._round_(row)
        row = self._filter_session(row)

        if row.empty:
            return False

        self.data[symbol] = pd.concat([self.data[symbol], row], ignore_index=True)
        return True

    def _resample_(self, data):
        if data.empty:
            return data

        ts = pd.to_datetime(data["Datetime"])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(ET)

        out = (
            data.set_index(ts)
            .resample("10min")
            .agg({
                "Open":   "first",
                "High":   "max",
                "Low":    "min",
                "Close":  "last",
                "Volume": "sum",
            })
            .dropna(subset=["Open", "Close"])
        )

        out["Adj Close"] = out["Close"]
        out = out.reset_index(names="Datetime")
        out["Datetime"] = out["Datetime"].dt.tz_convert(ET).dt.strftime(FMT)

        return out

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

    def _schedule_flush(self, symbol):
        task = self._flush_task[symbol]
        if task is not None and not task.done():
            return
        self._flush_task[symbol] = asyncio.create_task(self._flush(symbol))

    async def _flush(self, symbol):
        pending = self.data[symbol].iloc[self._rows_written[symbol]:].copy()
        if pending.empty:
            return

        header = self._rows_written[symbol] == 0
        rows   = len(pending)
        path   = self.paths[symbol]

        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(CandleStickCollector._append, pending, path, header)

        self._rows_written[symbol] += rows
        self._last_flush[symbol]     = CandleStickCollector._now()
        CandleStickCollector._print_flush(symbol, rows, path)

    @staticmethod
    def _yfinance_interval(interval):
        if interval in (BarInterval.OneMinute, BarInterval.TenMinute):
            return "1m"
        if interval == BarInterval.FiveMinute:
            return "5m"
        raise ValueError(f"interval {interval.name} has no yfinance history mapping.")

    @staticmethod
    def _floor_ts(ts, interval):
        if interval == BarInterval.OneSecond:
            return ts.replace(microsecond=0)
        if interval == BarInterval.OneMinute:
            return ts.replace(second=0, microsecond=0)

        span    = interval.value // 60
        minute  = (ts.minute // span) * span
        return ts.replace(minute=minute, second=0, microsecond=0)

    @staticmethod
    def _bar_label(bar_ts, interval):
        if interval == BarInterval.OneSecond:
            return bar_ts.strftime("%H:%M:%S")
        return bar_ts.strftime("%H:%M")

    @staticmethod
    def _resolve_paths(symbols, path, daily=False):
        if path is None:
            today = datetime.now(tz=ET).strftime(DATE_FMT)
            if daily:
                return {
                    s: PROJECT_ROOT / "datasets" / s / f"{s}.csv"
                    for s in symbols
                }
            return {
                s: PROJECT_ROOT / "datasets" / s / f"{s}-{today}.csv"
                for s in symbols
            }

        if isinstance(path, dict):
            resolved = {}
            for symbol in symbols:
                if symbol not in path:
                    raise Exception(f"path missing entry for {symbol}.")
                resolved[symbol] = CandleStickCollector._validate_path(path[symbol])
            return resolved

        if len(symbols) > 1:
            raise Exception(
                "path must be a dict mapping symbol to .csv when tracking multiple tickers."
            )

        return {symbols[0]: CandleStickCollector._validate_path(path)}

    @staticmethod
    def _validate_path(path):
        path = Path(path)
        if path.suffix.lower() != ".csv":
            raise Exception("Invalid file format. Must be .csv")
        return path

    @staticmethod
    def _tick_val(tick, *keys, default=None):
        for key in keys:
            if isinstance(tick, dict):
                val = tick.get(key)
            else:
                val = getattr(tick, key, None)
            if val is not None:
                return val
        return default

    @staticmethod
    def _tick_num(tick, *keys, default=None):
        val = CandleStickCollector._tick_val(tick, *keys, default=default)
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _now():
        return datetime.now(tz=ET)

    @staticmethod
    def _print_flush(symbol, rows, path):
        print(f"[{symbol}]  saved {rows} bar(s) → {path}")

    @staticmethod
    def _write(df, path):
        df.to_csv(path, index=False, float_format=f"%.{PRICE_DP}f")

    @staticmethod
    def _append(df, path, header):
        mode = "w" if header else "a"
        df.to_csv(
            path,
            mode=mode,
            header=header,
            index=False,
            float_format=f"%.{PRICE_DP}f",
        )


if __name__ == '__main__':

    INTERVAL = BarInterval.OneMinute
    # INTERVAL = BarInterval.OneSecond
    # INTERVAL = BarInterval.FiveMinute
    # INTERVAL = BarInterval.TenMinute

    today = datetime.now(tz=ET).strftime(DATE_FMT)

    SessionCollector = CandleStickCollector(
        ["QQQ", "MNQ", "SPY", "SPCX"],
        region=Region.NewYork,
        interval=INTERVAL,
        path={
            "QQQ": str(PROJECT_ROOT / "datasets" / "QQQ" / f"QQQ-{today}.csv"),
            "MNQ": str(PROJECT_ROOT / "datasets" / "MNQ" / f"MNQ-{today}.csv"),
            "SPY": str(PROJECT_ROOT / "datasets" / "SPY" / f"SPY-{today}.csv"),
            "SPCX": str(PROJECT_ROOT / "datasets" / "SPCX" / f"SPCX-{today}.csv"),
        },
    )

    # _ = SessionCollector.fetch_historical()
    # if _ is False:
    #     raise Exception("Historical fetch failed")

    # _ = SessionCollector.fetch_historical("2020-01-01", "2025-12-31")

    _ = SessionCollector.run()
    if _ is False:
        raise Exception("Session fetch failed")

    try:
        asyncio.run(SessionCollector.run_live())
    except KeyboardInterrupt:
        print("\nInterrupted — flushing remaining data.")
        _ = SessionCollector.flush()
