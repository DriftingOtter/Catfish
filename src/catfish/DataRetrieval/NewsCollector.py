import enum
import time
from datetime import datetime
from pathlib import Path
from typing import final

import requests
import yfinance as yf
from bs4 import BeautifulSoup

from catfish.paths import PROJECT_ROOT

DELAY:   final = 1.0
TIMEOUT: final = 30
HEADERS: final = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}


class FilingType(enum.Enum):
    Annual    = "10-K"
    Quarterly = "10-Q"
    Current   = "8-K"


class NewsCollector:

    def __init__(self, symbol, path=None):
        if not symbol or not symbol.strip():
            raise ValueError("symbol must be a non-empty ticker string.")

        self.filings = []

        self.symbol = symbol.upper().strip()
        self.path   = Path(path) if path is not None else PROJECT_ROOT / "datasets" / self.symbol / "SEC"
        self.ticker = yf.Ticker(self.symbol)

    def fetch(self, start, end):
        if start > end:
            raise Exception(f"start year ({start}) must not exceed end year ({end}).")

        raw = self.ticker.get_sec_filings()
        if not raw:
            raise ValueError(f"No SEC filings returned for {self.symbol}.")

        types = {t.value for t in FilingType}
        self.filings = [
            f for f in raw
            if f.get("type") in types and start <= f["date"].year <= end
        ]
        if not self.filings:
            raise ValueError(
                f"No {'/'.join(sorted(types))} filings found for {self.symbol} "
                f"between {start} and {end}."
            )

        print(
            f"[{self.symbol}]  {len(self.filings)} target filing(s) found  "
            f"({start}–{end})."
        )
        return True

    def fetch_latest(self):
        year = datetime.now().year
        self.fetch(year, year)

        if all(self._file_path(f).exists() for f in self.filings):
            print(f"[{self.symbol}]  {year} filings already on disk → {self.path}")
            return True

        return self.download()

    def download(self):
        if not self.filings:
            raise RuntimeError("No filings loaded. Call fetch() first.")

        self.path.mkdir(parents=True, exist_ok=True)
        saved = 0

        for filing in self.filings:
            file_path = self._file_path(filing)

            if file_path.exists():
                print(f"  [{self.symbol}]  Skip   {file_path.name}")
                continue

            url  = self._url(filing)
            html = NewsCollector._fetch_html(url)
            text = NewsCollector._clean(html)
            file_path.write_text(text, encoding="utf-8")

            saved += 1
            print(f"  [{self.symbol}]  Saved  {file_path.name}  ({len(text):,} chars)")
            time.sleep(DELAY)

        print(f"[{self.symbol}]  {saved} new filing(s) saved → {self.path}")
        return True

    def _file_path(self, filing):
        date = filing["date"].strftime("%Y-%m-%d")
        return self.path / f"{self.symbol}-{filing['type']}-{date}.txt"

    def _url(self, filing):
        url = filing.get("exhibits", {}).get(filing["type"])
        if url is None:
            raise ValueError(
                f"Primary exhibit URL missing for {filing['type']} "
                f"filing on {filing['date']} ({self.symbol})."
            )
        return url

    @staticmethod
    def _fetch_html(url):
        resp = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
        return resp.text

    @staticmethod
    def _clean(html):
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()
        text  = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        return "\n".join(line for line in lines if line)


if __name__ == '__main__':

    NVDACollector = NewsCollector("NVDA", path=str(PROJECT_ROOT / "datasets" / "NVDA" / "SEC"))

    _ = NVDACollector.fetch(2021, 2026)
    if _ is False:
        raise Exception("Fetch failed")

    _ = NVDACollector.download()
    if _ is False:
        raise Exception("Download failed")

    # _ = NVDACollector.fetch_latest()
    # if _ is False:
    #     raise Exception("Fetch latest failed")
