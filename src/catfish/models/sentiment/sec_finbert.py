import os
import re
from datetime import datetime
from pathlib import Path
from typing import final

from transformers import pipeline

from catfish.core.stdout import print_field, print_heading
from catfish.data.layout import sec_dir
from catfish.data.sec import SECCollector
from catfish.paths import PROJECT_ROOT

MODEL_PATH: final = PROJECT_ROOT / "ExternalModels" / "finbert"
MAX_LENGTH: final = 512
FILING_PATTERN: final = re.compile(
    r"^(.+)-(10-K|10-Q|8-K)-(\d{4})-(\d{2})-(\d{2})\.txt$"
)


class MarketSentimentModel:

    def __init__(self, symbol):
        if not symbol or not symbol.strip():
            raise ValueError("symbol must be a non-empty ticker string.")

        if not MODEL_PATH.is_dir():
            raise Exception(f"FinBERT model directory not found: {MODEL_PATH}")

        self.filings    = []
        self.sentiments = {}
        self.active     = None

        self._years   = None
        self.symbol   = symbol.upper().strip()
        self.sec_path = sec_dir()

        os.environ["HF_HUB_OFFLINE"]           = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        model_dir  = str(MODEL_PATH)
        self.model = pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=model_dir,
            max_length=MAX_LENGTH,
            truncation=True,
        )

    def load_filings(self, years=None):
        if years is not None and years < 1:
            raise Exception("years must be >= 1.")

        if not self.sec_path.is_dir():
            raise Exception(f"SEC directory not found: {self.sec_path}")

        self._years = years
        today       = datetime.now().date()
        cutoff      = None
        if years is not None:
            cutoff = today.replace(year=today.year - years)

        filings = []
        for path in sorted(self.sec_path.glob("*.txt")):
            match = FILING_PATTERN.match(path.name)
            if match is None:
                continue
            if match.group(1) != self.symbol:
                continue

            year  = int(match.group(3))
            month = int(match.group(4))
            day   = int(match.group(5))
            date  = datetime(year, month, day).date()

            if cutoff is not None and date < cutoff:
                continue

            filings.append({
                "doc_type": match.group(2),
                "year":     year,
                "month":    month,
                "day":      day,
                "date":     date,
                "path":     path,
            })

        if not filings:
            if cutoff is not None:
                raise ValueError(
                    f"No SEC filings found for {self.symbol} within the past {years} year(s)."
                )
            raise ValueError(f"No SEC filings found for {self.symbol} in {self.sec_path}.")

        filings.sort(key=lambda f: f["date"], reverse=True)
        self.filings = filings
        return True

    def filing_from_path(self, filing_path):
        path = Path(filing_path).resolve()
        if not path.is_file():
            return None

        match = FILING_PATTERN.match(path.name)
        if match is None or match.group(1) != self.symbol:
            return None

        year  = int(match.group(3))
        month = int(match.group(4))
        day   = int(match.group(5))
        date  = datetime(year, month, day).date()

        return {
            "doc_type": match.group(2),
            "year":     year,
            "month":    month,
            "day":      day,
            "date":     date,
            "path":     path,
        }

    def ensure_filing(self, filing_path):
        resolved = Path(filing_path).resolve()
        for filing in self.filings:
            if filing["path"].resolve() == resolved:
                return filing

        filing = self.filing_from_path(filing_path)
        if filing is None:
            return None

        self.filings.append(filing)
        self.filings.sort(key=lambda f: f["date"], reverse=True)
        return filing

    def analyse_filing(self, filing):
        if isinstance(filing, dict):
            path = Path(filing["path"])
        else:
            path = Path(filing)

        path = path.resolve()
        if not path.is_file():
            raise Exception(f"Filing not found: {path}")

        key = str(path)
        if key in self.sentiments:
            self.active = filing if isinstance(filing, dict) else self._filing_for_path(path)
            return self.sentiments[key]

        text = path.read_text(encoding="utf-8")
        sentiment = self.analyse(text)

        self.sentiments[key] = sentiment
        self.active = filing if isinstance(filing, dict) else self._filing_for_path(path)
        return sentiment

    def analyse_latest(self, years=None):
        if years is not None:
            self._years = years
        if self._years is None:
            raise RuntimeError("years not set. Call load_filings() or pass years to analyse_latest().")

        self.sec_path.mkdir(parents=True, exist_ok=True)

        collector = SECCollector(self.symbol, path=str(self.sec_path))
        _ = collector.fetch_latest()

        self.load_filings(self._years)

        latest = self.filings[0]
        return self.analyse_filing(latest)

    def analyse(self, text):
        if len(text) < 1:
            raise ValueError("Empty text. Nothing to analyse.")

        chunks = self._chunk_text(text)
        states_prob = self.model(chunks, top_k=None)

        if len(chunks) == 1:
            if states_prob and isinstance(states_prob[0], dict):
                return self._extract_sentiment(states_prob)
            return self._extract_sentiment(states_prob[0])

        label_totals = {}
        for chunk_scores in states_prob:
            for state in chunk_scores:
                label_totals[state["label"]] = (
                    label_totals.get(state["label"], 0.0) + state["score"]
                )

        averaged = [
            {"label": label, "score": total / len(chunks)}
            for label, total in label_totals.items()
        ]
        return self._extract_sentiment(averaged)

    def _filing_for_path(self, path):
        key = str(path)
        for filing in self.filings:
            if str(filing["path"]) == key:
                return filing
        return None

    def _extract_sentiment(self, scores):
        if scores is None:
            raise ValueError("No scores provided.")

        return [
            (
                state["label"],
                round(state["score"] * 100, 2)
            )
            for state in scores
        ]

    def _chunk_text(self, text):
        tokenizer = self.model.tokenizer
        chunk_size = MAX_LENGTH - 2  # room for [CLS] and [SEP]

        saved_max_length = tokenizer.model_max_length
        tokenizer.model_max_length = max(saved_max_length, len(text))
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        finally:
            tokenizer.model_max_length = saved_max_length

        if len(token_ids) <= chunk_size:
            return [text]

        return [
            tokenizer.decode(token_ids[i : i + chunk_size])
            for i in range(0, len(token_ids), chunk_size)
        ]

    def get_sentiment(self, filing_path=None):
        if filing_path is None:
            if self.active is None:
                raise RuntimeError("No active filing.")
            path = self.active["path"]
        else:
            path = Path(filing_path)

        key = str(path.resolve())
        scores = self.sentiments.get(key)
        if scores is None:
            raise RuntimeError(f"No sentiment scored for {path.name}.")
        return {label: pct for label, pct in scores}

    def get_dominant_tone(self, filing_path=None):
        scores = self.get_sentiment(filing_path)
        pos = scores.get("positive", 0.0)
        neu = scores.get("neutral", 0.0)
        neg = scores.get("negative", 0.0)

        if neg > pos and neg > neu:
            return "negative", neg
        if pos > neg and pos > neu:
            return "positive", pos
        return "neutral", neu

    def print_results(self):
        if not self.sentiments:
            raise RuntimeError("No filings analysed.")

        if self.active is None:
            raise RuntimeError("No active filing.")

        filing = self.active
        scores = self.get_sentiment()
        tone, dominant = self.get_dominant_tone()

        print_heading("Market Sentiment")
        print_field("Symbol", self.symbol)
        print_field("Filing", f"{filing['doc_type']} ({filing['date']})")
        print_field("Overall tone", tone)
        print_field("Dominant score", f"{dominant:.1f}%")
        print()
        print("  Label probabilities:")
        for label in ("positive", "neutral", "negative"):
            if label in scores:
                print(f"    {label:8s}: {scores[label]:.1f}%")
