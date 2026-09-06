import json
from datetime import datetime, timezone
from pathlib import Path
from typing import final

import yfinance as yf

from catfish.paths import PROJECT_ROOT

MIN_TRADES:     final = 5
ORDER_NOTIONAL: final = 10_000.0


class FinanceAgent:

    def __init__(
        self,
        memory_path,
        paper_log_path,
        confidence_threshold,
        min_trades_for_history,
        order_notional,
    ):
        self.memory_path            = Path(memory_path)
        self.paper_log_path         = Path(paper_log_path)
        self.confidence_threshold   = float(confidence_threshold)
        self.min_trades_for_history = int(min_trades_for_history)
        self.order_notional         = float(order_notional)
        self.memory                 = {}

        self._load_memory()

    @staticmethod
    def from_config(moot_cfg):
        memory = moot_cfg.get("memory_path", "runtime/broker_memory.json")
        paper  = moot_cfg.get("paper_log_path", "runtime/paper_orders.jsonl")

        memory_path = Path(memory)
        paper_path  = Path(paper)
        if not memory_path.is_absolute():
            memory_path = PROJECT_ROOT / memory_path
        if not paper_path.is_absolute():
            paper_path = PROJECT_ROOT / paper_path

        return FinanceAgent(
            memory_path=memory_path,
            paper_log_path=paper_path,
            confidence_threshold=moot_cfg.get("confidence_threshold", 0.55),
            min_trades_for_history=moot_cfg.get("min_trades_for_history", MIN_TRADES),
            order_notional=moot_cfg.get("order_notional", ORDER_NOTIONAL),
        )

    def _load_memory(self):
        if not self.memory_path.is_file():
            self.memory = {}
            return

        with open(self.memory_path, "r", encoding="utf-8") as f:
            self.memory = json.load(f)

    def _save_memory(self):
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2)

    def _ensure_broker(self, broker_id, weight_seed):
        if broker_id in self.memory:
            return

        self.memory[broker_id] = {
            "trades":      0,
            "wins":        0,
            "pnl":         0.0,
            "weight_seed": float(weight_seed),
        }

    def trust(self, broker_id, weight_seed=1.0):
        self._ensure_broker(broker_id, weight_seed)
        rec = self.memory[broker_id]

        seed   = float(rec.get("weight_seed", weight_seed))
        trades = int(rec.get("trades", 0))
        wins   = int(rec.get("wins", 0))

        if trades < self.min_trades_for_history:
            win_rate = 0.5
        else:
            win_rate = wins / max(trades, 1)

        return seed * (0.5 + 0.5 * win_rate)

    @staticmethod
    def _live_price(symbol):
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period="5d", interval="1d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])

    def _build_trade(self, symbol, side, score, proposals, agreeing):
        prices = [p["last_price"] for p in agreeing if p.get("last_price") is not None]
        live   = self._live_price(symbol)
        if live is not None:
            price = live
        elif prices:
            price = float(sum(prices) / len(prices))
        else:
            raise ValueError(f"No execution price available for {symbol}")

        weight_sum = 0.0
        er_acc     = 0.0
        for prop in agreeing:
            w = float(prop.get("score", prop.get("confidence", 0.0)))
            weight_sum += w
            er_acc     += w * float(prop.get("expected_return", 0.0))

        if weight_sum > 0.0:
            expected_return = er_acc / weight_sum
        else:
            expected_return = 0.0

        shares          = int(self.order_notional // price)
        notional        = shares * price
        expected_value  = notional * expected_return

        if shares < 1:
            raise ValueError(
                f"order_notional {self.order_notional} too small for {symbol} @ {price}"
            )

        return {
            "symbol":           symbol,
            "side":             side,
            "execution_price":  price,
            "shares":           shares,
            "notional":         notional,
            "expected_return":  expected_return,
            "expected_value":   expected_value,
            "score":            score,
            "brokers":          [p["broker_id"] for p in agreeing],
        }

    def adjudicate(self, proposals):
        if not proposals:
            raise ValueError("No proposals to adjudicate.")

        by_symbol = {}
        for prop in proposals:
            symbol = prop["symbol"]
            by_symbol.setdefault(symbol, []).append(prop)

        decisions = []
        for symbol, group in by_symbol.items():
            decisions.append(self._adjudicate_symbol(symbol, group))

        return decisions

    def _adjudicate_symbol(self, symbol, proposals):
        buy_score  = 0.0
        sell_score = 0.0
        hold_score = 0.0
        weighted   = []

        for prop in proposals:
            broker_id = prop["broker_id"]
            seed      = prop.get("weight_seed", 1.0)
            t         = self.trust(broker_id, seed)
            conf      = float(prop["confidence"])
            score     = t * conf
            side      = prop["side"]

            weighted.append({
                "broker_id":       broker_id,
                "side":            side,
                "confidence":      conf,
                "trust":           t,
                "score":           score,
                "weight_seed":     float(seed),
                "last_price":      prop.get("last_price"),
                "expected_return": prop.get("expected_return", 0.0),
                "detail":          prop.get("detail", {}),
            })

            if side == "buy":
                buy_score += score
            elif side == "sell":
                sell_score += score
            else:
                hold_score += score

        if buy_score >= sell_score and buy_score >= hold_score:
            side  = "buy"
            score = buy_score
        elif sell_score >= buy_score and sell_score >= hold_score:
            side  = "sell"
            score = sell_score
        else:
            side  = "hold"
            score = hold_score

        # portfolio value x flexibility checks deferred; threshold is the guard
        if side == "hold" or score < self.confidence_threshold:
            action = "hold"
            if side != "hold" and score < self.confidence_threshold:
                action = "reject"
            return {
                "symbol":    symbol,
                "side":      side,
                "score":     score,
                "action":    action,
                "threshold": self.confidence_threshold,
                "proposals": weighted,
                "trade":     None,
            }

        agreeing = [p for p in weighted if p["side"] == side]
        trade    = self._build_trade(symbol, side, score, proposals, agreeing)

        decision = {
            "symbol":    symbol,
            "side":      side,
            "score":     score,
            "action":    "execute",
            "threshold": self.confidence_threshold,
            "proposals": weighted,
            "trade":     trade,
        }
        self.execute(decision)
        return decision

    def execute(self, decision):
        # paper blotter only — no live broker API
        self.paper_log_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "symbol": decision["symbol"],
            "side":   decision["side"],
            "score":  decision["score"],
            "action": decision["action"],
            "trade":  decision.get("trade"),
        }

        with open(self.paper_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        for prop in decision.get("proposals", []):
            self._ensure_broker(prop["broker_id"], prop.get("weight_seed", 1.0))

        self._save_memory()
        return True
