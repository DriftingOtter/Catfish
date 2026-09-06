from catfish.models.sentiment.sec_finbert import MarketSentimentModel


class Broker:

    def __init__(self, broker_id, symbol, years, weight_seed):
        self.broker_id   = broker_id
        self.symbol      = symbol
        self.years       = years
        self.weight_seed = weight_seed

    @staticmethod
    def from_config(cfg, moot_cfg):
        return Broker(
            broker_id=cfg["id"],
            symbol=cfg.get("symbol", moot_cfg.get("symbols", ["UNKNOWN"])[0]).upper(),
            years=int(cfg.get("years", 2)),
            weight_seed=float(cfg.get("weight_seed", 1.0)),
        )

    def run(self):
        model = MarketSentimentModel(self.symbol)
        model.load_filings(years=self.years)

        if not model.filings:
            raise ValueError(f"[{self.broker_id}] No SEC filings for {self.symbol}")

        latest = model.filings[0]
        model.analyse_filing(latest)

        tone, dominant = model.get_dominant_tone()
        scores         = model.get_sentiment()
        conf           = float(dominant) / 100.0

        if conf > 1.0:
            conf = 1.0
        if conf < 0.0:
            conf = 0.0

        if tone == "positive":
            side = "buy"
            er   = conf * 0.01
        elif tone == "negative":
            side = "sell"
            er   = conf * 0.01
        else:
            side = "hold"
            er   = 0.0

        return {
            "broker_id":       self.broker_id,
            "symbol":          self.symbol,
            "side":            side,
            "confidence":      conf,
            "weight_seed":     self.weight_seed,
            "last_price":      None,
            "expected_return": er,
            "detail": {
                "tone":     tone,
                "dominant": dominant,
                "scores":   scores,
                "filing":   f"{latest['doc_type']} ({latest['date']})",
            },
        }
