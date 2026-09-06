from pathlib import Path

from catfish.core.time_index import TimeIndex
from catfish.models.regime.hmm import MarketPressureModel, ModelType
from catfish.paths import PROJECT_ROOT


class Broker:

    def __init__(self, broker_id, symbol, data_path, time_index, model_type, training_period, weight_seed):
        self.broker_id       = broker_id
        self.symbol          = symbol
        self.data_path       = data_path
        self.time_index      = time_index
        self.model_type      = model_type
        self.training_period = training_period
        self.weight_seed     = weight_seed

    @staticmethod
    def from_config(cfg, moot_cfg):
        time_name  = cfg.get("time_index", moot_cfg.get("time_index", "Date"))
        time_index = TimeIndex[time_name]

        model_name = cfg.get("model_type", "GaussianEmission")
        model_type = ModelType[model_name]

        path = Path(cfg["data"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return Broker(
            broker_id=cfg["id"],
            symbol=cfg.get("symbol", moot_cfg.get("symbols", ["UNKNOWN"])[0]).upper(),
            data_path=str(path),
            time_index=time_index,
            model_type=model_type,
            training_period=cfg.get("training_period", 504),
            weight_seed=float(cfg.get("weight_seed", 1.0)),
        )

    def run(self):
        model = MarketPressureModel(model_type=self.model_type, time_index=self.time_index)
        model.load_data(self.data_path)
        model.set_training_period(self.training_period)
        model.calculate_features()
        model.init_model()

        _ = model.train_model()
        if _ is False:
            raise Exception(f"[{self.broker_id}] HMM convergence failed")

        state   = model.get_current_state()
        probs   = model.get_current_probabilities()
        summary = model.get_state_summary()

        mean_r = float(summary.loc[state, ("r", "mean")])
        conf   = float(probs[state])
        price  = float(model.training_data["Close"].iloc[-1])

        if mean_r > 0.0:
            side = "buy"
        elif mean_r < 0.0:
            side = "sell"
        else:
            side = "hold"

        return {
            "broker_id":       self.broker_id,
            "symbol":          self.symbol,
            "side":            side,
            "confidence":      conf,
            "weight_seed":     self.weight_seed,
            "last_price":      price,
            "expected_return": mean_r,
            "detail": {
                "state":  state,
                "mean_r": mean_r,
                "probs":  [float(p) for p in probs],
            },
        }
