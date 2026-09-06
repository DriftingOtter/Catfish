from pathlib import Path

from catfish.core.time_index import TimeIndex
from catfish.models.signal.hermitian_mlp import HermitianMLPModel
from catfish.paths import PROJECT_ROOT


class Broker:

    def __init__(self, broker_id, symbol, data_path, time_index, tau, window, weight_seed):
        self.broker_id   = broker_id
        self.symbol      = symbol
        self.data_path   = data_path
        self.time_index  = time_index
        self.tau         = tau
        self.window      = window
        self.weight_seed = weight_seed

    @staticmethod
    def from_config(cfg, moot_cfg):
        time_name  = cfg.get("time_index", moot_cfg.get("time_index", "Date"))
        time_index = TimeIndex[time_name]

        path = Path(cfg["data"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return Broker(
            broker_id=cfg["id"],
            symbol=cfg.get("symbol", moot_cfg.get("symbols", ["UNKNOWN"])[0]).upper(),
            data_path=str(path),
            time_index=time_index,
            tau=float(cfg.get("tau", 0.52)),
            window=int(cfg.get("window", 504)),
            weight_seed=float(cfg.get("weight_seed", 1.0)),
        )

    def run(self):
        model = HermitianMLPModel(time_index=self.time_index)
        model.load_data(self.data_path)
        model.set_window(self.window)
        model.set_tau(self.tau)
        model.calculate_features()
        model.init_model()

        _ = model.train_model()
        if _ is False:
            raise Exception(f"[{self.broker_id}] MLP training hit max_iter")

        model.generate_signals()

        signal = model.get_latest_signal()
        proba  = model.get_latest_probability()
        price  = float(model.data["Close"].iloc[-1])
        conf   = abs(float(proba) - 0.5) * 2.0

        if signal == 1:
            side = "buy"
            er   = float(proba) - 0.5
        else:
            side = "sell"
            er   = 0.5 - float(proba)

        return {
            "broker_id":       self.broker_id,
            "symbol":          self.symbol,
            "side":            side,
            "confidence":      conf,
            "weight_seed":     self.weight_seed,
            "last_price":      price,
            "expected_return": er,
            "detail": {
                "signal": signal,
                "proba":  proba,
                "tau":    self.tau,
            },
        }
