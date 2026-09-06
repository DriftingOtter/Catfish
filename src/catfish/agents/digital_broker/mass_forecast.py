from pathlib import Path

from catfish.core.time_index import TimeIndex
from catfish.models.forecast.ben_daniel_duke import BenDanielDukeModel
from catfish.paths import PROJECT_ROOT


class Broker:

    def __init__(
        self,
        broker_id,
        symbol,
        data_path,
        time_index,
        training_period,
        forecast_steps,
        forecast_dt,
        forecast_snaps,
        weight_seed,
    ):
        self.broker_id       = broker_id
        self.symbol          = symbol
        self.data_path       = data_path
        self.time_index      = time_index
        self.training_period = training_period
        self.forecast_steps  = forecast_steps
        self.forecast_dt     = forecast_dt
        self.forecast_snaps  = forecast_snaps
        self.weight_seed     = weight_seed

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
            training_period=cfg.get("training_period", 504),
            forecast_steps=int(cfg.get("forecast_steps", 150)),
            forecast_dt=float(cfg.get("forecast_dt", 0.015)),
            forecast_snaps=int(cfg.get("forecast_snaps", 50)),
            weight_seed=float(cfg.get("weight_seed", 1.0)),
        )

    def run(self):
        model = BenDanielDukeModel(time_index=self.time_index)
        model.load_data(self.data_path)
        model.set_training_period(self.training_period)
        model.calculate_features()

        x0 = float(model.x_hist[-1])
        model.init_psi(x0)
        model.set_forecast_params(
            steps=self.forecast_steps,
            dt=self.forecast_dt,
            n_snaps=self.forecast_snaps,
        )
        model.forecast()

        fc     = model.get_forecast()
        xE     = float(fc["xE"][-1])
        sigma  = float(fc["sigma"][-1])
        regime = model.get_regime()
        price  = float(model.training_data["Close"].iloc[-1])
        conf   = 1.0 / (1.0 + max(sigma, 0.0))

        if conf > 1.0:
            conf = 1.0
        if conf < 0.0:
            conf = 0.0

        if xE > 0.0:
            side = "buy"
        elif xE < 0.0:
            side = "sell"
        else:
            side = "hold"

        # xE is displacement from VWAP — normalize by price level
        er = xE / max(price, 1e-9)

        return {
            "broker_id":       self.broker_id,
            "symbol":          self.symbol,
            "side":            side,
            "confidence":      conf,
            "weight_seed":     self.weight_seed,
            "last_price":      price,
            "expected_return": er,
            "detail": {
                "terminal_xE":    xE,
                "terminal_sigma": sigma,
                "regime":         regime,
            },
        }
