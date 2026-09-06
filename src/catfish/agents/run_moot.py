import importlib
import json
import tomllib
from pathlib import Path
from typing import final

from catfish.agents.finance_agent.finance_agent import FinanceAgent
from catfish.paths import PROJECT_ROOT

DEFAULT_CONFIG: final = "configs/moot.toml"


class MootRunner:

    def __init__(self, config_path=DEFAULT_CONFIG):
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path

        self.config    = {}
        self.moot      = {}
        self.moot_id   = None
        self.proposals = []
        self.decisions = []

        self._load_config()

    def _load_config(self):
        if not self.config_path.is_file():
            raise Exception(f"Config not found: {self.config_path}")

        with open(self.config_path, "rb") as f:
            self.config = tomllib.load(f)

        self.moot    = self.config.get("moot", {})
        self.moot_id = self.moot.get("id", self.config_path.stem)

    def run_brokers(self, on_proposal=None):
        self.proposals = []

        for cfg in self.config.get("brokers", []):
            if not cfg.get("enabled", True):
                continue

            module_name = cfg["module"]
            mod = importlib.import_module(module_name)
            if not hasattr(mod, "Broker"):
                raise Exception(f"Module {module_name} has no Broker class")

            # each broker builds and trains its own model instance — never shared
            broker   = mod.Broker.from_config(cfg, self.moot)
            proposal = broker.run()
            self.proposals.append(proposal)

            if on_proposal is not None:
                on_proposal(proposal)

        return self.proposals

    def run(self, on_proposal=None):
        self.run_brokers(on_proposal=on_proposal)
        if not self.proposals:
            raise Exception("No enabled brokers produced proposals.")

        agent          = FinanceAgent.from_config(self.moot)
        self.decisions = agent.adjudicate(self.proposals)
        return self.proposals, self.decisions

    def human_trades(self):
        trades = []
        for decision in self.decisions:
            if decision.get("action") != "execute":
                continue
            trade = decision.get("trade")
            if trade is None:
                continue
            trades.append(trade)
        return trades

    def payload(self):
        return self._jsonable({
            "moot_id":   self.moot_id,
            "config":    str(self.config_path),
            "symbols":   self.moot.get("symbols", []),
            "proposals": self.proposals,
            "decisions": self.decisions,
            "trades":    self.human_trades(),
        })

    @staticmethod
    def resolve_config(config_path):
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @staticmethod
    def _jsonable(obj):
        if isinstance(obj, dict):
            return {str(k): MootRunner._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [MootRunner._jsonable(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (str, bool)) or obj is None:
            return obj
        if isinstance(obj, int) and not isinstance(obj, bool):
            return int(obj)
        if isinstance(obj, float):
            return float(obj)
        if hasattr(obj, "item"):
            return MootRunner._jsonable(obj.item())
        return str(obj)


class MootCLI:

    def __init__(self, config_path=DEFAULT_CONFIG, mode="tui"):
        self.config_path = config_path
        self.mode        = mode

    def run(self):
        if self.mode == "tui":
            from catfish.agents.dashboard import MootDashboard
            MootDashboard(config_path=self.config_path).run()
            return True

        if self.mode == "headless":
            runner = MootRunner(self.config_path)
            runner.run()
            print(json.dumps(runner.payload(), indent=2))
            return True

        if self.mode == "api":
            from catfish.agents.api import MootAPI
            MootAPI.from_config(self.config_path).serve()
            return True

        raise Exception(f"Unknown mode: {self.mode}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a Catfish multi-broker moot")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=["tui", "headless", "api"],
        default="tui",
        help="tui (default), headless JSON stdout, or api HTTP server",
    )
    args = parser.parse_args()
    MootCLI(config_path=args.config, mode=args.mode).run()
