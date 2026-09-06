import importlib
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
        self.proposals = []
        self.decisions = []

        self._load_config()

    def _load_config(self):
        if not self.config_path.is_file():
            raise Exception(f"Config not found: {self.config_path}")

        with open(self.config_path, "rb") as f:
            self.config = tomllib.load(f)

        self.moot = self.config.get("moot", {})

    def run_brokers(self, on_proposal=None):
        self.proposals = []

        for cfg in self.config.get("brokers", []):
            if not cfg.get("enabled", True):
                continue

            module_name = cfg["module"]
            mod = importlib.import_module(module_name)
            if not hasattr(mod, "Broker"):
                raise Exception(f"Module {module_name} has no Broker class")

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


if __name__ == "__main__":
    import argparse

    from catfish.agents.dashboard import MootDashboard

    parser = argparse.ArgumentParser(description="Run a Catfish multi-broker moot")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to moot TOML (default: configs/moot.toml)",
    )
    args = parser.parse_args()
    MootDashboard(config_path=args.config).run()
