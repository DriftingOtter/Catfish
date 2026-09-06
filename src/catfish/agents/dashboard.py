from typing import final

from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Static
from textual.worker import WorkerState

from catfish.agents.run_moot import DEFAULT_CONFIG, MootRunner

CSS: final = """
Screen {
    layout: vertical;
}

#status {
    height: 3;
    padding: 0 1;
    color: $text-muted;
}

#panels {
    height: 1fr;
}

#brokers-panel, #trades-panel {
    width: 1fr;
    height: 1fr;
    border: solid $accent;
    margin: 0 1 1 1;
}

#brokers-panel {
    border-title-color: $accent;
}

#trades-panel {
    border-title-color: $success;
}

DataTable {
    height: 1fr;
}
"""


class MootDashboard(App):

    TITLE = "Catfish Moot"
    CSS = CSS
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "rerun", "Rerun"),
    ]

    def __init__(self, config_path=DEFAULT_CONFIG):
        super().__init__()
        self.config_path = config_path
        self.runner      = None
        self.proposals   = []
        self.decisions   = []

    def compose(self):
        yield Header(show_clock=True)
        yield Static("Starting moot…", id="status")
        with Horizontal(id="panels"):
            with Vertical(id="brokers-panel"):
                yield DataTable(id="brokers")
            with Vertical(id="trades-panel"):
                yield DataTable(id="trades")
        yield Footer()

    def on_mount(self):
        brokers = self.query_one("#brokers", DataTable)
        trades  = self.query_one("#trades", DataTable)

        self.query_one("#brokers-panel").border_title = "Broker Activity"
        self.query_one("#trades-panel").border_title  = "Human Trades To Execute"

        brokers.add_columns(
            "Broker",
            "Symbol",
            "Side",
            "Confidence",
            "E[R]",
            "Last Px",
        )
        trades.add_columns(
            "Symbol",
            "Side",
            "Exec Px",
            "Shares",
            "Notional",
            "E[R]",
            "E[Value]",
            "Score",
            "Brokers",
        )

        self._start_moot()

    def _start_moot(self):
        status = self.query_one("#status", Static)
        status.update(f"Running moot · {self.config_path}")
        self.query_one("#brokers", DataTable).clear()
        self.query_one("#trades", DataTable).clear()
        self.run_worker(self._run_moot, exclusive=True, thread=True)

    def _on_proposal_thread(self, proposal):
        self.call_from_thread(self._append_proposal, proposal)

    def _run_moot(self):
        runner = MootRunner(self.config_path)
        proposals, decisions = runner.run(on_proposal=self._on_proposal_thread)
        self.call_from_thread(self._finish, runner, proposals, decisions)

    def _append_proposal(self, proposal):
        table = self.query_one("#brokers", DataTable)
        price = proposal.get("last_price")
        er    = proposal.get("expected_return", 0.0)
        table.add_row(
            proposal["broker_id"],
            proposal["symbol"],
            proposal["side"].upper(),
            f"{proposal['confidence']:.1%}",
            f"{er:+.2%}",
            f"{price:.2f}" if price is not None else "—",
        )

    def _finish(self, runner, proposals, decisions):
        self.runner    = runner
        self.proposals = proposals
        self.decisions = decisions

        trades = self.query_one("#trades", DataTable)
        trades.clear()

        human = runner.human_trades()
        for trade in human:
            trades.add_row(
                trade["symbol"],
                trade["side"].upper(),
                f"{trade['execution_price']:.2f}",
                str(trade["shares"]),
                f"${trade['notional']:,.2f}",
                f"{trade['expected_return']:+.2%}",
                f"${trade['expected_value']:+,.2f}",
                f"{trade['score']:.3f}",
                ", ".join(trade["brokers"]),
            )

        status = self.query_one("#status", Static)
        moot   = runner.moot_id or "?"
        if human:
            status.update(
                f"{moot} complete · {len(proposals)} proposal(s) · "
                f"{len(human)} trade(s) for manual execution · q quit · r rerun"
            )
        else:
            status.update(
                f"{moot} complete · {len(proposals)} proposal(s) · "
                f"no trades cleared threshold · q quit · r rerun"
            )

    def on_worker_state_changed(self, event):
        worker = event.worker
        if worker.state != WorkerState.ERROR:
            return
        status = self.query_one("#status", Static)
        status.update(f"Moot failed · {worker.error}")

    def action_rerun(self):
        self._start_moot()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Catfish moot TUI dashboard")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    MootDashboard(config_path=args.config).run()
