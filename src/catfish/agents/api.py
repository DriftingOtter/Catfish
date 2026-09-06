import json
import threading
import tomllib
from pathlib import Path
from typing import final

import uvicorn
from fastapi import FastAPI, HTTPException

from catfish.agents.run_moot import DEFAULT_CONFIG, MootRunner
from catfish.paths import PROJECT_ROOT

API_HOST: final = "127.0.0.1"
API_PORT: final = 8080


class MootAPI:

    def __init__(self, config_path=DEFAULT_CONFIG, host=API_HOST, port=API_PORT):
        self.default_config = MootRunner.resolve_config(config_path)
        self.host           = host
        self.port           = int(port)

        self.last_by_moot = {}
        self._locks       = {}
        self._locks_guard = threading.Lock()

        self.app = FastAPI(title="Catfish Moot API", version="0.1.0")
        self._wire()

    @staticmethod
    def from_config(config_path=DEFAULT_CONFIG):
        path = MootRunner.resolve_config(config_path)

        host = API_HOST
        port = API_PORT

        if path.is_file():
            with open(path, "rb") as f:
                cfg = tomllib.load(f)
            api  = cfg.get("api", {})
            host = api.get("host", host)
            port = api.get("port", port)

        return MootAPI(config_path=path, host=host, port=port)

    def _wire(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/v1/moot/run", self.run_moot, methods=["POST"])
        self.app.add_api_route("/v1/moot/last", self.last_moot, methods=["GET"])
        self.app.add_api_route("/v1/trades", self.trades, methods=["GET"])
        self.app.add_api_route("/v1/memory", self.memory, methods=["GET"])

    def _resolve(self, config):
        if config is None or config == "":
            return self.default_config
        return MootRunner.resolve_config(config)

    def _lock_for(self, path):
        key = str(path.resolve())
        with self._locks_guard:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def health(self):
        return {
            "status":          "ok",
            "default_config":  str(self.default_config),
            "moots_cached":    list(self.last_by_moot.keys()),
        }

    def run_moot(self, config=None):
        path = self._resolve(config)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Config not found: {path}")

        lock = self._lock_for(path)
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail=f"Moot already running: {path}")

        try:
            runner = MootRunner(path)
            runner.run()
            payload = runner.payload()
            self.last_by_moot[runner.moot_id] = payload
            return payload
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            lock.release()

    def last_moot(self, moot_id=None, config=None):
        if moot_id is not None and moot_id != "":
            payload = self.last_by_moot.get(moot_id)
            if payload is None:
                raise HTTPException(status_code=404, detail=f"No cached result for moot_id={moot_id}")
            return payload

        if config is not None and config != "":
            path   = self._resolve(config)
            runner = MootRunner(path)
            payload = self.last_by_moot.get(runner.moot_id)
            if payload is None:
                raise HTTPException(status_code=404, detail=f"No cached result for config={path}")
            return payload

        if not self.last_by_moot:
            raise HTTPException(status_code=404, detail="No moot has been run yet")

        # most recent insert order (py3.7+ dict)
        key = next(reversed(self.last_by_moot))
        return self.last_by_moot[key]

    def trades(self, moot_id=None, config=None):
        try:
            payload = self.last_moot(moot_id=moot_id, config=config)
            return {"moot_id": payload.get("moot_id"), "trades": payload.get("trades", [])}
        except HTTPException:
            pass

        path = self._resolve(config)
        paper = self._paper_log_path(path)
        if paper is None or not paper.is_file():
            return {"trades": []}

        trades = []
        with open(paper, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                trade  = record.get("trade")
                if trade is not None:
                    trades.append(trade)

        return {"trades": trades}

    def memory(self, config=None):
        path = self._resolve(config)
        mem  = self._memory_path(path)
        if mem is None or not mem.is_file():
            return {"memory": {}}

        with open(mem, "r", encoding="utf-8") as f:
            return {"memory": json.load(f)}

    def _moot_cfg(self, config_path):
        path = Path(config_path)
        if not path.is_file():
            return {}
        with open(path, "rb") as f:
            return tomllib.load(f).get("moot", {})

    def _memory_path(self, config_path):
        moot = self._moot_cfg(config_path)
        raw  = moot.get("memory_path")
        if raw is None:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def _paper_log_path(self, config_path):
        moot = self._moot_cfg(config_path)
        raw  = moot.get("paper_log_path")
        if raw is None:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def serve(self):
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Catfish moot HTTP API")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    args = parser.parse_args()

    api = MootAPI.from_config(args.config)
    if args.host is not None:
        api.host = args.host
    if args.port is not None:
        api.port = int(args.port)
    api.serve()
