from catfish.paths import PROJECT_ROOT

__all__ = [
    "PROJECT_ROOT",
    "BarCollector",
    "MarketPressureModel",
    "ModelType",
    "TimeIndex",
    "HermitianMLPModel",
    "MarketSentimentModel",
]


def __getattr__(name):
    if name == "BarCollector":
        from catfish.data.ohlcv import CandleStickCollector
        return CandleStickCollector
    if name == "MarketPressureModel":
        from catfish.models.regime.hmm import MarketPressureModel
        return MarketPressureModel
    if name == "ModelType":
        from catfish.models.regime.hmm import ModelType
        return ModelType
    if name == "TimeIndex":
        from catfish.core.time_index import TimeIndex
        return TimeIndex
    if name == "HermitianMLPModel":
        from catfish.models.signal.hermitian_mlp import HermitianMLPModel
        return HermitianMLPModel
    if name == "MarketSentimentModel":
        from catfish.models.sentiment.sec_finbert import MarketSentimentModel
        return MarketSentimentModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
