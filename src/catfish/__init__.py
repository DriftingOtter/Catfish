from catfish.paths import PROJECT_ROOT

__all__ = [
    "PROJECT_ROOT",
    "BarCollector",
    "MarketPressureModel",
    "ModelType",
    "HermitianMLPModel",
    "MarketSentimentModel",
]


def __getattr__(name):
    if name == "BarCollector":
        from catfish.DataRetrieval.CandleStickCollector import CandleStickCollector
        return CandleStickCollector
    if name == "MarketPressureModel":
        from catfish.AlphaModels.HMM.VariableWindowHMM import MarketPressureModel
        return MarketPressureModel
    if name == "ModelType":
        from catfish.AlphaModels.HMM.VariableWindowHMM import ModelType
        return ModelType
    if name == "HermitianMLPModel":
        from catfish.AlphaModels.HermitianMLP.HermitianMLP import HermitianMLPModel
        return HermitianMLPModel
    if name == "MarketSentimentModel":
        from catfish.MarketSentiment.MarketSentimentModel import MarketSentimentModel
        return MarketSentimentModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
