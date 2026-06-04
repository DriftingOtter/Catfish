import os
from pathlib import Path
from typing import final

import matplotlib.pyplot as plt
from transformers import pipeline

MODEL_PATH: final = (
    Path(__file__).resolve().parent.parent.parent / "ExternalModels" / "finbert"
)
MAX_LENGTH: final = 512


class MarketSentimentModel:
    def __init__(self):
        if not MODEL_PATH.is_dir():
            raise FileNotFoundError(
                f"FinBERT model directory not found: {MODEL_PATH}"
            )

        os.environ["HF_HUB_OFFLINE"]            = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

        model_dir  = str(MODEL_PATH)
        self.model = pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=model_dir,
            max_length=MAX_LENGTH,
            truncation=True,
        )

    def extract_sentiment(self, scores):
        if scores is None:
            raise ValueError("No scores provided.")

        return [
            (
                state["label"],
                round(state["score"] * 100, 2)
            )
            for state in scores
        ]

    def _chunk_text(self, text: str) -> list[str]:
        tokenizer = self.model.tokenizer
        chunk_size = MAX_LENGTH - 2  # room for [CLS] and [SEP]

        saved_max_length = tokenizer.model_max_length
        tokenizer.model_max_length = max(saved_max_length, len(text))
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        finally:
            tokenizer.model_max_length = saved_max_length

        if len(token_ids) <= chunk_size:
            return [text]

        return [
            tokenizer.decode(token_ids[i : i + chunk_size])
            for i in range(0, len(token_ids), chunk_size)
        ]

    def analyse(self, text):
        if len(text) < 1:
            raise ValueError("Empty text. Nothing to analyse.")

        chunks = self._chunk_text(text)
        states_prob = self.model(chunks, top_k=None)

        if len(chunks) == 1:
            if states_prob and isinstance(states_prob[0], dict):
                return self.extract_sentiment(states_prob)
            return self.extract_sentiment(states_prob[0])

        label_totals: dict[str, float] = {}
        for chunk_scores in states_prob:
            for state in chunk_scores:
                label_totals[state["label"]] = (
                    label_totals.get(state["label"], 0.0) + state["score"]
                )

        averaged = [
            {"label": label, "score": total / len(chunks)}
            for label, total in label_totals.items()
        ]
        return self.extract_sentiment(averaged)

    def plot_sentiment(self, sentiment, ax=None):
        standalone = ax is None

        if standalone:
            fig, ax = plt.subplots(figsize=(6, 6))

        labels = [label.capitalize() for label, _ in sentiment]
        scores = [score for _, score in sentiment]

        ax.pie(
            scores,
            labels=labels,
            autopct="%1.1f%%"
        )

        ax.set_title("FinBERT Sentiment Distribution")

        if standalone:
            plt.tight_layout()
            return fig

        return None


if __name__ == "__main__":

    text: str = open("/Users/daksh/Documents/Catfish/src/MarketPressure/8k.txt").read()
    print(text)

    SentimentModel = MarketSentimentModel()

    sentiment = SentimentModel.analyse(text=text)

    print(sentiment)

    fig = SentimentModel.plot_sentiment(sentiment)
    plt.show()