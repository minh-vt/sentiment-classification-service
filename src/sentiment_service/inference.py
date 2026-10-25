"""Real checkpoint loading and inference. Reads local files only."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment_service.artifacts import verify_bundle


class OverlongInput(ValueError):
    """A sentence needs more positions than the model can serve."""

    def __init__(self, index, positions, limit):
        super().__init__(f"sentence {index} needs {positions} positions, limit is {limit}")
        self.index = index
        self.positions = positions
        self.limit = limit


class SentimentModel:
    """Sequence-classification model served on CPU with its own tokenizer."""

    def __init__(self, model_dir, manifest):
        directory = str(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True
        )
        self.model.eval()
        self.labels = self.model.config.id2label
        self.positions_limit = int(self.model.config.max_position_embeddings)
        self.version = manifest["version"]

    def predict(self, texts, max_positions):
        """Encode once, refuse overlong sentences, then infer.

        Raises OverlongInput instead of truncating, so no prediction is ever
        returned for text the caller did not send.
        """
        batch = self.tokenizer(texts, padding=True, return_tensors="pt")
        for index, positions in enumerate(batch["attention_mask"].sum(dim=-1).tolist()):
            if positions > max_positions:
                raise OverlongInput(index, positions, max_positions)
        with torch.inference_mode():
            logits = self.model(**batch).logits
        probabilities = logits.softmax(dim=-1)
        scores, indices = probabilities.max(dim=-1)
        return [
            {"label": self.labels[int(index)], "score": float(score)}
            for score, index in zip(scores, indices, strict=True)
        ]


def load_model(settings):
    """Verify the bundle, load it, and refuse a limit the model cannot serve.

    Raises BundleError on an artifact mismatch and ValueError on an unusable
    configuration; both leave the process alive and permanently unready.
    """
    model = SentimentModel(settings.model_dir, verify_bundle(settings.model_dir))
    if settings.max_positions > model.positions_limit:
        raise ValueError(
            f"MAX_POSITIONS={settings.max_positions} exceeds the model limit "
            f"of {model.positions_limit} positions"
        )
    return model
