import os

from .preprocessing import PreprocessorData
from .gambling import HybridAFMClassifier, load_model, predict

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "test.pt")


class ModelData:
    def __init__(self, label=None, probabilities=None):
        self.Label = label
        self.Probabilities = probabilities  # list[float], one per class, sums to 1.0


class Model:
    def __init__(self, model_path: str | None = None):
        self._model_path = model_path or DEFAULT_MODEL_PATH
        self._model: HybridAFMClassifier | None = None  # loaded on first use

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"No trained model found at '{self._model_path}'. "
                "Run training first: python src/processing/gambling.py --data ../data/raw"
            )
        self._model = load_model(self._model_path)

    def run(self, preprocessed: PreprocessorData) -> ModelData:
        self._ensure_loaded()
        label, probs = predict(self._model, preprocessed.image)
        return ModelData(label=label, probabilities=probs)
