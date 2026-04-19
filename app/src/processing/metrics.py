import sys
from datetime import datetime
from pathlib import Path

from .model import ModelData
from .preprocessing import PreprocessorData


_MODEL_NAMES = ["RESNET + RF + FFT", "Neural Network(RESNET)", "Random Forest"]


class MetricData:
    def __init__(self, datetime=None, filename=None, probabilities=None):
        self.Datetime = datetime
        self.FileName = filename
        self.FilePath = None
        self.Probabilities = probabilities
        self.labels = ["Diabetes", "Primary Open-Angle Glaucoma", "Multiple Sclerosis", "Dry Eye Disease", "Healthy"]
        self.AllProbabilities: list[list[float]] = []
        self.ModelNames: list[str] = []


class Metrics:
    def __init__(self):
        pass

    def run(self, preprocessed: PreprocessorData, predictions: list[ModelData], path: str = "") -> MetricData:
        metrics = MetricData()
        now = datetime.now()
        metrics.Datetime = now.strftime("%Y-%m-%d %H:%M:%S")
        ms = int(now.timestamp() * 1000)
        metrics.FileName = f"{Path(path).stem}_{ms}{Path(path).suffix}.bmp"
        metrics.FilePath = path
        if not predictions[0].Probabilities:
            print("[Metrics] No model output — probabilities unavailable", file=sys.stderr)
            metrics.Probabilities = None
            return metrics
        metrics.Probabilities = [round(p * 100, 2) for p in predictions[0].Probabilities]
        all_probs = []
        for pred in predictions:
            if pred.Probabilities:
                all_probs.append([round(p * 100, 2) for p in pred.Probabilities])
        metrics.AllProbabilities = all_probs
        metrics.ModelNames = _MODEL_NAMES[:len(all_probs)]
        return metrics