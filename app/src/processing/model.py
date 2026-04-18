import os

from .preprocessing import PreprocessorData



class ModelData:
    def __init__(self, label=None, probabilities=None):
        self.Label = label
        self.Probabilities = probabilities  # list[float], one per class, sums to 1.0


class Model:
    def __init__(self, model_path: str | None = None):
        pass
    
    def _ensure_loaded(self):
        pass

    def run(self, preprocessed: PreprocessorData) -> ModelData:
        modelData =  ModelData()
        return modelData
