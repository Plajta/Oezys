from dataclasses import dataclass
from PIL import Image

from .processing import Preprocessor, Model, Metrics

@dataclass
class PipelineOut:
    preprocessed: object
    prediction: object
    metrics: object
    


class Pipeline:
    def __init__(
        self, metrics, model, preprocessor
    ):
        self._metrics = metrics
        self._model = model
        self._preprocessor = preprocessor
        pass

    def run(self, image: Image.Image):
        # preprocess
        # predict
        # data/metrics

        pass
       