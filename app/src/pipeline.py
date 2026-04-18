from dataclasses import dataclass
from PIL import Image

from .processing import Preprocessor, Model, Metrics, MetricData, PreprocessorData, ModelData

@dataclass
class PipelineOut:
    preprocessed: PreprocessorData
    prediction: ModelData
    metrics: MetricData



class Pipeline:
    def __init__(
        self, metrics: Metrics, model: Model, preprocessor: Preprocessor
    ):
        self._metrics = metrics
        self._model = model
        self._preprocessor = preprocessor
        pass

    def run(self, image: Image.Image):
        # preprocess
        # predict
        # data/metrics

        preprocessorData = self._preprocessor.run(image)
        modelData = self._model.run(preprocessorData)
        metricsData = self._metrics.run(image.filename, preprocessorData, modelData)

        return PipelineOut(
            preprocessed=preprocessorData,
            prediction=modelData,
            metrics=metricsData
        )