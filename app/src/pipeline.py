from dataclasses import dataclass

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

    def run(self, path: str):
        # preprocess
        # predict
        # data/metrics

        preprocessorData = self._preprocessor.run(path)
        modelData = self._model.run(preprocessorData)
        metricsData = self._metrics.run(preprocessorData, modelData)

        return PipelineOut(
            preprocessed=preprocessorData,
            prediction=modelData,
            metrics=metricsData
        )