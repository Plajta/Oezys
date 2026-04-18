import numpy as np

from ..utils.afm import prepare_image


class PreprocessorData:
    def __init__(self, image: np.ndarray | None = None):
        self.image = image


class Preprocessor:
    def __init__(self):
        pass

    def run(self, path: str) -> PreprocessorData:
        data = PreprocessorData()
        data.image = prepare_image(path)
        return data
