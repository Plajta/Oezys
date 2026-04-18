import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

from .preprocessing import PreprocessorData

NUM_CLASSES = 5
_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "classifier_best.pth")

_VAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_model() -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES),
    )
    return m


class ModelData:
    def __init__(self, label=None, probabilities=None):
        self.Label = label
        self.Probabilities = probabilities  # list[float], one per class, sums to 1.0


class Model:
    def __init__(self, model_path: str | None = None):
        self._path = model_path or _WEIGHTS_PATH
        self._model: nn.Module | None = None
        self._device = torch.device("cpu")

    def _ensure_loaded(self):
        if self._model is not None:
            return
        net = _build_model()
        state = torch.load(self._path, map_location=self._device, weights_only=True)
        net.load_state_dict(state)
        net.eval()
        self._model = net

    def run(self, preprocessed: PreprocessorData) -> ModelData:
        self._ensure_loaded()

        arr = preprocessed.image
        # Normalize 2D float array to uint8 and expand to RGB
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr_uint8 = (arr * 255).astype(np.uint8)
        rgb = np.stack([arr_uint8, arr_uint8, arr_uint8], axis=-1)  # (H, W, 3)

        tensor = _VAL_TF(rgb).unsqueeze(0).to(self._device)  # (1, 3, 224, 224)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

        label = int(np.argmax(probs)) + 1  # back to 1-5
        return ModelData(label=label, probabilities=probs)
