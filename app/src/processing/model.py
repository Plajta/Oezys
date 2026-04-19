import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from .preprocessing import PreprocessorData

NUM_CLASSES = 5
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier_best.pth")

_VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

LABELS = {
    0: "Diabetes",
    1: "PGOV_Glaukom",
    2: "SklerózaMultiplex",
    3: "SucheOko",
    4: "ZdraviLudia",
}


def _build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, NUM_CLASSES),
    )
    return model


class ModelData:
    def __init__(self, label=None, probabilities=None):
        self.Label = label
        self.Probabilities = probabilities  # list[float], one per class, sums to 1.0


class Model:
    def __init__(self, model_path: str | None = None):
        self._model_path = model_path or _MODEL_PATH
        self._model: nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self):
        if self._model is not None:
            return
        net = _build_model()
        state = torch.load(self._model_path, map_location=self._device, weights_only=True)
        net.load_state_dict(state)
        net.to(self._device).eval()
        self._model = net

    def run(self, preprocessed: PreprocessorData) -> ModelData:
        self._ensure_loaded()

        arr = preprocessed.image  # float32 H×W×3, per-channel [0,1]
        arr_u8 = (arr * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr_u8, mode="RGB")
        tensor = _VAL_TF(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

        pred = int(torch.tensor(probs).argmax())
        return ModelData(label=LABELS[pred], probabilities=probs)
