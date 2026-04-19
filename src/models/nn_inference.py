#
# Package requiring everything to run sample ResNet18 inference
# TODO: now is deorganized, but thats for the future
#

import torch
import cv2
import lightning as L
import albumentations as A
import torch.nn.functional as F
import timm

from torchmetrics.classification import Accuracy, F1Score
from albumentations import ToTensorV2


class ResNet18Model(L.LightningModule):
    def __init__(self, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = 5

        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        self.model = timm.create_model(
            'resnet18.a1_in1k',
            pretrained=True,
            num_classes=self.num_classes
        )

        # Metrics setup
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        # F1 score setup
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

        # Backbone freeze
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreezing the classifier head
        for param in self.model.get_classifier().parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class ResNetInference:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ResNet18Model.load_from_checkpoint(
            model_path,
            weights_only=False
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def run(self, path):
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        tensor = self.transform(image=img_rgb)["image"].unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)

        return probs.tolist()


if __name__ == "__main__":
    checkpoint = "../../stashed_checkpoints/run32best-checkpoint-epoch=36-val_acc=0.85.ckpt"
    img_path = "../../data/imaginary/imgs/imag_Diabetes_0_0.bmp"

    model_inference = ResNetInference(checkpoint)
    print(model_inference.run(img_path))
