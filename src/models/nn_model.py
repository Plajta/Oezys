import timm
import lightning as L
import torch
import torch.nn.functional as F

from torchmetrics.classification import Accuracy, F1Score


class ResNet18Model(L.LightningModule):
    def __init__(self, config, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.num_classes = config["num_classes"]

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

    def _shared_step(self, batch):
        images, targets = batch
        logits = self(images)

        loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss, preds, targets = self._shared_step(batch)
        acc = self.train_acc(preds, targets)

        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        acc = self.val_acc(preds, targets)
        f1 = self.val_f1(preds, targets)

        self.log_dict({"val_loss": loss, "val_acc": acc, "val_f1": f1}, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        acc = self.test_acc(preds, targets)
        f1 = self.test_f1(preds, targets)

        self.log_dict({"test_loss": loss, "test_acc": acc, "test_f1": f1})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=float(self.config["lr"]),
            weight_decay=float(self.config["weight_decay"])
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss" # Tell Lightning to watch the validation loss
            }
        }
