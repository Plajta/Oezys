import timm
import lightning as L
import torch
import torch.nn.functional as F

from torchmetrics.classification import Accuracy


class ResNet18Model(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.num_classes = config["num_classes"]

        self.model = timm.create_model(
            'resnet18.a1_in1k',
            pretrained=True,
            num_classes=self.num_classes
        )

        # Metrics setup
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        # Backbone freeze
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreezing the classifier head
        for param in self.model.get_classifier().parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels.float())

        targets = torch.argmax(labels, dim=1)
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

        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        acc = self.test_acc(preds, targets)

        self.log_dict({"test_loss": loss, "test_acc": acc})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=float(self.config["lr"])
        )
        return optimizer
