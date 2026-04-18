import timm
import lightning as L
import torch


class ResNet18Model(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = timm.create_model('resnet18.a1_in1k', pretrained=True)
        self.model = self.model.eval()
        self.config = config

    def training_step():

        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config)
        return optimizer
