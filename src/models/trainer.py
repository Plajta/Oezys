from data import load_config, TearDataloader, TearDataset
from nn_model import ResNet18Model

import lightning as L


def train_nn_model():
    resnet_config = load_config("./config/nn_models/resnet.yaml")
    resnet18 = ResNet18Model(resnet_config)

    dataset = TearDataset()
    train_dataloader = TearDataloader(dataset)

    model_trainer = L.Trainer()
    model_trainer.fit(model=resnet18, train_dataloaders=train_dataloader)
