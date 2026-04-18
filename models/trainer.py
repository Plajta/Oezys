from data import load_config, TearDataloader, TearDataset
from nn_model import ResNet18Model


def train_nn_model():
    resnet_config = load_config("./config/nn_models/resnet.yaml")
    ResNet18Model(resnet_config)

    dataset = TearDataset()
    train_dataloader = TearDataloader(dataset)

    