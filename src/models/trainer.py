from src.models.data import TearDataloader, TearAggregator
from src.models.data import load_config, stratified_split

from src.models.nn_model import ResNet18Model
from src.logger.logger import LOGI, LOGE
from os.path import join

import lightning as L
import torch

TAG = "TRAINER"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_nn_model(abs_path):
    LOGI(TAG, "Setting up dataset...")

    config_path = join(abs_path, "src/models/config")

    # Configs load
    resnet_config = load_config(join(config_path, "nn_models/resnet.yaml"))
    data_config = load_config(join(config_path, "data.yaml"))

    # Dataset path
    label_dir_path = join(abs_path, "data/clean/labels")
    img_dir_path = join(abs_path, "data/clean/imgs")

    aggregator = TearAggregator(
        data_config,
        label_dir_path,
        img_dir_path
    )

    # Stratified splitting + automatic augmentation on-the-grab
    labels, images = aggregator.extract()
    train, test, val = stratified_split(labels, images, data_config, DEVICE)

    LOGI(TAG, "Starting NN training sequence...")

    # Model setup
    resnet18 = ResNet18Model(resnet_config)

    model_trainer = L.Trainer()
    model_trainer.fit(model=resnet18, train_dataloaders=train)
