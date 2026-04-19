from src.models.data import TearDataloader, TearAggregator
from src.models.data import load_config, stratified_split
from lightning.pytorch.loggers import WandbLogger
from src.models.nn_model import ResNet18Model
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from src.logger.logger import LOGI, LOGE
from os.path import join

import lightning as L
import numpy as np
import torch

TAG = "TRAINER"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_full_pipeline(abs_path):
    LOGI(TAG, "Setting up dataset...")

    config_path = join(abs_path, "src/models/config")
    resnet_config = load_config(join(config_path, "nn_models/resnet.yaml"))
    data_config = load_config(join(config_path, "data.yaml"))

    dataloader_config = data_config["dataloader"]

    # Dataset path
    label_dir_path = join(abs_path, "data/imaginary/labels")
    img_dir_path = join(abs_path, "data/imaginary/imgs")

    aggregator = TearAggregator(
        data_config,
        label_dir_path,
        img_dir_path
    )

    # Stratified splitting + automatic augmentation on-the-grab
    labels, images = aggregator.extract()

    train, test, val = stratified_split(labels, images, data_config, DEVICE)

    train_loader = TearDataloader(train, dataloader_config)
    test_loader = TearDataloader(test, dataloader_config)
    val_loader = TearDataloader(val, dataloader_config)

    LOGI(TAG, "Starting NN training sequence...")

    # Logs to W&B
    wandb_logger = WandbLogger(project="ResNet18-FineTune", name="resnet_experiment_v1")

    checkpoint_callback = ModelCheckpoint(
        dirpath=join(abs_path, resnet_config["checkpoint_path"]),
        filename="best-checkpoint-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        monitor="val_acc",
        mode="max"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=15, # Stop if no improvement for 10 epochs
        mode="min"
    )

    # Weight for every class (because of unbalanced dataset)
    unique_classes, counts = np.unique(train.labels, return_counts=True)
    total_train_samples = len(train.labels)
    num_classes = len(unique_classes)
    weights = total_train_samples / (num_classes * counts)

    # Model setup
    resnet18 = ResNet18Model(resnet_config, class_weights=weights)

    # Model training
    model_trainer = L.Trainer(
        max_epochs=resnet_config["epochs"],
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=3,
        accumulate_grad_batches=2
    )
    model_trainer.fit(
        model=resnet18,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    # Model testing
    LOGI(TAG, "Starting NN testing sequence...")
    model_trainer.test(resnet18, dataloaders=test_loader, ckpt_path="best", weights_only=False)


def run_top_pipeline():
    LOGI(TAG, "Setting up dataset...")

    

    LOGI(TAG, "Startin MoE-NN testing sequence")
