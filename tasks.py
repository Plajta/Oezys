from invoke import task

#
# Preprocessing imports
#
from src.preprocessing.config import CONFIG
from src.preprocessing.download import download_data
from src.preprocessing.prepare import prepare_datas, prepare_experimental, prepare_imaginary_datas
from src.preprocessing.cleaner import remove_raw_data, remove_clean_data, remove_all, remove_zip

#
# Trainer imports
#
from src.models.trainer import run_full_pipeline, run_top_pipeline
from src.models.data import DataInspector, load_config

#
# Inference module
#
from src.models.test_inference import inference_setup

from pathlib import Path
from os.path import join

import shutil
import os

ABS_PATH = str(Path(__file__).parent)


@task
def prepare(ctx, clean=None):
    """RadBrecim Data Preparation Pipeline"""
    PATH_RAW = join(ABS_PATH, CONFIG.raw_dir)
    PATH_CLEAN = join(ABS_PATH, CONFIG.clean_dir)

    match(clean):
        case "raw":
            remove_raw_data(PATH_RAW)
        case "zip":
            remove_zip(PATH_RAW)
        case "prepared":
            remove_clean_data(PATH_CLEAN)
        case "all":
            remove_all(PATH_RAW, PATH_CLEAN)

    download_data(CONFIG.source_url, PATH_RAW)
    prepare_datas(PATH_RAW, PATH_CLEAN, CONFIG.classes_raw_dirs)


@task
def experimental(ctx, clean=False):
    """Generate direction-map images from clean dataset into data/clean/exp/"""
    import shutil, os
    if clean and os.path.exists(CONFIG.exp_dir):
        shutil.rmtree(CONFIG.exp_dir)
    prepare_experimental(CONFIG.clean_dir, CONFIG.exp_dir)


@task
def imaginary(ctx, clean=False):
    """Generate multi-layered (Amplitude/Phase/Height) augmented crops into data/imaginary/"""
    import shutil, os
    if clean and os.path.exists(CONFIG.imaginary_dir):
        shutil.rmtree(CONFIG.imaginary_dir)
    prepare_imaginary_datas(CONFIG.raw_dir, CONFIG.imaginary_dir, CONFIG.classes_raw_dirs)


@task
def run_pipeline(ctx):
    """Oezys Neural training sequence"""
    run_full_pipeline(ABS_PATH)


@task
def run_moe_pipeline(ctx):
    """Oezys MoE training sequence"""

    rf_model_best = join(ABS_PATH, "src/models/config/cl_models/tear_classifier.pkl")
    resnet_model_best = join(ABS_PATH, "checkpoints/resnet18/best-checkpoint-epoch=25-val_acc=0.88.ckpt")

    run_top_pipeline(
        ABS_PATH,
        rf_model_best,
        resnet_model_best
    )


@task
def inspect_dataset(ctx):
    """RadBrecim Dataset inspector"""
    data_inspector = DataInspector(ABS_PATH)
    data_inspector.debug_dataloader(n_batches=3)


@task
def wandb_login(ctx):
    ctx.run("wandb login", pty=True)


@task
def stash_checkpoints(ctx):
    naming = input("Checkpoint names: ")
    directory = input("Directory: ")

    checkpoints_dir = join(ABS_PATH, "checkpoints", directory)
    stashed_checkpoints_dir = join(ABS_PATH, "stashed_checkpoints")

    for i, filename in enumerate(os.listdir(checkpoints_dir)):
        orig_file_path = join(checkpoints_dir, filename)
        new_file_path = join(stashed_checkpoints_dir, f"{naming}{i}{filename}")
        shutil.move(orig_file_path, new_file_path)


@task
def run_inference(ctx):
    resnet_config_path = "src/models/config/nn_models/resnet.yaml"
    resnet_infer_config_path = "src/models/config/nn_models/resnet_inference.yaml"

    resnet_config_abs_path = join(ABS_PATH, resnet_config_path)
    inference_config = load_config(join(ABS_PATH, resnet_infer_config_path))

    model_name = inference_config["model_name"]
    checkpoint_path = "stashed_checkpoints"
    imgs_path = "data/clean/imgs"

    inference_setup(
        checkpoint_path=join(checkpoint_path, model_name),
        config_path=resnet_config_abs_path,
        image_folder=imgs_path
    )