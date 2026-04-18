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
from src.models.trainer import train_nn_model
from src.models.data import DataInspector

from pathlib import Path
from os.path import join

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
<<<<<<< HEAD
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
=======
def train_nn(ctx):
    """RadBrecim Neural training sequence"""
    train_nn_model(ABS_PATH)


@task
def inspect_dataset(ctx):
    """RadBrecim Dataset inspector"""
    data_inspector = DataInspector(ABS_PATH)
    data_inspector.debug_dataloader(n_batches=3)
>>>>>>> 7a6ae4d (Finally loading the data)
