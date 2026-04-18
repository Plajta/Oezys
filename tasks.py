from invoke import task
from src.preprocessing.config import CONFIG
from src.preprocessing.download import download_data
from src.preprocessing.prepare import prepare_datas
from src.preprocessing.cleaner import remove_raw_data, remove_clean_data, remove_all, remove_zip

# Path utils
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

    # Continue with normal pipeline (will bypass automatically if already done)
    download_data(CONFIG.source_url, PATH_RAW)
    prepare_datas(PATH_RAW, PATH_CLEAN, CONFIG.classes_raw_dirs)
