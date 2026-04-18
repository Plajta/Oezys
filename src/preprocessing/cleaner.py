import os
import shutil
from src.logger.logger import LOGI, LOGE

TAG = "CLEANER"

def remove_clean_data(clean_dir: str):
    if os.path.exists(clean_dir):
        try:
            shutil.rmtree(clean_dir)
            LOGI(TAG, f"Removed clean data directory: {clean_dir}")
        except Exception as e:
            LOGE(TAG, f"Failed to remove clean data: {e}")
    else:
        LOGI(TAG, f"Clean data directory {clean_dir} does not exist. Skipping.")

def remove_raw_data(raw_dir: str):
    if os.path.exists(raw_dir):
        try:
            for item in os.listdir(raw_dir):
                if item == "data.zip":
                    continue
                item_path = os.path.join(raw_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            LOGI(TAG, f"Removed extracted raw data in {raw_dir} (kept zip).")
        except Exception as e:
            LOGE(TAG, f"Failed to remove raw data: {e}")
    else:
        LOGI(TAG, f"Raw data directory {raw_dir} does not exist. Skipping.")

def remove_zip(raw_dir: str):
    zip_path = os.path.join(raw_dir, "data.zip")
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
            LOGI(TAG, f"Removed zip file: {zip_path}")
        except Exception as e:
            LOGE(TAG, f"Failed to remove zip file: {e}")
    else:
        LOGI(TAG, f"Zip file {zip_path} does not exist. Skipping.")

def remove_all(raw_dir: str, clean_dir: str):
    remove_clean_data(clean_dir)
    remove_raw_data(raw_dir)
    remove_zip(raw_dir)
    LOGI(TAG, "Completed full dataset cleanup.")