import albumentations as A
import os
import numpy as np
import pySPM
import matplotlib.pyplot as plt
from logger.logger import LOGI, LOGE, LOGW

TAG = "PREPARER"

def prepare_data(input_dir: str, output_dir: str, classes_raw_dirs: dict):
    LOGI(TAG, f"Preparing data from {input_dir} to {output_dir}")
    sentinel_file = os.path.join(output_dir, ".prepare_done")
    if os.path.exists(sentinel_file):
        LOGI(TAG, f"Dataset already prepared in {output_dir}. Skipping.")
        return
    _create_clean_folder(output_dir)
    index = 0
    for raw_dir in classes_raw_dirs:
        raw_dir_path = os.path.join(input_dir, raw_dir)
        with os.scandir(raw_dir_path) as entries:
            for entry in entries:
                if entry.is_file() and is_spm_file(entry.path):
                    coverted_path = os.path.join(output_dir, "imgs", str(index) + ".bmp")
                    label_path = os.path.join(output_dir, "labels", str(index) + ".txt")
                    convertAFMtoImage(entry.path, coverted_path)
                    with open(label_path, 'w') as f:
                        f.write(str(classes_raw_dirs[raw_dir]))
                    index += 1
    with open(sentinel_file, 'w') as f:
        f.write("done")
    LOGI(TAG, f"Dataset prepared at {output_dir}")

def convertAFMtoImage(afmraw_path: str, output_path: str):
    scan = pySPM.Bruker(afmraw_path)
    height = scan.get_channel('Height Sensor').correct_lines()
    Z = height.pixels
    plt.imsave(output_path, Z, cmap='afmhot')
    print(f"Saved successfully as {output_path}")

def _create_clean_folder(output_dir: str):
    clean_imgs_dir_path = os.path.join(output_dir, "imgs")
    clean_labels_dir_path = os.path.join(output_dir, "labels")
    os.makedirs(clean_imgs_dir_path, exist_ok=True)
    os.makedirs(clean_labels_dir_path, exist_ok=True)

def is_spm_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            first_line = f.read(15).decode('ascii', errors='ignore')
            if first_line.startswith('\\*File list'):
                return True
    except Exception:
        pass
        
    return False
