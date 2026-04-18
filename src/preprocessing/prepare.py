import albumentations as A
import os
import shutil
import numpy as np
import pySPM
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from src.logger.logger import LOGI, LOGE, LOGW

TAG = "PREPARER"
GRID_SIZE = 64  # direction map grid resolution (always outputs GRID_SIZE x GRID_SIZE patches)

def prepare_image(path: str):
    Z = convertAFMtoArray(path)
    norm = Z / np.linalg.norm(Z)
    return norm


def prepare_datas(input_dir: str, output_dir: str, classes_raw_dirs: dict):
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
                    img = prepare_image(entry.path)
                    plt.imsave(coverted_path, img, cmap='afmhot')
                    print(f"Saved successfully as {coverted_path}")
                    with open(label_path, 'w') as f:
                        f.write(str(classes_raw_dirs[raw_dir]))
                    index += 1
    with open(sentinel_file, 'w') as f:
        f.write("done")
    LOGI(TAG, f"Dataset prepared at {output_dir}")

def convertAFMtoArray(afmraw_path: str):
    scan = pySPM.Bruker(afmraw_path)
    height = scan.get_channel('Height Sensor').correct_lines()
    print(height)
    return height.pixels 

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


# ---------- Experimental: direction map generation ----------

def _direction_map_rgb(gray: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    Compute a per-patch dominant crystal orientation map using the structure tensor.
    Returns an RGB image (same size as input) where hue = fiber direction, saturation = coherence.
    """
    H, W = gray.shape
    ph, pw = H // grid_size, W // grid_size
    if ph == 0 or pw == 0:
        raise ValueError(f"Image too small ({H}x{W}) for grid size {grid_size}")

    Gy, Gx = np.gradient(gray)
    angle_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    coherence_map = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = i * ph, (i + 1) * ph
            x0, x1 = j * pw, (j + 1) * pw
            gx = Gx[y0:y1, x0:x1]
            gy = Gy[y0:y1, x0:x1]

            Jxx = np.mean(gx * gx)
            Jyy = np.mean(gy * gy)
            Jxy = np.mean(gx * gy)

            trace = Jxx + Jyy
            disc = max(0.0, (trace / 2.0) ** 2 - (Jxx * Jyy - Jxy ** 2))
            lam1 = trace / 2.0 + np.sqrt(disc)
            lam2 = trace / 2.0 - np.sqrt(disc)

            if lam1 + lam2 > 1e-10:
                coherence_map[i, j] = (lam1 - lam2) / (lam1 + lam2)

            fiber_angle = (0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy) + np.pi / 2.0) % np.pi
            angle_map[i, j] = fiber_angle / np.pi

    # Upsample patch maps to original image size
    angle_up = np.repeat(np.repeat(angle_map, ph, axis=0), pw, axis=1)[:H, :W]
    coh_up = np.repeat(np.repeat(coherence_map, ph, axis=0), pw, axis=1)[:H, :W]

    hsv = np.zeros((H, W, 3), dtype=np.float32)
    hsv[..., 0] = angle_up
    hsv[..., 1] = coh_up
    hsv[..., 2] = 1.0
    return mcolors.hsv_to_rgb(hsv)


def prepare_experimental(clean_dir: str, exp_dir: str):
    """
    Generate direction-map images from the already-prepared clean dataset.
    Outputs to exp_dir/imgs/ with the same filenames and copies labels from clean_dir/labels/.
    """
    LOGI(TAG, f"Preparing experimental direction maps from {clean_dir} to {exp_dir}")
    sentinel_file = os.path.join(exp_dir, ".exp_done")
    if os.path.exists(sentinel_file):
        LOGI(TAG, f"Experimental data already prepared in {exp_dir}. Skipping.")
        return

    imgs_src = os.path.join(clean_dir, "imgs")
    labels_src = os.path.join(clean_dir, "labels")
    imgs_dst = os.path.join(exp_dir, "imgs")
    labels_dst = os.path.join(exp_dir, "labels")
    os.makedirs(imgs_dst, exist_ok=True)
    os.makedirs(labels_dst, exist_ok=True)

    bmp_files = sorted([f for f in os.listdir(imgs_src) if f.endswith('.bmp')])
    for fname in bmp_files:
        stem = os.path.splitext(fname)[0]
        src_img = os.path.join(imgs_src, fname)
        dst_img = os.path.join(imgs_dst, fname)
        src_label = os.path.join(labels_src, stem + ".txt")
        dst_label = os.path.join(labels_dst, stem + ".txt")

        try:
            raw = mpimg.imread(src_img)
            gray = rgb2gray(raw) if raw.ndim == 3 else raw.astype(np.float64)
            gray = np.clip(gray.astype(np.float64), 0.0, 1.0)

            dir_rgb = _direction_map_rgb(gray)
            plt.imsave(dst_img, dir_rgb)

            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

            LOGI(TAG, f"[exp] {fname}")
        except Exception as e:
            LOGE(TAG, f"Failed on {fname}: {e}")

    with open(sentinel_file, 'w') as f:
        f.write("done")
    LOGI(TAG, f"Experimental dataset ready at {exp_dir}")
