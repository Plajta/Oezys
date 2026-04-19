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
GRID_SIZE = 64
TARGET_RESOL_UM = 0.1
CROP_SIZE = 500


def prepare_image(path: str):
    print(f"Preparing image from {path}")
    Z = convertAFMtoArray(path)

    h, w = Z.shape
    cy, cx = h // 2, w // 2
    half = CROP_SIZE // 2

    if h < CROP_SIZE or w < CROP_SIZE:
        pad_y = max(0, CROP_SIZE - h)
        pad_x = max(0, CROP_SIZE - w)
        Z = np.pad(Z, ((pad_y // 2, (pad_y + 1) // 2), (pad_x // 2, (pad_x + 1) // 2)), mode='reflect')
        h, w = Z.shape
        cy, cx = h // 2, w // 2

    crop = Z[cy - half:cy + half, cx - half:cx + half]
    norm = crop / (np.linalg.norm(crop) + 1e-8)
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
    channel = scan.get_channel('Height Sensor')
    try:
        pixels = channel.correct_lines().pixels
    except ValueError:
        pixels = channel.pixels
        pixels = pixels - np.mean(pixels, axis=1, keepdims=True)
    
    try:
        from skimage.transform import rescale
        sz = channel.size
        res_x = sz['real']['x'] / sz['pixels']['x']
        res_y = sz['real']['y'] / sz['pixels']['y']
        scale_x = res_x / TARGET_RESOL_UM
        scale_y = res_y / TARGET_RESOL_UM

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled_img = rescale(pixels, (scale_y, scale_x), anti_aliasing=True, preserve_range=True)

        return scaled_img
    except Exception as e:
        LOGW(TAG, f"Could not rescale {afmraw_path} to physical units, using raw dimensions. ({e})")
        return pixels


def _create_clean_folder(output_dir: str):
    os.makedirs(os.path.join(output_dir, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)


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

    angle_up = np.repeat(np.repeat(angle_map, ph, axis=0), pw, axis=1)[:H, :W]
    coh_up = np.repeat(np.repeat(coherence_map, ph, axis=0), pw, axis=1)[:H, :W]

    hsv = np.zeros((H, W, 3), dtype=np.float32)
    hsv[..., 0] = angle_up
    hsv[..., 1] = coh_up
    hsv[..., 2] = 1.0
    return mcolors.hsv_to_rgb(hsv)


def prepare_experimental(clean_dir: str, exp_dir: str):
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


# ---------- Imaginary (Multi-Layer & Augmented) generation ----------

def extract_afm_channel_scaled(scan, channel_name):
    try:
        channel = scan.get_channel(channel_name)
    except:
        return None

    if not channel:
        return None
    try:
        try:
            pixels = channel.correct_lines().pixels
        except ValueError:
            pixels = channel.pixels
            pixels = pixels - np.mean(pixels, axis=1, keepdims=True)
            
        from skimage.transform import rescale
        sz = channel.size
        res_x = sz['real']['x'] / sz['pixels']['x']
        res_y = sz['real']['y'] / sz['pixels']['y']
        scale_x = res_x / TARGET_RESOL_UM
        scale_y = res_y / TARGET_RESOL_UM

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = rescale(pixels, (scale_y, scale_x), anti_aliasing=True, preserve_range=True)
        return scaled
    except Exception as e:
        if 'pixels' not in locals():
            try:
                pixels = channel.pixels
            except:
                return None
        return pixels

def get_augmented_crops(rgb_image) -> np.ndarray:
    """
    Extracts crops, removes duplicates/NaNs, applies augmentations, and normalizes channels.
    Returns a NumPy array of shape (N, CROP_SIZE, CROP_SIZE, 3).
    """
    h, w, _ = rgb_image.shape
    CROP = CROP_SIZE
    crops = []

    def extract_crop(y0, x0):
        y1, x1 = y0 + CROP, x0 + CROP
        if y1 > h or x1 > w:
            return None
        return rgb_image[y0:y1, x0:x1, :]

    if h >= CROP and w >= CROP:
        crops.append(extract_crop(0, 0))
        crops.append(extract_crop(0, w - CROP))
        crops.append(extract_crop(h - CROP, 0))
        crops.append(extract_crop(h - CROP, w - CROP))
        crops.append(extract_crop((h - CROP) // 2, (w - CROP) // 2))
    else:
        pad_y = max(0, CROP - h)
        pad_x = max(0, CROP - w)
        padded = np.pad(rgb_image, ((pad_y // 2, (pad_y + 1) // 2), (pad_x // 2, (pad_x + 1) // 2), (0, 0)), mode='reflect')
        ph, pw, _ = padded.shape
        crops.append(padded[(ph - CROP) // 2:(ph + CROP) // 2, (pw - CROP) // 2:(pw + CROP) // 2, :])

    valid_crops = []
    for c in crops:
        if c is not None and not np.isnan(c).any():
            skip = False
            for vc in valid_crops:
                if np.array_equal(c, vc): skip = True
            if not skip: valid_crops.append(c)
            
    all_variations = []
    for crop in valid_crops:
        variations = [
            crop,
            np.fliplr(crop),
            np.flipud(crop),
            np.rot90(crop, k=1),
            np.rot90(crop, k=2),
            np.rot90(crop, k=3),
            np.fliplr(np.rot90(crop, k=1)),
            np.flipud(np.rot90(crop, k=1)),
        ]

        for v in variations:
            # Channel-wise normalization to 0-1 range
            v_norm = np.zeros_like(v, dtype=np.float32)
            for ch in range(3):
                c_min = v[:, :, ch].min()
                c_max = v[:, :, ch].max()
                if c_max > c_min:
                    v_norm[:,:,ch] = (v[:,:,ch] - c_min) / (c_max - c_min)
            all_variations.append(v_norm)
            
    return np.array(all_variations) if all_variations else np.array([])

def augment_and_save_crops(rgb_image, base_name, label_id, imgs_dst, labels_dst):
    augmented_arrays = get_augmented_crops(rgb_image)
    aug_idx = 0
    
    for v_norm in augmented_arrays:
        img_path = os.path.join(imgs_dst, f"{base_name}_{aug_idx}.bmp")
        lbl_path = os.path.join(labels_dst, f"{base_name}_{aug_idx}.txt")
        
        plt.imsave(img_path, v_norm)
        with open(lbl_path, 'w') as lf:
            lf.write(str(label_id))
            
        aug_idx += 1
        
    return aug_idx

def prepare_imaginary_image(src: str) -> np.ndarray:
    LOGI(TAG, f"Extracting imaginary images from {src} as np.array")
    if not os.path.exists(src):
        LOGE(TAG, f"Error: File not found -> {src}")
        return np.array([])
    
    try:
        scan = pySPM.Bruker(src)
    except Exception as e:
        LOGE(TAG, f"Failed to read file {src}: {e}")
        return np.array([])
        
    # Fetch 3 distinct layers to make RGB imaginary stack
    ch_r = extract_afm_channel_scaled(scan, "Height Sensor")
    ch_g = extract_afm_channel_scaled(scan, "Amplitude Error")
    ch_b = extract_afm_channel_scaled(scan, "Phase")
    
    if ch_r is None: 
        LOGW(TAG, f"Skipped {src}: 'Height Sensor' channel missing.")
        return np.array([])
        
    if ch_g is None: ch_g = ch_r
    if ch_b is None: ch_b = ch_r
    
    min_h = min(ch_r.shape[0], ch_g.shape[0], ch_b.shape[0])
    min_w = min(ch_r.shape[1], ch_g.shape[1], ch_b.shape[1])
    
    rgb = np.stack([
        ch_r[:min_h, :min_w],
        ch_g[:min_h, :min_w],
        ch_b[:min_h, :min_w]
    ], axis=-1)
    
    augmented_arrays = get_augmented_crops(rgb)
    
    LOGI(TAG, f"Prepared single imaginary file: generated {len(augmented_arrays)} variations.")
    return augmented_arrays

def prepare_imaginary_datas(raw_dir: str, imaginary_dir: str, classes_raw_dirs: dict):
    LOGI(TAG, f"Preparing imaginary from {raw_dir} to {imaginary_dir}")
    sentinel_file = os.path.join(imaginary_dir, ".imaginary_done")
    if os.path.exists(sentinel_file):
        LOGI(TAG, f"Imaginary data already prepared in {imaginary_dir}. Skipping.")
        return

    imgs_dst = os.path.join(imaginary_dir, "imgs")
    labels_dst = os.path.join(imaginary_dir, "labels")
    os.makedirs(imgs_dst, exist_ok=True)
    os.makedirs(labels_dst, exist_ok=True)

    file_idx = 0
    total_imgs = 0
    for class_name, label_id in classes_raw_dirs.items():
        class_path = os.path.join(raw_dir, class_name)
        if not os.path.exists(class_path):
            continue

        for entry in os.scandir(class_path):
            if entry.is_file() and is_spm_file(entry.path):
                rgb = prepare_imaginary_array(entry.path)

                if rgb is None:
                    LOGW(TAG, f"Skipping {entry.name}: Missing primary Height Sensor channel.")
                    continue

                n_saved = augment_and_save_crops(
                    rgb,
                    f"imag_{class_name}_{file_idx}",
                    label_id,
                    imgs_dst,
                    labels_dst,
                )

                total_imgs += n_saved
                LOGI(TAG, f"[imaginary] {entry.name} -> appended {n_saved} variations. Total generated so far: {total_imgs}")
                file_idx += 1

    with open(sentinel_file, 'w') as f:
        f.write("done")
    LOGI(TAG, f"Imaginary dataset ready! Generated exactly {total_imgs} images.")
