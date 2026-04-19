import warnings
import numpy as np
import matplotlib.cm as cm
from PIL import Image

TARGET_RESOL_UM = 0.1
CROP_SIZE = 500


def _extract_channel(scan, channel_name: str):
    try:
        channel = scan.get_channel(channel_name)
    except Exception:
        return None
    try:
        try:
            pixels = channel.correct_lines().pixels
        except ValueError:
            pixels = channel.pixels
            pixels = pixels - np.mean(pixels, axis=1, keepdims=True)
        from skimage.transform import rescale
        sz = channel.size
        scale_x = (sz['real']['x'] / sz['pixels']['x']) / TARGET_RESOL_UM
        scale_y = (sz['real']['y'] / sz['pixels']['y']) / TARGET_RESOL_UM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rescale(pixels, (scale_y, scale_x), anti_aliasing=True, preserve_range=True)
    except Exception:
        return getattr(channel, 'pixels', None)


def convertAFMtoArray(afmraw_path: str) -> np.ndarray:
    import pySPM
    return _extract_channel(pySPM.Bruker(afmraw_path), 'Height Sensor')


def convertAFMtoImage(afmraw_path: str) -> Image.Image:
    Z = convertAFMtoArray(afmraw_path)
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-12)
    rgba = (cm.afmhot(Z_norm) * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode='RGBA').convert('RGB')


def prepare_image(path: str) -> np.ndarray:
    """Returns H×W×3 float32 in [0,1] — identical pipeline to prepare_imaginary_datas."""
    import pySPM
    scan = pySPM.Bruker(path)
    ch_r = _extract_channel(scan, 'Height Sensor')
    ch_g = _extract_channel(scan, 'Amplitude Error')
    ch_b = _extract_channel(scan, 'Phase')

    if ch_r is None:
        raise ValueError(f"Could not load Height Sensor from {path}")
    if ch_g is None:
        ch_g = ch_r
    if ch_b is None:
        ch_b = ch_r

    min_h = min(ch_r.shape[0], ch_g.shape[0], ch_b.shape[0])
    min_w = min(ch_r.shape[1], ch_g.shape[1], ch_b.shape[1])
    rgb = np.stack([ch_r[:min_h, :min_w], ch_g[:min_h, :min_w], ch_b[:min_h, :min_w]], axis=-1)

    h, w = rgb.shape[:2]
    if h < CROP_SIZE or w < CROP_SIZE:
        pad_y = max(0, CROP_SIZE - h)
        pad_x = max(0, CROP_SIZE - w)
        rgb = np.pad(rgb, ((pad_y // 2, (pad_y + 1) // 2), (pad_x // 2, (pad_x + 1) // 2), (0, 0)), mode='reflect')
        h, w = rgb.shape[:2]
    cy, cx = h // 2, w // 2
    half = CROP_SIZE // 2
    crop = rgb[cy - half: cy + half, cx - half: cx + half]

    result = np.zeros(crop.shape, dtype=np.float32)
    for ch in range(3):
        c_min, c_max = crop[:, :, ch].min(), crop[:, :, ch].max()
        if c_max > c_min:
            result[:, :, ch] = (crop[:, :, ch] - c_min) / (c_max - c_min)
    return result


def is_spm_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            first_line = f.read(15).decode('ascii', errors='ignore')
            if first_line.startswith(r'\*File list'):
                return True
    except Exception:
        pass
    return False
