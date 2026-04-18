import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


def convertAFMtoImage(afmraw_path: str) -> Image.Image:
    import pySPM
    scan = pySPM.Bruker(afmraw_path)
    height = scan.get_channel('Height Sensor').correct_lines()
    Z = height.pixels
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-12)
    rgba = (cm.afmhot(Z_norm) * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode='RGBA').convert('RGB')


def convertAFMtoArray(afmraw_path: str) -> np.ndarray:
    import pySPM
    scan = pySPM.Bruker(afmraw_path)
    height = scan.get_channel('Height Sensor').correct_lines()
    return height.pixels


def prepare_image(path: str) -> np.ndarray:
    Z = convertAFMtoArray(path)
    return Z / np.linalg.norm(Z)


def is_spm_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            first_line = f.read(15).decode('ascii', errors='ignore')
            if first_line.startswith('\*File list'):
                return True
    except Exception:
        pass

    return False
