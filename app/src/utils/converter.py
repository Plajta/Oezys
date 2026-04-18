import re
import numpy as np
from PIL import Image


def parse_afm_to_bmp(file_path, out_path):
    with open(file_path, "rb") as f:
        content = f.read()

    header_text = content.decode(errors="ignore")

    def get_int(key, default=None):
        m = re.search(rf"\\{key}:\s*([0-9]+)", header_text)
        return int(m.group(1)) if m else default

    width = get_int("Samps/line", 512)
    height = get_int("Lines", 512)
    bytes_per_pixel = get_int("Bytes/pixel", 2)
    data_offset = get_int("Data offset")
    data_length = get_int("Data length")

    if data_offset is not None:
        raw = content[data_offset: data_offset + (data_length or width * height * bytes_per_pixel)]
    else:
        # fallback: binary starts after the last ASCII header marker
        end_marker = b"\\*File list end"
        marker_pos = content.rfind(end_marker)
        if marker_pos != -1:
            raw = content[marker_pos + len(end_marker):]
            raw = raw.lstrip(b"\r\n")
        else:
            raw = content[-(width * height * bytes_per_pixel):]

    dtype = np.int16 if bytes_per_pixel == 2 else np.float32
    data = np.frombuffer(raw[:width * height * bytes_per_pixel], dtype=dtype)

    if data.size != width * height:
        # infer square dimensions from available data
        side = int(np.sqrt(data.size))
        data = data[:side * side]
        width = height = side

    img = data.reshape((height, width)).astype(np.float32)
    img -= img.min()
    img /= (img.max() + 1e-8)
    img = (img * 255).astype(np.uint8)
    img = np.flipud(img)

    Image.fromarray(img).save(out_path, format="BMP")
