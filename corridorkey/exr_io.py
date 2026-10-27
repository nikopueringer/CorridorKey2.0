"""OpenEXR reading and writing through the `OpenEXR` package, laid out the way OpenCV lays it out.

OpenCV wheels from 5.0 are built without OpenEXR (`OpenEXR: NO` in `cv2.getBuildInformation()`):
`cv2.imread` / `cv2.imwrite` cannot open `.exr` files with them, whatever
`OPENCV_IO_ENABLE_OPENEXR` says. These helpers keep OpenCV's conventions so the callers did not
change: images come back as float32 with channels in B G R (A) order, single-channel images as 2-D
arrays; files are written as ZIP-compressed scanline images (`IMWRITE_EXR_COMPRESSION_ZIP`) with
channels B, G, R (A) or Y, as half or float.
"""

from __future__ import annotations

import os

import numpy as np
import OpenEXR

# OpenCV names a single-channel EXR's channel "Y" and a color one's "B", "G", "R" (and "A").
_COLOR = ("B", "G", "R", "A")


def is_exr(path: str | os.PathLike) -> bool:
    return str(path).lower().endswith(".exr")


def read_exr(path: str | os.PathLike) -> np.ndarray:
    """`cv2.imread(path, cv2.IMREAD_UNCHANGED)` of an EXR file: float32, `[h, w, 3|4]` in B G R (A)
    order for a color image, `[h, w]` for a single-channel one."""
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with OpenEXR.File(path, separate_channels=True) as f:
        planes = {name: np.asarray(ch.pixels) for name, ch in f.channels().items()}
    if all(c in planes for c in ("B", "G", "R")):
        names = [c for c in _COLOR if c in planes]
        return np.stack([planes[c].astype(np.float32, copy=False) for c in names], axis=-1)
    if "Y" in planes:
        return planes["Y"].astype(np.float32, copy=False)
    if len(planes) == 1:
        return next(iter(planes.values())).astype(np.float32, copy=False)
    raise ValueError(f"{path}: no RGB, Y or single channel to read (channels: {sorted(planes)})")


def write_exr(path: str | os.PathLike, image: np.ndarray, *, half: bool) -> None:
    """`cv2.imwrite(path, image, [IMWRITE_EXR_TYPE, HALF|FLOAT, IMWRITE_EXR_COMPRESSION, ZIP])`:
    `image` is `[h, w]` / `[h, w, 1]` (channel Y) or `[h, w, 3|4]` in B G R (A) order."""
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim == 2:
        planes = {"Y": image}
    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        planes = {name: image[..., i] for i, name in enumerate(_COLOR[: image.shape[-1]])}
    else:
        raise ValueError(f"cannot write an EXR of shape {image.shape}: expected [h, w], [h, w, 1|3|4]")
    dtype = np.float16 if half else np.float32
    channels = {name: np.ascontiguousarray(p, dtype=dtype) for name, p in planes.items()}
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, channels) as f:
        f.write(str(path))
