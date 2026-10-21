"""Frames for the viewer: videos and image sequences, decoded off the UI thread."""
from __future__ import annotations

import os
import threading
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # as corridorkey.video_io does

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from .clips import frame_files

_MAX_OPEN = 6


def _srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def to_rgb8(img: np.ndarray, linear_float: bool) -> np.ndarray:
    """A decoded frame (gray/BGR/BGRA; 8-bit, 16-bit or float) as display RGB.
    Float frames are scene-linear EXR, so they get an sRGB encode unless they
    are mattes (linear_float=False)."""
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    img = img[:, :, :3]  # alpha is not shown
    if img.dtype == np.uint8:
        out = img
    elif img.dtype == np.uint16:
        out = (img >> 8).astype(np.uint8)
    else:
        x = img.astype(np.float32)
        if linear_float:
            x = _srgb_encode(x)
        out = (np.clip(x, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(out[:, :, ::-1])


class FrameSource:
    """Random access to a video file or a folder of frames."""

    def __init__(self, path: Path, linear_float: bool = True):
        self.linear_float = linear_float
        self.files = frame_files(path) if path.is_dir() else None
        self.cap = None
        self._next = 0
        if self.files is None:
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise OSError(f"Cannot open {path.name}")
            self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 24.0)
        else:
            if not self.files:
                raise OSError(f"No frames in {path.name}")
            self.count = len(self.files)
            self.fps = 24.0

    def read(self, index: int) -> np.ndarray:
        if self.files is not None:
            # imdecode copes with non-ASCII paths, which cv2.imread does not on Windows.
            data = np.fromfile(str(self.files[index]), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        else:
            if index != self._next:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, img = self.cap.read()
            self._next = index + 1
            img = img if ok else None
        if img is None:
            raise OSError(f"Cannot decode frame {index + 1}")
        return to_rgb8(img, self.linear_float)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class FrameLoader(QObject):
    """Decodes one frame at a time on a worker thread. A new request replaces
    one still waiting, so scrubbing never queues up stale frames."""

    loaded = Signal(str, int, int, float, QImage)   # path, index, frame count, fps, image
    failed = Signal(str, str)                       # path, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._request: tuple[str, int, bool] | None = None
        self._release = False
        self._alive = True
        self._sources: dict[str, FrameSource] = {}
        self._thread = threading.Thread(target=self._run, name="frame-loader", daemon=True)
        self._thread.start()

    def request(self, path: Path, index: int, linear_float: bool = True) -> None:
        with self._cond:
            self._request = (str(path), index, linear_float)
            self._cond.notify()

    def release(self) -> None:
        """Close every open file, e.g. before infer.py rewrites a clip's outputs."""
        with self._cond:
            self._release = True
            self._cond.notify()

    def shutdown(self) -> None:
        with self._cond:
            self._alive = False
            self._cond.notify()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._alive and self._request is None and not self._release:
                    self._cond.wait()
                if not self._alive:
                    break
                request, self._request = self._request, None
                release, self._release = self._release, False
            if release:
                for source in self._sources.values():
                    source.close()
                self._sources.clear()
            if request is not None:
                self._load(*request)
        for source in self._sources.values():
            source.close()

    def _load(self, path: str, index: int, linear_float: bool) -> None:
        try:
            source = self._sources.get(path)
            if source is None:
                source = self._sources[path] = FrameSource(Path(path), linear_float)
                if len(self._sources) > _MAX_OPEN:
                    oldest = next(iter(self._sources))
                    self._sources.pop(oldest).close()
            index = max(0, min(index, source.count - 1))
            rgb = source.read(index)
            h, w, _ = rgb.shape
            image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            self.loaded.emit(path, index, source.count, source.fps, image)
        except Exception as exc:  # a bad frame must not kill the loader thread
            self.failed.emit(path, str(exc))
