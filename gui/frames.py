"""Frames for the viewer: videos and image sequences, decoded off the UI thread."""
from __future__ import annotations

import os
from collections import OrderedDict
import shutil
import subprocess
import threading
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # as corridorkey.video_io does

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from .clips import frame_files
from corridorkey.media_probe import validate_video
from corridorkey.camera_color import PROFILES, decode_to_working
from corridorkey.camera_video import CameraVideoReader, read_frame_dir_codes

_MAX_OPEN = 6
_CUTOUT_CACHE_BYTES = 32 * 1024 * 1024


def _srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _srgb_decode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


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


def _camera_display8(rgb_codes: np.ndarray, profile_id: str, transform: bool) -> np.ndarray:
    """Camera RGB codes to display RGB8; clipping happens only for display."""
    if transform:
        linear = decode_to_working(rgb_codes, profile_id)
        rgb = _srgb_encode(linear)
    else:
        rgb = rgb_codes
    return np.ascontiguousarray((np.clip(rgb, 0, 1) * 255 + .5).astype(np.uint8))


def _camera_codes_from_bgr(img: np.ndarray) -> np.ndarray:
    rgb = np.asarray(img[:, :, :3])[:, :, ::-1]
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255
    if rgb.dtype == np.uint16:
        return rgb.astype(np.float32) / 65535
    return rgb.astype(np.float32, copy=False)


def _linear_display8(img: np.ndarray) -> np.ndarray:
    """Interpret integer or float BGR pixels as linear Rec.709 for display."""
    rgb = _camera_codes_from_bgr(img)
    return np.ascontiguousarray((np.clip(_srgb_encode(rgb), 0, 1) * 255 + .5).astype(np.uint8))


def _camera_checker8(img: np.ndarray, profile_id: str) -> np.ndarray:
    codes = _camera_codes_from_bgr(img)
    working = decode_to_working(codes, profile_id)
    alpha = (img[:, :, 3].astype(np.float32) / (np.iinfo(img.dtype).max
             if np.issubdtype(img.dtype, np.integer) else 1.0)
             if img.ndim == 3 and img.shape[2] >= 4 else np.ones(img.shape[:2], np.float32))
    linear_bgra = np.dstack((working[:, :, ::-1], alpha))
    return to_cutout_checker8(linear_bgra, linear=True)


def to_cutout_checker8(img: np.ndarray, checker_size: int = 16, linear: bool = False,
                       source_transfer: str | None = None) -> np.ndarray:
    """Composite a straight-alpha cutout over a checker in linear light.

    By default, integer cutouts are display-coded sRGB and floating point EXRs
    are scene-linear. A sidecar can explicitly override that transfer function.
    """
    if img.ndim == 2:
        img = img[:, :, None]
    height, width = img.shape[:2]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    integer = np.issubdtype(img.dtype, np.integer)
    scale = float(np.iinfo(img.dtype).max) if integer else 1.0
    columns = np.arange(width, dtype=np.uint32)[None, :] // checker_size
    # Work in row tiles: 8K 16-bit/float sources otherwise need several GB of
    # temporary float arrays just to produce a display-sized RGB image.
    for top in range(0, height, 128):
        bottom = min(height, top + 128)
        tile = img[top:bottom]
        if tile.shape[2] == 1:
            bgr = np.repeat(tile, 3, axis=2)
            alpha = np.ones(tile.shape[:2], dtype=np.float32)
        else:
            bgr = tile[:, :, :3]
            if tile.shape[2] >= 4:
                alpha = tile[:, :, 3].astype(np.float32) / scale if integer else tile[:, :, 3].astype(np.float32)
            else:
                alpha = np.ones(tile.shape[:2], dtype=np.float32)
        foreground = bgr.astype(np.float32) / scale
        if source_transfer == "srgb" or (source_transfer is None and integer and not linear):
            foreground = _srgb_decode(foreground)
        row = np.arange(top, bottom, dtype=np.uint32)[:, None] // checker_size
        light = ((row + columns) & 1) == 0
        checker = np.where(light[:, :, None], 0.72, 0.48).astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
        composite = foreground * alpha + _srgb_decode(checker) * (1.0 - alpha)
        rgb[top:bottom] = (_srgb_encode(composite) * 255 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


class FrameSource:
    """Random access to a video file or a folder of frames."""

    def __init__(self, path: Path, linear_float: bool = True, cutout: bool = False,
                 sequence_fps: float = 24.0, cutout_linear: bool = False,
                 profile_id: str = "legacy", display_transform: bool = True,
                 input_range: str = "auto"):
        self.path = Path(path)
        self.linear_float = linear_float
        self.cutout = cutout
        self.cutout_linear = cutout_linear
        self.profile_id = profile_id
        self.display_transform = display_transform
        self.input_range = input_range
        self._camera_reader: CameraVideoReader | None = None
        self.files = frame_files(self.path) if self.path.is_dir() else None
        self.cap = None
        self._ffmpeg: subprocess.Popen | None = None
        self._ffmpeg_next = 0
        self._cutout_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._interrupted = threading.Event()
        self._next = 0
        if self.files is None:
            validate_video(self.path)
            self.cap = cv2.VideoCapture(str(self.path))
            if not self.cap.isOpened():
                raise OSError(f"Cannot open {self.path.name}")
            self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 24.0)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            if not self.files:
                raise OSError(f"No frames in {self.path.name}")
            self.count = len(self.files)
            self.fps = sequence_fps

    def read(self, index: int) -> np.ndarray:
        if self._interrupted.is_set():
            raise OSError("Frame decode stopped")
        if self.cutout and self.files is None:
            image = self._read_video_cutout(index)
            if self.profile_id != "legacy":
                if self.profile_id in PROFILES:
                    return _camera_checker8(image, self.profile_id)
                transfer = "linear" if self.profile_id == "linear_rec709" else "srgb"
                return to_cutout_checker8(image, linear=transfer == "linear", source_transfer=transfer)
            return to_cutout_checker8(image, linear=self.cutout_linear)
        if self.cutout and self.files is not None and self.profile_id in PROFILES and self.profile_id != "legacy":
            data = np.fromfile(str(self.files[index]), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise OSError(f"Cannot decode frame {index + 1}")
            return _camera_checker8(img, self.profile_id)
        if self.profile_id in PROFILES and self.profile_id != "legacy":
            if self.files is not None:
                codes = read_frame_dir_codes(str(self.files[index]))
                return _camera_display8(codes, self.profile_id, self.display_transform)
            if index < (self._camera_reader.next_index if self._camera_reader else 0):
                self._camera_reader.close()
                self._camera_reader = None
            if self._camera_reader is None:
                self._camera_reader = CameraVideoReader(
                    str(self.path), width=self.width, height=self.height,
                    input_range=self.input_range, total_frames=self.count)
            codes = self._camera_reader.read_code_frame(index)
            return _camera_display8(codes, self.profile_id, self.display_transform)
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
        if self.profile_id in PROFILES and self.profile_id != "legacy":
            codes = _camera_codes_from_bgr(img)
            return _camera_display8(codes, self.profile_id, self.display_transform)
        if self.profile_id == "linear_rec709":
            if self.cutout:
                return to_cutout_checker8(img, linear=True)
            return _linear_display8(img)
        if self.profile_id in ("srgb", "matte"):
            if self.cutout:
                return to_cutout_checker8(img, linear=False, source_transfer="srgb")
            return to_rgb8(img, linear_float=False)
        return (to_cutout_checker8(img, linear=self.cutout_linear)
                if self.cutout else to_rgb8(img, self.linear_float))

    def _read_video_cutout(self, index: int) -> np.ndarray:
        if self._interrupted.is_set():
            raise OSError("Frame decode stopped")
        cached = self._cutout_cache.get(index)
        if cached is not None:
            self._cutout_cache.move_to_end(index)
            return cached
        if index < self._ffmpeg_next:
            self._stop_ffmpeg()
        if self._ffmpeg is None:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise OSError("Cutout video preview needs ffmpeg on PATH to preserve alpha")
            self._ffmpeg = subprocess.Popen(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(self.path),
                 "-map", "0:v:0", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgra", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
            )
            if self._interrupted.is_set():
                self._ffmpeg.terminate()
            self._ffmpeg_next = 0
        frame_size = self.width * self.height * 4
        requested = None
        while self._ffmpeg_next <= index:
            if self._interrupted.is_set():
                self._stop_ffmpeg()
                raise OSError("Frame decode stopped")
            parts = bytearray()
            while len(parts) < frame_size:
                chunk = self._ffmpeg.stdout.read(frame_size - len(parts))
                if not chunk:
                    break
                parts.extend(chunk)
            raw = bytes(parts)
            if len(raw) != frame_size:
                self._stop_ffmpeg()
                raise OSError(f"ffmpeg could not decode cutout frame {self._ffmpeg_next + 1}")
            bgra = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4).copy()
            current = self._ffmpeg_next
            self._ffmpeg_next += 1
            if current == index:
                requested = bgra
            self._cutout_cache[current] = bgra
            self._cutout_cache.move_to_end(current)
            while sum(frame.nbytes for frame in self._cutout_cache.values()) > _CUTOUT_CACHE_BYTES:
                self._cutout_cache.popitem(last=False)
        return self._cutout_cache.get(index, requested)

    def _stop_ffmpeg(self) -> None:
        proc, self._ffmpeg = self._ffmpeg, None
        if proc is None:
            return
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1)
        self._ffmpeg_next = 0

    def close(self) -> None:
        self._stop_ffmpeg()
        if self._camera_reader is not None:
            self._camera_reader.close()
            self._camera_reader = None
        if self.cap is not None:
            self.cap.release()

    def interrupt(self) -> None:
        """Interrupt the FFmpeg reader; VideoCapture stays owned by its reader thread."""
        self._interrupted.set()
        proc = self._ffmpeg
        if proc is not None and proc.poll() is None:
            proc.terminate()
        reader = self._camera_reader
        if reader is not None and reader.proc is not None and reader.proc.poll() is None:
            reader.proc.terminate()


class FrameLoader(QObject):
    """Decodes one frame at a time on a worker thread. A new request replaces
    one still waiting, so scrubbing never queues up stale frames."""

    loaded = Signal(str, int, int, float, QImage)   # path, index, frame count, fps, image
    failed = Signal(str, str)                       # path, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cond = threading.Condition()
        self._request: tuple[str, int, bool, bool, float, bool, str, bool, str, int] | None = None
        self._release = False
        self._alive = True
        self._sources: dict[str, FrameSource] = {}
        self._active_lock = threading.Lock()
        self._active_source: FrameSource | None = None
        self._active_generation: int | None = None
        self._shutdown_event = threading.Event()
        self._generation = 0
        self._thread = threading.Thread(target=self._run, name="frame-loader", daemon=True)
        self._thread.start()

    def request(self, path: Path, index: int, linear_float: bool = True,
                cutout: bool = False, sequence_fps: float = 24.0,
                cutout_linear: bool = False, profile_id: str = "legacy",
                display_transform: bool = True, input_range: str = "auto") -> None:
        with self._cond:
            self._generation += 1
            generation = self._generation
            self._request = (str(path), index, linear_float, cutout,
                             sequence_fps, cutout_linear, profile_id,
                             display_transform, input_range, generation)
            self._interrupt_active_before(generation)
            self._cond.notify()

    def release(self) -> None:
        """Close every open file, e.g. before infer.py rewrites a clip's outputs."""
        with self._cond:
            self._release = True
            self._generation += 1
            self._request = None
            generation = self._generation
            self._interrupt_active_before(generation)
            self._cond.notify()

    def shutdown(self) -> None:
        self._shutdown_event.set()
        with self._cond:
            self._alive = False
            self._generation += 1
            self._request = None
            generation = self._generation
            self._interrupt_active_before(generation)
            self._cond.notify()
        self._thread.join(timeout=1.8)

    def _interrupt_active_before(self, generation: int) -> None:
        with self._active_lock:
            if (self._active_source is not None and self._active_generation is not None and
                    self._active_generation < generation):
                self._active_source.interrupt()

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

    def _load(self, path: str, index: int, linear_float: bool, cutout: bool,
              sequence_fps: float, cutout_linear: bool, profile_id: str,
              display_transform: bool, input_range: str, generation: int) -> None:
        try:
            if self._is_stale(generation):
                return
            source = self._sources.get(path)
            if (source is not None and (source.linear_float != linear_float or
                    source.cutout != cutout or source.cutout_linear != cutout_linear or
                    source._interrupted.is_set() or source.profile_id != profile_id or
                    source.display_transform != display_transform or source.input_range != input_range)):
                source.close()
                del self._sources[path]
                source = None
            if source is None:
                source = self._sources[path] = FrameSource(Path(path), linear_float,
                                                          cutout, sequence_fps, cutout_linear,
                                                          profile_id, display_transform, input_range)
                if len(self._sources) > _MAX_OPEN:
                    oldest = next(iter(self._sources))
                    self._sources.pop(oldest).close()
            elif source.files is not None:
                source.fps = sequence_fps
            with self._active_lock:
                self._active_source = source
                self._active_generation = generation
            if self._is_stale(generation):
                source.interrupt()
                return
            index = max(0, min(index, source.count - 1))
            rgb = source.read(index)
            if self._is_stale(generation):
                return
            h, w, _ = rgb.shape
            image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            self.loaded.emit(path, index, source.count, source.fps, image)
        except Exception as exc:  # a bad frame must not kill the loader thread
            if not self._is_stale(generation):
                self.failed.emit(path, str(exc))
        finally:
            with self._active_lock:
                if self._active_generation == generation:
                    self._active_source = None
                    self._active_generation = None

    def _is_stale(self, generation: int) -> bool:
        with self._cond:
            return self._shutdown_event.is_set() or generation != self._generation
