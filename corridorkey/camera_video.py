"""High-bit-depth video decoding for camera-log profile workflows."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import os

import numpy as np


class CameraVideoReader:
    """Sequential FFmpeg RGB48 decoder for camera-log video.

    FFmpeg handles container decode and Y'CbCr-to-RGB matrix conversion;
    transfer curves and camera gamut conversion remain explicit in
    ``camera_color``. Frames are returned as normalized RGB float32 code values.
    """

    def __init__(self, path: str, *, width: int, height: int,
                 ffmpeg_bin: str = "ffmpeg", input_range: str = "auto",
                 total_frames: int | None = None):
        if input_range not in ("auto", "full", "legal"):
            raise ValueError(f"invalid input range {input_range!r}")
        if shutil.which(ffmpeg_bin) is None:
            raise FileNotFoundError(f"ffmpeg binary not found: {ffmpeg_bin}")
        self.path = str(path)
        self.width, self.height = int(width), int(height)
        self.total_frames = int(total_frames) if total_frames else None
        cmd = [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
               "-i", self.path]
        if input_range != "auto":
            # Force FFmpeg's YUV range interpretation; RGB output is full-range.
            cmd += ["-vf", f"scale=in_range={'pc' if input_range == 'full' else 'tv'}:out_range=pc"]
        cmd += ["-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough",
                "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1"]
        # Redirect stderr to a temporary file so decoder diagnostics cannot fill
        # a pipe and block raw-frame reads on long clips.
        self._stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._stderr)
        self.next_index = 0
        self.frame_bytes = self.width * self.height * 3 * 2

    def _check_diagnostics(self):
        size = os.fstat(self._stderr.fileno()).st_size
        if not size:
            return
        text = os.pread(self._stderr.fileno(), min(size, 65536),
                        max(0, size - 65536)).decode("utf-8", "replace")
        lowered = text.lower()
        markers = ("unsupported codec", "unsupported feature",
                   "decoder does not support", "not implemented",
                   "error while decoding", "failed to decode",
                   "variable act", "act flag")
        if any(marker in lowered for marker in markers):
            raise RuntimeError(f"{self.path}: FFmpeg camera decode error: {text[-2000:]}")

    def read_code_frame(self, index: int) -> np.ndarray:
        """Read an increasing absolute frame index as normalized RGB codes."""
        index = int(index)
        if index < self.next_index:
            raise ValueError("CameraVideoReader only supports increasing sequential frame indices")
        while self.next_index <= index:
            raw = self.proc.stdout.read(self.frame_bytes)
            if len(raw) != self.frame_bytes:
                self.proc.wait()
                self._stderr.seek(0)
                err = self._stderr.read().decode("utf-8", "replace")
                raise RuntimeError(
                    f"{self.path}: FFmpeg ended before frame {self.next_index}: {err}")
            frame = (np.frombuffer(raw, dtype="<u2")
                     .reshape(self.height, self.width, 3).astype(np.float32) / 65535.0)
            self.next_index += 1
            self._check_diagnostics()
        return frame

    def close(self):
        if self.proc is not None:
            proc = self.proc
            error = None
            try:
                if self.total_frames is not None and self.next_index >= self.total_frames:
                    # If the requested region reaches EOF, let FFmpeg finish so
                    # late decode diagnostics and nonzero exit status are visible.
                    trailing = False
                    while proc.stdout.read(65536):
                        trailing = True
                    if trailing:
                        error = RuntimeError(
                            f"{self.path}: FFmpeg emitted bytes beyond its probed frame count")
                    rc = proc.wait()
                    self._check_diagnostics()
                    if rc and error is None:
                        error = RuntimeError(
                            f"{self.path}: FFmpeg camera decoder exited with status {rc}")
                else:
                    if proc.stdout:
                        proc.stdout.close()
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                    proc.wait()
            finally:
                self._stderr.close()
                self.proc = None
            if error is not None:
                raise error


def read_frame_dir_codes(path: str) -> np.ndarray:
    """Read an RGB still image at its native integer code precision."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"camera-profile input must be RGB, got {image.shape} from {path}")
    rgb = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB)
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    if rgb.dtype == np.uint16:
        return rgb.astype(np.float32) / 65535.0
    if np.issubdtype(rgb.dtype, np.floating):
        return rgb.astype(np.float32, copy=False)
    raise ValueError(f"unsupported camera-profile frame dtype {rgb.dtype} in {path}")
