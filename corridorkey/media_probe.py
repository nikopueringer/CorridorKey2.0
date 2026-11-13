"""Small ffprobe wrapper for early, cached checks of video inputs.

This module intentionally has no OpenCV or PyTorch imports so GUI callers can
validate a clip before opening either runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import shutil
import subprocess
from pathlib import Path


class UnsupportedVideoError(ValueError):
    """The installed FFmpeg cannot decode this input's codec features."""


@dataclass(frozen=True)
class VideoInfo:
    codec: str
    width: int
    height: int
    fps: float
    frame_count: int | None


def _act_error(stderr: str) -> bool:
    message = stderr.lower()
    return ("variable act flag" in message or
            ("unsupported" in message and "adaptive color transform" in message))


def _act_help(path: Path, stderr: str) -> UnsupportedVideoError:
    return UnsupportedVideoError(
        f"Cannot decode {path.name}: this DNxHR/DNxHD stream uses a variable ACT flag "
        "that the installed FFmpeg does not support. Export ProRes 4444 or an EXR "
        "frame sequence from Resolve, then try again.\n\nffprobe: " + stderr.strip()
    )


@lru_cache(maxsize=128)
def _probe_cached(path_text: str, mtime_ns: int, size: int, ffprobe_bin: str) -> VideoInfo:
    del mtime_ns, size  # included in the cache key so file replacements are reprobed
    path = Path(path_text)
    ffprobe = shutil.which(ffprobe_bin)
    if ffprobe is None:
        raise FileNotFoundError(f"ffprobe binary not found: {ffprobe_bin}. Install FFmpeg.")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=8,
    )
    if _act_error(result.stderr):
        raise _act_help(path, result.stderr)
    if result.returncode:
        raise ValueError(f"ffprobe could not inspect {path.name}: {result.stderr.strip()}")
    try:
        streams = json.loads(result.stdout).get("streams", [])
        stream = streams[0]
        rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
        numerator, denominator = (float(part) for part in rate.split("/", 1))
        fps = numerator / denominator if denominator else 0.0
        raw_count = stream.get("nb_frames")
        frame_count = int(raw_count) if raw_count not in (None, "N/A") else None
        return VideoInfo(str(stream.get("codec_name", "unknown")),
                         int(stream["width"]), int(stream["height"]), fps, frame_count)
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"ffprobe returned incomplete video metadata for {path.name}") from exc


def probe_video(path: str | Path, ffprobe_bin: str = "ffprobe") -> VideoInfo:
    """Probe one video stream, caching by canonical path, mtime, and size."""
    video = Path(path).expanduser().resolve()
    stat = video.stat()
    return _probe_cached(str(video), stat.st_mtime_ns, stat.st_size, ffprobe_bin)


def validate_video(path: str | Path, ffprobe_bin: str = "ffprobe") -> VideoInfo:
    """Reject known unsupported variable-ACT streams with a conversion hint."""
    return probe_video(path, ffprobe_bin)
