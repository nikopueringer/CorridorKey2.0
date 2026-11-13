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


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=128)
def _count_decoded_cached(path_text: str, identity: tuple[int, int, int, int],
                          ffmpeg: str, executable_identity: tuple[int, int, int, int]) -> int:
    del executable_identity  # invalidates the cache after decoder replacement
    path = Path(path_text)
    print(f"  Counting decodable frames: {path.name}", flush=True)
    # Container sample counts can include frames hidden by MP4 edit lists.
    # Match CameraVideoReader's stream selection and timestamp passthrough,
    # without expensive RGB conversion or writing decoded pixels to disk.
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
         "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
         "-fps_mode", "passthrough", "-nostats", "-progress", "pipe:1",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if _act_error(result.stderr):
        raise _act_help(path, result.stderr)
    if result.returncode or result.stderr.strip():
        raise ValueError(f"FFmpeg could not count decoded frames for {path.name}: "
                         f"{result.stderr.strip() or 'exit status ' + str(result.returncode)}")
    progress = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    try:
        count = int(progress["frame"].strip())
    except (KeyError, ValueError) as exc:
        raise ValueError(f"FFmpeg returned no decoded frame count for {path.name}") from exc
    if progress.get("progress") != "end" or count <= 0:
        raise ValueError(f"FFmpeg did not finish counting video frames for {path.name}")
    if _file_identity(path) != identity:
        raise ValueError(f"Input changed while counting frames: {path.name}; retry after the file is stable")
    return count


def count_decoded_video_frames(path: str | Path, ffmpeg_bin: str = "ffmpeg") -> int:
    """Count actual FFmpeg output frames, caching only successful stable scans.

    This full decode is for the render preflight, not interactive preview.
    The queue's process-group cancellation also stops this decoder subprocess.
    """
    video = Path(path).expanduser().resolve()
    executable = shutil.which(ffmpeg_bin)
    if executable is None:
        raise FileNotFoundError(f"ffmpeg binary not found: {ffmpeg_bin}")
    decoder = Path(executable).resolve()
    return _count_decoded_cached(str(video), _file_identity(video),
                                 str(decoder), _file_identity(decoder))
