"""Clips in the queue: discovery, companion hints and where outputs land.

Discovery goes through the terminal wizard's helpers in run_interactive.py, so
a clip dropped on the window picks up the same hints the wizard would find.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from corridorkey.outputs import OUTPUT_SUFFIXES, default_output_dir, read_outputs

from run_interactive import (
    check_image_sequence_dir,
    find_alpha_hint,
    find_first_frame_hint,
    get_batch_clips,
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
# The frame types infer.py reads from a folder (corridorkey.video_io.frame_paths).
FRAME_EXTS = {".exr", ".png", ".jpg", ".jpeg"}

HINT_NONE = "none"
HINT_FIRST_FRAME = "first_frame"
HINT_VIDEO = "video"

QUEUED, RUNNING, DONE, FAILED, STOPPED = "Queued", "Running", "Done", "Failed", "Stopped"

@dataclass(eq=False)
class Clip:
    # path is what was added (a video, a frame folder or an Input/AlphaHint
    # folder); source is what infer.py reads from it as --input.
    path: Path
    source: Path
    is_frames: bool
    hint_kind: str = HINT_NONE
    hint_path: Path | None = None
    input_color_space: str = "legacy"
    status: str = QUEUED
    progress: float = 0.0
    detail: str = ""
    timing_summary: str = ""
    output_dir: Path | None = None  # where the last run wrote

    @property
    def name(self) -> str:
        return raw_clip_name(self.source)

    @property
    def hint_label(self) -> str:
        if self.hint_kind == HINT_NONE or self.hint_path is None:
            return "None"
        kind = "Frame" if self.hint_kind == HINT_FIRST_FRAME else "Video"
        return f"{kind} · {self.hint_path.name}"


def raw_clip_name(path) -> str:
    """corridorkey.video_io.raw_clip_name, copied because importing that module loads torch."""
    norm = os.path.normpath(str(path))
    name = os.path.splitext(os.path.basename(norm))[0]
    if name.lower() in {"input", "frames", "rgb"}:
        return os.path.basename(os.path.dirname(norm)) or name
    return name


def frame_files(folder: Path) -> list[Path]:
    try:
        return sorted((p for p in Path(folder).iterdir()
                       if p.is_file() and p.suffix.lower() in FRAME_EXTS),
                      key=lambda p: [int(t) if t.isdigit() else t
                                     for t in re.split(r"(\d+)", p.name.lower())])
    except OSError:
        return []


def discover(path: Path) -> list[Clip]:
    """Every clip in a dropped path: a video, a frame folder, an Input/AlphaHint
    folder, or a folder holding any of those."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTS and "_alphahint" not in path.name.lower():
            return [_video_clip(path)]
        return []
    if not path.is_dir():
        return []
    seq_input, seq_hint = check_image_sequence_dir(path)
    if seq_input:
        return [Clip(path, seq_input, True, HINT_VIDEO if seq_hint else HINT_NONE, seq_hint)]
    if frame_files(path):
        return [Clip(path, path, True)]
    return [clip for item in get_batch_clips(path) for clip in discover(item)]


def _video_clip(path: Path) -> Clip:
    # The wizard's batch-mode precedence, then its first-frame lookup.
    hint = find_alpha_hint(path)
    if hint is not None:
        kind = HINT_VIDEO if hint.suffix.lower() in VIDEO_EXTS else HINT_FIRST_FRAME
        return Clip(path, path, False, kind, hint)
    hint = find_first_frame_hint(path)
    if hint is not None:
        return Clip(path, path, False, HINT_FIRST_FRAME, hint)
    return Clip(path, path, False)


def output_dir_for(clip: Clip, output_parent: str = "") -> Path:
    """Stable source-specific destination, independent of queue order."""
    parent = Path(output_parent) if output_parent else clip.path.parent
    return default_output_dir(clip.source, parent=parent, display_name=clip.path.stem)


def find_outputs(clip: Clip, output_dir: Path) -> dict[str, Path]:
    """Only completed generations, or flat outputs from older versions."""
    return read_outputs(output_dir, clip.source, clip.name)
