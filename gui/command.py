"""Job settings and the infer.py command line they turn into.

The command matches what the terminal wizard (run_interactive.py) assembles,
plus the infer.py switches the wizard leaves at their defaults.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from run_interactive import DEFAULT_TEMP_DIR

from .clips import HINT_FIRST_FRAME, HINT_VIDEO, Clip, frame_files

ROOT = Path(__file__).resolve().parents[1]
INFER = ROOT / "infer.py"
ALL_OUTPUTS = ("alpha", "fg", "cutout", "checker")


@dataclass
class JobSettings:
    outputs: list[str] = field(default_factory=lambda: list(ALL_OUTPUTS))
    output_format: str = "video"     # video | exr
    exr_dtype: str = "half"          # half | float
    cutout_linear: bool = False
    despill_strength: float = 0.5
    hint_quality: float = 0.75
    carry_hint: bool = True
    carry_hint_quality: float = 0.95
    frame_colour: str = "auto"       # auto (EXR linear, PNG/JPG sRGB) | linear | srgb
    frame_dir_fps: float = 24.0
    auto_vram: bool = True
    low_vram: bool = False
    hann_chunk: int = 48
    hann_stride: int = 24
    temp_dir: str = DEFAULT_TEMP_DIR
    num_frames: int = 0              # 0 = every frame
    start_mode: str = "begin"        # begin | middle | random_middle
    output_parent: str = ""          # "" = out_<clip> next to each clip

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "JobSettings":
        names = {f.name for f in fields(cls)}
        try:
            return cls(**{k: v for k, v in data.items() if k in names})
        except TypeError:
            return cls()


def frames_are_linear(clip: Clip, settings: JobSettings) -> bool:
    if settings.frame_colour != "auto":
        return settings.frame_colour == "linear"
    files = frame_files(clip.source)
    return bool(files) and files[0].suffix.lower() == ".exr"


def infer_args(clip: Clip, s: JobSettings, output_dir: Path) -> list[str]:
    args = [str(INFER), "--input", str(clip.source), "--output_dir", str(output_dir)]
    if clip.hint_path is not None and clip.hint_kind == HINT_FIRST_FRAME:
        args += ["--hint_first_frame", str(clip.hint_path)]
    elif clip.hint_path is not None and clip.hint_kind == HINT_VIDEO:
        args += ["--hint_video", str(clip.hint_path)]
    args += ["--hint_quality", f"{s.hint_quality:g}"]
    if s.carry_hint:
        args += ["--carry_hint_quality", f"{s.carry_hint_quality:g}"]
    else:
        args.append("--no_carry_hint")
    args += ["--outputs", *s.outputs, "--output_format", s.output_format]
    if s.output_format == "exr":
        args += ["--exr_dtype", s.exr_dtype]
    elif s.cutout_linear:
        args.append("--cutout_linear")
    args += ["--despill_strength", f"{s.despill_strength:g}"]
    if clip.is_frames:
        args += ["--frame_dir_fps", f"{s.frame_dir_fps:g}"]
        if frames_are_linear(clip, s):
            args.append("--frame_dir_linear")
    # With no memory flags, infer.py sizes the windows from free GPU memory.
    if not s.auto_vram:
        if s.low_vram:
            args.append("--low_vram")
        args += ["--hann_chunk", str(s.hann_chunk), "--hann_stride", str(s.hann_stride)]
    args += ["--temp_dir", s.temp_dir]
    if s.num_frames > 0:
        args += ["--num_frames", str(s.num_frames), "--start_mode", s.start_mode]
    return args


def python_exe() -> str:
    """The venv's console interpreter. infer.py streams its progress on stdout,
    so it runs under python.exe even when the UI was started with pythonw.exe."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").is_file():
        return str(exe.with_name("python.exe"))
    return str(exe)


def command_line(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv)
