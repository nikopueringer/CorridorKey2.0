from __future__ import annotations

import contextlib
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import color_utils as cu
from .blocks import set_region_index_cache
from .model import CorridorKeyV3
from .device import area_resize, empty_cache, memory_budget_gb, model_device, synchronize
from .mps_attention import available as native_attention_available
from .pinned import HostBuffer
from .video_io import (
    ExrSequenceWriter,
    ThreadedVideoWriter,
    WriterConfig,
    apply_linear_adjust,
    frame_paths,
    has_linear_adjust,
    iter_foundation_frames,
    iter_frame_dir_chunks,
    iter_frame_dir_views_on_device,
    iter_video_frame_chunks,
    load_initial_hint_frame,
    make_checker,
    make_hint_reader,
    none_path,
    preprocess_frame_dir_path,
    preprocess_video_frame_views,
    probe_frame_dir_meta,
    probe_video_meta,
    raw_clip_name,
    safe_remove,
    taper_weights,
    window_starts,
    SequentialFrameDirWindowReader,
    SequentialVideoWindowReader,
)


@dataclass
class RuntimePaths:
    package_root: Path
    weights: Path
    foundation_stats: Path
    rvm_ckpt: Path
    moge_checkpoint: Path
    cradio_dir: Path

    @staticmethod
    def _resolve_user_or_package_path(root: Path, value, default: str) -> Path:
        if value is None:
            return (root / default).resolve()
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        cwd_path = (Path.cwd() / path).resolve()
        if cwd_path.exists():
            return cwd_path
        return (root / path).resolve()

    @staticmethod
    def _prefer_safetensors(root: Path, stem: str) -> str:
        # Legacy pickle .pt files are still accepted if that is all that is present.
        if not (root / f"{stem}.safetensors").exists() and (root / f"{stem}.pt").exists():
            return f"{stem}.pt"
        return f"{stem}.safetensors"

    @classmethod
    def from_package(cls, package_root: str | os.PathLike,
                     weights: str | os.PathLike | None = None):
        root = Path(package_root).resolve()
        return cls(
            package_root=root,
            weights=cls._resolve_user_or_package_path(
                root, weights, cls._prefer_safetensors(root, "weights/corridorkey_v2")),
            foundation_stats=(
                root / cls._prefer_safetensors(root, "weights/foundation_stats")).resolve(),
            rvm_ckpt=(root / "vfms/rvm/pretrain_rvm_small16_256_204031069.npz").resolve(),
            moge_checkpoint=(root / "vfms/moge/model.pt").resolve(),
            cradio_dir=(root / "vfms/cradio/C-RADIOv4-SO400M").resolve(),
        )

    def validate(self) -> None:
        missing = [
            p for p in (
                self.weights, self.foundation_stats, self.rvm_ckpt,
                self.moge_checkpoint, self.cradio_dir,
            )
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "missing runtime asset(s): " + ", ".join(str(p) for p in missing)
                + "\nRun `python scripts/download_weights.py` to fetch them.")


@dataclass
class InferenceConfig:
    hann_chunk: int = 80
    hann_stride: int = 40
    carry_hint: bool = True
    hint_quality: float = 0.95
    carry_hint_quality: float = 0.95
    shape_quantum: int = 128
    model_scale: float = 1.0
    num_frames: int = -1
    start_mode: str = "begin"
    seed: int = 0
    low_vram: bool = False
    # Pick hann_chunk/hann_stride/low_vram per clip from free GPU memory.
    auto_vram: bool = False
    # Auto VRAM: read the memory table measured for the optimized path
    # (bf16 weights + lean activations) instead of the shipped one.
    auto_vram_optimized: bool = False
    # Set by auto-VRAM: estimate of the chosen profile, and the estimate that
    # any retry after an OOM must stay below.
    vram_estimate_gb: float | None = None
    vram_ceiling_gb: float | None = None
    low_vram_foundation_chunk: int = 4
    low_vram_mlp_chunk_tokens: int = 65536
    low_vram_enc_frame_chunk: int = 1
    low_vram_swin_window_batch: int = 16
    low_vram_full_attn_query_chunk_tokens: int = 2048
    stage2_batch: int = 1
    outputs: set[str] = field(default_factory=lambda: {"alpha", "fg"})
    temp_dir: Path = Path(tempfile.gettempdir()) / "corridorkey_v2"
    linear_brightness: float = 1.0
    linear_contrast: float = 1.0
    linear_contrast_pivot: float = 0.18
    frame_dir_fps: float = 24.0
    frame_dir_linear: bool = False
    cutout_linear: bool = False
    despill_strength: float = 0.5
    output_format: str = "video"
    exr_dtype: str = "half"
    # Opt-in: batched-window attention without full-grid padded copies.
    lean_activations: bool = False
    # Opt-in: decode/preprocess the next window's frames on a background
    # thread while the GPU runs the current one.
    prefetch_frames: bool = False
    # Opt-in: keep shifted-window region index tensors on the GPU.
    cache_region_indices: bool = False
    # Opt-in, for allocators without expandable segments: reserve the
    # window's estimated working memory as one segment before stage 1.
    activation_arena: bool = False
    # Opt-in: quantize output frames on the GPU, copy them to pinned host
    # memory without stalling the GPU, and hand them to the writers one batch
    # later. Same bytes.
    pinned_io: bool = False
    # Opt-in, low-VRAM mode: accumulate stage-1 batons and write finished
    # frames on a background thread while the GPU runs the next window.
    # Same arithmetic in the same order.
    async_baton: bool = False
    # Opt-in: where a stage only uses the model's view of a frame (low-VRAM
    # stage 1 after the prepass, low-VRAM stage 2), skip preparing the
    # display views. Same model input.
    lean_reads: bool = False
    # Opt-in, with prefetch_frames: read stage 1's first window during the
    # prepass, later windows on the worker threads, and stage 2's frames
    # eight at a time. Same frames.
    read_ahead: bool = False
    # Opt-in, with prefetch_frames: the prepass decodes each frame once and
    # keeps the model's input (its exact bf16 bits) in a temporary memmap
    # that stages 1 and 2 read instead of decoding the clip again; in normal
    # mode, where stage 1 runs the foundation models itself, their view too.
    # Used when the memmaps fit in half the temp folder's free space.
    decode_once: bool = False
    # Threads the prepass prepares frames on.
    read_workers: int = 4
    gpu_prep: bool = False
    # Opt-in: EXR frames compressed and written by this many threads per
    # output (video_io.ExrSequenceWriter); the same files.
    exr_workers: int = 1


# Peak GPU memory in GiB by megapixels of model input, measured at each Hann
# chunk size. Keys are low_vram False/True.
_VRAM_PEAK_GB = {
    False: ((8, 24, 48, 80), (
        (0.98, (6.19, 7.52, 9.77, 12.89)),
        (2.21, (7.50, 11.33, 18.66, 24.89)),
        (3.93, (9.06, 14.78, 23.92, 33.19)),
        (8.36, (13.94, 25.96, 45.15, 70.45)),
    )),
    True: ((1, 2, 4, 8, 24, 48, 80), (
        (0.98, (2.71, 2.78, 2.94, 3.25, 4.54, 6.57, 9.50)),
        (2.21, (2.80, 2.97, 3.35, 4.12, 7.69, 13.20, 20.56)),
        (3.93, (2.92, 3.23, 3.85, 5.09, 10.09, 17.57, 27.75)),
        (8.36, (3.25, 3.92, 5.28, 8.38, 20.78, 39.66, 64.65)),
    )),
}
_VRAM_MARGIN_SCALE = 1.1
# Fixed headroom for other GPU processes and allocator slack.
_VRAM_HEADROOM_GB = 1.5
_VRAM_HEADROOM_FRAC = 0.05


def _vram_headroom_gb(total_gb: float) -> float:
    return max(_VRAM_HEADROOM_GB, _VRAM_HEADROOM_FRAC * total_gb)

# The same table for the --optimized path (bf16 Linear/Conv weights, lean
# activations), measured on an RTX 6000 Ada
# (torch 2.9.1, CUDA 12.8), with the window sizes in between so the selector
# can use what a card actually has. Normal mode at 80 frames for 4K and 4K
# DCI is extrapolated from 56 and 64 (out of memory on the 48 GB card).
_VRAM_PEAK_GB_OPTIMIZED = {
    False: ((1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80), (
        (0.98, (2.58, 2.58, 2.63, 2.77, 2.91, 3.05, 3.18, 3.32, 3.46, 3.74, 4.02, 4.3, 4.58, 5.14, 5.7, 6.26, 6.82, 7.93)),
        (2.21, (2.95, 2.95, 3.09, 3.47, 3.84, 4.21, 4.59, 4.96, 5.33, 6.08, 6.83, 7.57, 8.32, 7.94, 8.49, 9.37, 10.37, 12.38)),
        (3.93, (3.31, 3.31, 3.52, 4.11, 4.69, 5.28, 5.86, 6.45, 7.04, 8.21, 9.38, 8.59, 9.48, 11.26, 13.05, 14.83, 16.61, 20.18)),
        (5.51, (3.71, 3.71, 4.01, 4.84, 5.67, 6.5, 7.33, 8.16, 8.99, 8.61, 9.84, 11.08, 12.33, 14.83, 17.33, 19.82, 22.32, 27.31)),
        (8.36, (4.48, 4.48, 5.03, 6.37, 7.71, 9.06, 8.42, 8.98, 9.92, 11.82, 13.71, 15.61, 17.5, 21.29, 25.08, 28.87, 32.66, 40.24)),
        (8.91, (4.63, 4.63, 5.21, 6.65, 8.08, 9.51, 8.65, 9.42, 10.43, 12.45, 14.47, 16.49, 18.51, 22.55, 26.6, 30.64, 34.68, 42.76)),
    )),
    True: ((1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80), (
        (0.98, (1.17, 1.17, 1.17, 1.26, 1.38, 1.51, 1.64, 1.76, 1.89, 2.15, 2.4, 2.66, 2.91, 3.42, 3.93, 4.44, 4.95, 5.97)),
        (2.21, (1.24, 1.24, 1.24, 1.35, 1.5, 1.66, 1.82, 1.97, 2.13, 2.45, 2.76, 3.08, 3.39, 4.02, 4.65, 5.28, 5.91, 7.17)),
        (3.93, (1.35, 1.35, 1.35, 1.55, 1.79, 1.95, 2.13, 2.34, 2.54, 2.96, 3.38, 3.8, 4.22, 5.05, 5.89, 6.72, 7.56, 9.45)),
        (5.51, (1.75, 1.75, 1.75, 1.85, 2.03, 2.2, 2.46, 2.73, 2.99, 3.52, 4.05, 4.58, 5.11, 6.17, 7.24, 8.31, 9.38, 11.5)),
        (8.36, (2.52, 2.52, 2.52, 2.52, 2.52, 2.87, 3.27, 3.67, 4.07, 4.87, 5.67, 6.47, 7.27, 9.1, 10.75, 12.4, 14.04, 17.33)),
        (8.91, (2.66, 2.66, 2.66, 2.66, 2.66, 3.01, 3.43, 3.86, 4.28, 5.14, 5.99, 6.84, 7.9, 9.65, 11.41, 13.16, 14.92, 18.43)),
    )),
}


def optimized_vram_margin(megapixels: float, expandable_segments: bool):
    """(scale, extra GiB) over the --optimized table's estimate, one pair or
    a dict of them keyed by low_vram.

    With expandable segments (Linux) reserved memory stays within ~0.4 GiB
    of the allocated peak. Without them (Windows) the driver spills to
    system RAM before an allocation fails, so nothing makes the allocator
    release its cache and the margin has to cover what it holds
    unconstrained. With the activation arena that is the arena itself,
    _ARENA_SCALE times the estimate's working memory, plus the model, up to
    1440p. From 3K up some low-VRAM windows still overflow the arena once,
    to at most 1.50x the estimate (3K 40, 4K 12, 24 and 32 frames), so
    low-VRAM windows there get that margin. Capping the
    process at its budget instead (with max_split_size_mb) fails on Windows:
    the cache fills the cap, the display has nothing left and the driver
    errors.
    """
    if expandable_segments:
        return 1.05, 0.3
    low = _ARENA_SCALE[True] if megapixels < 4.5 else 1.5
    return {True: (low, 0.3), False: (1.08, 0.3)}


def _interp(x: float, points) -> float:
    """Piecewise-linear through (x, y) points, extrapolating past the ends."""
    points = list(points)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1 or (x1, y1) == points[-1]:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[0][1]


def estimate_peak_vram_gb(megapixels: float, frames: int, low_vram: bool,
                          table=None) -> float:
    """Estimated peak GPU memory for a Hann window of `frames` frames."""
    chunks, rows = (table or _VRAM_PEAK_GB)[low_vram]
    frames = max(frames, chunks[0])
    per_chunk = [
        max(_interp(megapixels, [(mp, peaks[i]) for mp, peaks in rows]), rows[0][1][i])
        for i in range(len(chunks))
    ]
    return _interp(frames, zip(chunks, per_chunk))


def select_vram_profile(shape_hw, num_frames: int, budget_gb: float,
                        table=None, margin=None, ceiling_gb: float | None = None):
    """Largest Hann window that fits `budget_gb`, as (chunk, stride, low_vram, estimate).

    Bigger windows give better temporal consistency, and low-VRAM mode is
    exact, so a larger low-VRAM window is preferred over a smaller normal one.
    `table`/`margin` default to the shipped `_VRAM_PEAK_GB` and margins;
    `margin` is (scale, extra GiB), or a dict of those keyed by low_vram.
    Profiles estimated at `ceiling_gb` or more are skipped. Falls back to a
    single-frame window when nothing is estimated to fit.
    """
    table = table or _VRAM_PEAK_GB
    margin = margin or (_VRAM_MARGIN_SCALE, _VRAM_MARGIN_GB)
    megapixels = shape_hw[0] * shape_hw[1] / 1e6
    estimate = 0.0
    for chunk in reversed(table[True][0]):
        for low_vram in (False, True):
            if chunk not in table[low_vram][0]:
                continue
            estimate = estimate_peak_vram_gb(
                megapixels, min(chunk, num_frames), low_vram, table)
            if ceiling_gb is not None and estimate >= ceiling_gb:
                continue
            scale, extra_gb = margin[low_vram] if isinstance(margin, dict) else margin
            if estimate * scale + extra_gb <= budget_gb:
                return chunk, max(1, chunk // 2), low_vram, estimate
    return 1, 1, True, estimate


def configure_allocator() -> None:
    """Expandable allocator segments on Linux, so freed memory never strands
    in partly used segments, unless PYTORCH_CUDA_ALLOC_CONF already says
    otherwise. Call before the first CUDA allocation.

    Windows builds of PyTorch have no expandable segments and keep the
    default allocator, with the activation arena. `max_split_size_mb` would
    stop the large-request failures there too, but it lets cached memory grow
    to several times the working set when nothing forces a release, and on
    Windows the driver spills to system RAM instead of failing an
    allocation; capping the process at its budget to force the release
    fails on the real hardware."""
    if sys.platform.startswith("linux") and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")


def max_split_configured() -> bool:
    """Whether the caching allocator was given max_split_size_mb."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") + os.environ.get("PYTORCH_ALLOC_CONF", "")
    return "max_split_size_mb" in conf


def expandable_segments_active() -> bool:
    """Whether the CUDA caching allocator is handing out expandable segments."""
    if not torch.cuda.is_available():
        return False
    return any(seg.get("is_expandable") for seg in torch.cuda.memory_snapshot())


# The activation arena is this many times the optimized table's estimate,
# less what is already allocated, keyed by low_vram. Low-VRAM windows need
# the headroom: below x1.15 a large request can find no gap big enough and
# take a segment of its own (allocator replays of 720p to 4K snapshots).
# Normal-mode estimates already sit above the allocated peak.
_ARENA_SCALE = {True: 1.15, False: 1.05}


def reserve_activation_arena(nbytes: int) -> int:
    """Hold `nbytes` of GPU memory as one free cached segment.

    PyTorch's default caching allocator (no expandable segments, as on
    Windows) never merges memory across segments: each cudaMalloc it makes is
    sized for the request that caused it, and once freed it can only serve
    requests that fit inside it. Over a stage-1 window that leaves about 30%
    of the reserved memory in pieces too small for the next large tensor.
    Allocating one block of the window's working size and freeing it again
    makes the window's large tensors come out of that single segment, where
    blocks freed together merge back. Only addresses change, never values.

    Skipped when the allocator has expandable segments (they already avoid
    this) or `max_split_size_mb` (blocks above it are never split, so the
    arena would sit unused). Returns the bytes reserved."""
    if not torch.cuda.is_available() or max_split_configured() or expandable_segments_active():
        return 0
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info()
    nbytes = int(min(nbytes, free_bytes - (256 << 20)))
    if nbytes < (64 << 20):
        return 0
    block = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    del block
    return nbytes


# Fixed headroom floor for other GPU processes and allocator slack.
_VRAM_HEADROOM_GB = 1.5
_VRAM_HEADROOM_FRAC = 0.05


def _vram_headroom_gb(total_gb: float) -> float:
    return max(_VRAM_HEADROOM_GB, _VRAM_HEADROOM_FRAC * total_gb)


def _read_vram_budget_gb() -> tuple[float, float]:
    """(budget, total) in GiB for CUDA; reads twice a second apart.

    Keeps the lower value to avoid being misled by another process still
    releasing memory at startup.
    """
    gc.collect()
    torch.cuda.empty_cache()
    readings = []
    for i in range(2):
        if i:
            time.sleep(1.0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        readings.append((free_bytes + torch.cuda.memory_reserved()) / 2**30)
    if max(readings) - min(readings) > 1.0:
        print(f"  [WARNING] GPU free memory is changing ({readings[0]:.1f} -> "
              f"{readings[1]:.1f} GB); another process is using this GPU.")
    return min(readings), total_bytes / 2**30


_MPS_ESTIMATE_SCALE = 1.5


def apply_auto_vram(cfg: InferenceConfig, shape_hw, num_frames: int, device="cuda") -> None:
    dev = torch.device(device)
    if dev.type == "cuda":
        budget_gb, total_gb = _read_vram_budget_gb()
        headroom_gb = _vram_headroom_gb(total_gb)
    else:
        budget_gb = memory_budget_gb(device)
        headroom_gb = _VRAM_HEADROOM_GB
    # CUDA reference peaks understate MPS memory usage (larger attention
    # intermediates, unified-memory pressure from other allocations) and
    # recommended_max_memory() returns virtual address space rather than the
    # actual MPS watermark ceiling. Deflate the effective budget so the window
    # selector is conservative enough to avoid MPS OOM.
    effective_budget_gb = budget_gb / _MPS_ESTIMATE_SCALE if dev.type == "mps" else budget_gb
    table = margin = None
    if cfg.auto_vram_optimized:
        table = _VRAM_PEAK_GB_OPTIMIZED
        expandable = expandable_segments_active() if dev.type == "cuda" else False
        margin = optimized_vram_margin(shape_hw[0] * shape_hw[1] / 1e6, expandable)
    # Incorporate headroom into the margin's extra_gb floor.
    if margin is None:
        margin = (_VRAM_MARGIN_SCALE, headroom_gb)
    elif isinstance(margin, dict):
        margin = {k: (s, max(e, headroom_gb)) for k, (s, e) in margin.items()}
    else:
        margin = (margin[0], max(margin[1], headroom_gb))
    chunk, stride, low_vram, estimate = select_vram_profile(
        shape_hw, num_frames, effective_budget_gb, table, margin,
        ceiling_gb=cfg.vram_ceiling_gb)
    cfg.hann_chunk, cfg.hann_stride, cfg.low_vram = chunk, stride, low_vram
    cfg.vram_estimate_gb = estimate
    if dev.type == "mps" and low_vram:
        # CUDA-sized batches retain much larger attention intermediates on MPS.
        # This applies only to automatic memory selection. Explicit Hann or
        # low-VRAM mode retains the user's batch settings.
        for name, limit in (
            ("low_vram_full_attn_query_chunk_tokens", 512),
            ("low_vram_swin_window_batch", 4),
            ("low_vram_mlp_chunk_tokens", 16384),
        ):
            current = getattr(cfg, name)
            if current > 0:
                setattr(cfg, name, min(current, limit))
        if budget_gb < 6:
            # Small unified-memory Macs need smaller *spatial* batches too.
            native_attention = native_attention_available()
            for name, limit in (
                ("low_vram_foundation_chunk", 1),
                ("low_vram_full_attn_query_chunk_tokens", 512 if native_attention else 128),
                ("low_vram_swin_window_batch", 4 if native_attention else 1),
                ("low_vram_mlp_chunk_tokens", 16384 if native_attention else 4096),
            ):
                current = getattr(cfg, name)
                if current > 0:
                    setattr(cfg, name, min(current, limit))
    estimate_label = "CUDA reference peak" if dev.type == "mps" else "estimated peak"
    scale, extra_gb = margin[low_vram] if isinstance(margin, dict) else margin
    margin_gb = effective_budget_gb - (estimate * scale + extra_gb)
    print(f"  Auto VRAM: {budget_gb:.1f} GB available at {shape_hw[1]}x{shape_hw[0]} -> "
          f"hann_chunk={chunk}, hann_stride={stride}, low_vram={low_vram} "
          f"({estimate_label} {estimate:.1f} GB, margin {margin_gb:+.1f} GB)")
    if margin_gb < 0:
        print("  [WARNING] Even a single-frame window may not fit this GPU at this "
              "resolution. Inference may run out of memory.")


def prepare_runtime_environment(paths: RuntimePaths) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    moge_root = paths.package_root / "third_party/MoGe"
    if str(moge_root) not in sys.path:
        sys.path.insert(0, str(moge_root))


def _resolve_package_path(paths: RuntimePaths, value: str | None) -> str | None:
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = paths.package_root / p
    return str(p.resolve())


def load_runtime_checkpoint(path: Path) -> dict:
    """Read a runtime checkpoint from .safetensors (preferred) or legacy .pt."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework="pt") as f:
            header = f.metadata() or {}
        return {
            "format": header.get("format"),
            "model_config": json.loads(header.get("model_config", "{}")),
            "metadata": json.loads(header.get("metadata", "{}")),
            "state_dict": load_file(str(path), device="cpu"),
        }
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


_BF16_WEIGHT_MODULES = (
    nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)


@contextlib.contextmanager
def _skip_param_init():
    """Random parameter initializers as no-ops while the model is built.

    Every tensor they would fill is replaced afterwards: the stages from the
    runtime checkpoint (load_model fails on any missing key) and the frozen
    foundations from their own checkpoints. Filling them first is several
    seconds of CPU work (mostly DINOv2's truncated-normal init and stage 1's
    Kaiming init). Every parameter and buffer comes out bitwise the same."""
    import torch.nn.init as init
    names = ("uniform_", "normal_", "_no_grad_trunc_normal_", "kaiming_uniform_",
             "kaiming_normal_", "xavier_uniform_", "xavier_normal_", "orthogonal_")
    saved = {n: getattr(init, n) for n in names if hasattr(init, n)}
    tensor_saved = {n: getattr(torch.Tensor, n) for n in ("uniform_", "normal_")}

    def keep(tensor, *_args, **_kwargs):
        return tensor

    try:
        for n in saved:
            setattr(init, n, keep)
        for n in tensor_saved:
            setattr(torch.Tensor, n, keep)
        yield
    finally:
        for n, fn in saved.items():
            setattr(init, n, fn)
        for n, fn in tensor_saved.items():
            setattr(torch.Tensor, n, fn)


def load_model(paths: RuntimePaths, device: str = "cuda", *,
               skip_init: bool = False) -> CorridorKeyV3:
    """`skip_init`: build the modules without random initialization (see
    _skip_param_init); the loaded weights are the same."""
    paths.validate()
    prepare_runtime_environment(paths)
    ckpt = load_runtime_checkpoint(paths.weights)
    if ckpt.get("format") != "corridorkey_v2_runtime_checkpoint":
        raise RuntimeError(f"{paths.weights} is not a CorridorKey v2 runtime checkpoint")
    cfg = dict(ckpt["model_config"])
    cfg["cradio_repo"] = str(paths.cradio_dir)
    cfg["moge_checkpoint"] = str(paths.moge_checkpoint)
    cfg["rvm_ckpt"] = str(paths.rvm_ckpt)
    cfg["foundation_stats_path"] = str(paths.foundation_stats)
    cfg["gradient_checkpoint"] = False
    cfg["device"] = device
    cfg["stage1_recurrent_64_sites"] = ()

    print(f"Loading CorridorKey v2 runtime weights: {paths.weights}")
    with _skip_param_init() if skip_init else contextlib.nullcontext():
        model = CorridorKeyV3(**cfg)
    model._runtime_device = torch.device(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    train_missing = [k for k in missing if not k.startswith("foundations.")]
    train_unexpected = [k for k in unexpected if not k.startswith("foundations.")]
    if train_missing or train_unexpected:
        raise RuntimeError(
            "runtime checkpoint did not match model: "
            f"missing={train_missing[:8]} unexpected={train_unexpected[:8]}")
    print(f"  loaded weights; omitted frozen foundation tensors={len(missing)}")
    model.eval()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, _BF16_WEIGHT_MODULES):
            for p in module.parameters(recurse=False):
                p.data = p.data.to(torch.bfloat16)
    del ckpt
    return model


def apply_weight_policy(model, *, freeze: bool = False,
                        bf16_linear_conv: bool = False) -> None:
    """Opt-in inference weight storage. Both options leave outputs bit-identical.

    freeze: requires_grad_(False) on every parameter. load_model leaves the
        trained stages with requires_grad=True, and autocast then caches a
        bf16 copy of each such parameter for the whole autocast region (a full
        stage-1 window), which holds up to half the stage-1 weight size at the
        memory peak.
    bf16_linear_conv: store nn.Linear / nn.Conv* weights and biases as bf16
        (implies freeze). Autocast casts exactly these to bf16 on every call,
        so the kernels see the same values; norms, scales and buffers stay
        fp32. Halves the weight footprint on the GPU and in host RAM.
    """
    if not (freeze or bf16_linear_conv):
        return
    model.requires_grad_(False)
    if bf16_linear_conv:
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.modules.conv._ConvNd)):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_UNUSED_MOGE_PARTS = ("neck", "points_head", "normal_head", "mask_head", "scale_head")


def drop_unused_foundation_parts(model) -> int:
    moge = getattr(model.foundations, "moge", None)
    freed = 0
    for name in _UNUSED_MOGE_PARTS if moge is not None else ():
        part = getattr(moge, name, None)
        if isinstance(part, torch.nn.Module):
            freed += sum(p.numel() for p in part.parameters())
            setattr(moge, name, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return freed


def configure_low_vram_inference(model, cfg: InferenceConfig):
    model.stage1.low_vram_enc_frame_chunk = max(0, int(cfg.low_vram_enc_frame_chunk))
    spatial_batches = model_device(model).type == "mps"
    if spatial_batches and native_attention_available():
        print("  Native MPSGraph attention enabled for chunked full-spatial layers", flush=True)
    for module in model.modules():
        name = module.__class__.__name__
        if name in ("SwiGLU", "FusionLinSwiglu"):
            module.inference_chunk_tokens = max(0, int(cfg.low_vram_mlp_chunk_tokens))
        if name == "SwinAttention3D" or (spatial_batches and name == "SwinAttention2D"):
            module.inference_window_batch = max(0, int(cfg.low_vram_swin_window_batch))
        if name == "CSwinAttention3D" or (spatial_batches and name == "CSwinAttention2D"):
            module.inference_strip_batch = max(0, int(cfg.low_vram_swin_window_batch))
        if name in ("FullSpatialAttention2D", "FullAttention3D"):
            module.inference_query_chunk_tokens = max(
                0, int(cfg.low_vram_full_attn_query_chunk_tokens))


def configure_lean_activations(model, enabled: bool) -> None:
    """When enabled, route batched-window attention through
    SwinAttention3D._forward_lean (same math, no full-grid padded copies),
    let Stage1 hand tensor ownership to its mid/decoder stacks so input
    and skip duplicates are freed early, give the Swin-3D/CSwin-3D
    attention, MLP and local-mix branches their normalized input already in
    bf16 (the dtype their Linear/Conv layers cast it to anyway), run the
    per-frame QKV projection + depthwise conv one frame and one window-batch
    band of rows at a time and the local mix one frame at a time, let the
    mid/decoder blocks add their branches into the residual
    stream in place, compute the MLP gate in place, park the late-skip
    features in host memory until the decoder needs them, concat and
    patchify the stage-1 input one frame at a time, free the CSwin strip
    path's intermediates as soon as they are consumed, add the stage-1 skip
    connections in place, and (in the runtime) hand stage 1 its window
    inputs so they are freed once consumed."""
    for module in model.modules():
        name = module.__class__.__name__
        if name in ("SwinAttention3D", "Stage1"):
            module.inference_lean = bool(enabled)
        if name == "Stage1":
            module.inference_offload_late_skip = bool(enabled)
            module.inference_stem_frame_chunk = 1 if enabled else 0
            module.inference_inplace_skips = bool(enabled)
        if name == "SwinAttention3D":
            module.inference_qkv_frame_chunk = 1 if enabled else 0
            module.inference_qkv_band = bool(enabled)
        if name == "CSwinAttention3D":
            module.inference_free_strips = bool(enabled)
        if name in ("SwinBlock3D", "CSwinBlock3D", "DepthwiseSwiGLU2D"):
            module.inference_bf16_branch_input = bool(enabled)
        if name in ("SwinBlock3D", "CSwinBlock3D", "FullSpatialBlock3D",
                    "FullSpatioTemporalBlock"):
            module.inference_local_mix_frame_chunk = 1 if enabled else 0
        if name == "SwiGLU":
            module.inference_inplace_mlp = bool(enabled)
    # The mid/decoder stacks run through Stage1._swin3d_run_lean, which owns
    # their residual stream, so their blocks may add into it in place.
    stage1 = getattr(model, "stage1", None)
    for stack in (getattr(stage1, "mid_16", ()), getattr(stage1, "dec_16", ())):
        for blk in stack:
            blk.inference_inplace_residual = bool(enabled)


def _build_foundation_inputs(rgb_bt: torch.Tensor):
    B, T, _, H, W = rgb_bt.shape
    flat = rgb_bt.reshape(B * T, 3, H, W)
    rgb_radio = area_resize(flat, size=(H // 4, W // 4))
    H_moge, W_moge = (H * 7) // 32, (W * 7) // 32
    rgb_moge = area_resize(flat, size=(H_moge, W_moge))
    return (
        rgb_radio.reshape(B, T, 3, H // 4, W // 4),
        rgb_moge.reshape(B, T, 3, H_moge, W_moge),
    )


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


class _ModelViewReader:
    """Window reader over the views the prepass wrote (decode_once): the
    same bf16 tensors the frame readers would build, read back bit for bit.
    The model view, and with `foundation_path` the foundation view (returned
    as the review view too, as the frame-folder readers do); otherwise the
    other views come back as None."""

    def __init__(self, path: Path, meta: dict, foundation_path: Path | None = None):
        H, W = meta["shape_hw"]
        shape = (int(meta["num_frames"]), 3, H, W)
        self.views = np.memmap(path, dtype=np.int16, mode="r", shape=shape)
        self.foundation = (np.memmap(foundation_path, dtype=np.int16, mode="r", shape=shape)
                           if foundation_path is not None else None)
        self.model_only = self.foundation is None
        self.next_t = 0

    def _read_until(self, t_end: int):
        self.next_t = max(self.next_t, t_end)

    def window(self, t0: int, t1: int):
        model = torch.from_numpy(np.array(self.views[t0:t1])).view(torch.bfloat16)
        if self.foundation is None:
            return model, None, None
        found = torch.from_numpy(np.array(self.foundation[t0:t1])).view(torch.bfloat16)
        return model, found, found

    def close(self):
        del self.views, self.foundation


class _PrefetchingWindowReader:
    """Wraps a Sequential*WindowReader so the frames of the next window are
    read and preprocessed on a background thread while the caller works on
    the current one. The first window, which nothing else overlaps, is
    started at once and preprocessed `workers` frames at a time with the
    reader's own per-frame function (a video is still decoded in order);
    later windows are read one frame at a time behind the GPU work, where
    more threads would only compete with the main thread on a small CPU, or,
    with `parallel`, `workers` frames at a time as well. The tensors are
    identical; window() waits for any read in flight before touching the
    reader, so the reader is never used from two threads at once."""

    def __init__(self, reader, windows, workers: int = 4, parallel: bool = False):
        self.reader = reader
        self.ends = [t1 for _t0, t1 in windows]
        self.parallel = parallel
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.workers = ThreadPoolExecutor(max_workers=max(1, workers))
        self.pending = None
        if self.ends:
            self.pending = self.pool.submit(self._read_parallel, self.ends[0])

    def _read_parallel(self, t_end: int):
        r = self.reader
        need_display = not getattr(r, "model_only", False)
        futures = []
        if isinstance(r, SequentialFrameDirWindowReader):
            for t in range(r.next_t, t_end):
                futures.append((t, self.workers.submit(
                    preprocess_frame_dir_path, r.paths[t], r.meta, need_model=True,
                    need_display=need_display)))
        elif isinstance(r, SequentialVideoWindowReader):
            for t in range(r.next_t, t_end):
                ret, frame = r.cap.read()
                if not ret:
                    raise RuntimeError(
                        f"{r.video_path}: failed decoding frame {r.start_frame + t}")
                futures.append((t, self.workers.submit(
                    preprocess_video_frame_views, frame, r.meta, need_model=True,
                    need_display=need_display)))
        else:
            r._read_until(t_end)
            return
        for t, fut in futures:
            r.cache[t] = fut.result()
        r.next_t = max(r.next_t, t_end)

    def _wait(self):
        if self.pending is not None:
            self.pending.result()
            self.pending = None

    def window(self, t0: int, t1: int):
        self._wait()
        out = self.reader.window(t0, t1)
        later = [e for e in self.ends if e > t1]
        if later:
            read = self._read_parallel if self.parallel else self.reader._read_until
            self.pending = self.pool.submit(read, min(later))
        return out

    def close(self):
        self._wait()
        self.pool.shutdown()
        self.workers.shutdown()
        self.reader.close()


@torch.no_grad()
def precompute_rvm_to_memmap(model, source, meta, rvm_path: Path,
                             *, chunk: int, reader_kind: str):
    assert model.foundations.rvm is not None
    device = next(model.foundations.rvm.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    H_e, W_e = H // 4, W // 4
    D = model.foundations.rvm.embed_dim
    h_e, w_e = H_e // 16, W_e // 16
    print(f"    RVM continuous prepass: T={T_all}, chunk={chunk}, "
          f"input={W_e}x{H_e}, grid={w_e}x{h_e}")
    out = np.memmap(rvm_path, dtype=np.float16, mode="w+",
                    shape=(T_all, D, h_e, w_e))
    mean = model.foundations.rvm_mean.view(1, 1, -1, 1, 1).float().cpu()
    std = model.foundations.rvm_std.view(1, 1, -1, 1, 1).float().cpu().clamp(min=1e-6)
    if reader_kind == "video":
        chunks = iter_video_frame_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    else:
        chunks = iter_frame_dir_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    state = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_next_or_none, chunks)
        while True:
            item = future.result()
            if item is None:
                break
            future = pool.submit(_next_or_none, chunks)
            t0, _model, display = item
            Tc = display.shape[0]
            # Keep Metal uploads synchronous. Asynchronous transfers produced
            # non-finite output pixels in end-to-end inference tests.
            rgb = display.unsqueeze(0).to(device, non_blocking=device.type == "cuda").float()
            rgb_rvm = F.interpolate(
                rgb.reshape(Tc, 3, H, W), size=(H_e, W_e),
                mode="bilinear", align_corners=False,
            ).reshape(1, Tc, 3, H_e, W_e).clamp(0.0, 1.0)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                feats, state = model.foundations.rvm(rgb_rvm, state=state)
            feats_cpu = (feats.float().cpu() - mean) / std
            out[t0:t0 + Tc] = feats_cpu[0].numpy().astype(np.float16)
            del rgb, rgb_rvm, feats, feats_cpu, display
    out.flush()
    del out
    empty_cache(device)


@torch.no_grad()
def precompute_foundations_to_memmap(model, source, meta, prefix: Path,
                                     *, chunk: int, reader_kind: str):
    device = next(model.foundations.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    h64, w64 = H // 64, W // 64
    C_cradio = int(getattr(model.stage1, "cradio_dim", 0) or 0)
    C_moge = int(getattr(model.stage1, "moge_dim", 0) or 0)
    if C_moge <= 0:
        raise RuntimeError("MoGe is required for this model")
    paths = {
        "moge": str(prefix) + "_moge.f32",
        "moge_shape": (T_all, C_moge, h64, w64),
    }
    moge_out = np.memmap(paths["moge"], dtype=np.float32, mode="w+",
                         shape=paths["moge_shape"])
    cradio_out = None
    if C_cradio > 0:
        paths["cradio"] = str(prefix) + "_cradio.f32"
        paths["cradio_shape"] = (T_all, C_cradio, h64, w64)
        cradio_out = np.memmap(paths["cradio"], dtype=np.float32, mode="w+",
                               shape=paths["cradio_shape"])
    print(f"    frozen VFM prepass: T={T_all}, grid={w64}x{h64}, chunk={chunk}")
    if reader_kind == "video":
        chunks = iter_video_frame_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    else:
        chunks = iter_frame_dir_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_next_or_none, chunks)
        while True:
            item = future.result()
            if item is None:
                break
            future = pool.submit(_next_or_none, chunks)
            t0, _model, display = item
            Tc = display.shape[0]
            rgb = display.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
            rgb_radio, rgb_moge = _build_foundation_inputs(rgb)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                ff = model.foundations(
                    rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                    rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
                )
            moge_out[t0:t0 + Tc] = ff["moge"].float().cpu().numpy()
            if cradio_out is not None:
                cradio_out[t0:t0 + Tc] = ff["cradio"].float().cpu().numpy()
            del rgb, rgb_radio, rgb_moge, ff, display
    moge_out.flush()
    del moge_out
    if cradio_out is not None:
        cradio_out.flush()
        del cradio_out
    empty_cache(device)
    return paths


@torch.no_grad()
def precompute_prepasses_to_memmap(model, source, meta, *, rvm_path: Path | None,
                                   foundation_prefix: Path | None, rvm_chunk: int,
                                   foundation_chunk: int, reader_kind: str,
                                   workers: int = 4, model_view_path: Path | None = None,
                                   foundation_view_path: Path | None = None,
                                   gpu_prep: bool = False):
    """precompute_rvm_to_memmap and precompute_foundations_to_memmap in one
    read of the clip: each frame is decoded once (on `workers` threads,
    without the scene-linear view these passes do not use) and fed to RVM in
    chunks of `rvm_chunk` and to the foundations in chunks of
    `foundation_chunk`, at the same chunk boundaries as the separate passes.
    Same models, inputs and chunks, so the memmaps are identical. Either
    path may be None to skip that pass. Returns the foundation paths.
    `model_view_path`: also write each frame's model view (bf16 bits, from
    the same decode) there, for _ModelViewReader; `foundation_view_path`
    (with it): the foundation view too."""
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    rvm_out = foundation_paths = moge_out = cradio_out = None
    if rvm_path is not None:
        rvm = model.foundations.rvm
        H_e, W_e = H // 4, W // 4
        print(f"    RVM continuous prepass: T={T_all}, chunk={rvm_chunk}, "
              f"input={W_e}x{H_e}, grid={W_e // 16}x{H_e // 16}")
        rvm_out = np.memmap(rvm_path, dtype=np.float16, mode="w+",
                            shape=(T_all, rvm.embed_dim, H_e // 16, W_e // 16))
        mean = model.foundations.rvm_mean.view(1, 1, -1, 1, 1).float().cpu()
        std = model.foundations.rvm_std.view(1, 1, -1, 1, 1).float().cpu().clamp(min=1e-6)
        rvm_state = None
    if foundation_prefix is not None:
        h64, w64 = H // 64, W // 64
        C_cradio = int(getattr(model.stage1, "cradio_dim", 0) or 0)
        C_moge = int(getattr(model.stage1, "moge_dim", 0) or 0)
        if C_moge <= 0:
            raise RuntimeError("MoGe is required for this model")
        foundation_paths = {"moge": str(foundation_prefix) + "_moge.f32",
                            "moge_shape": (T_all, C_moge, h64, w64)}
        moge_out = np.memmap(foundation_paths["moge"], dtype=np.float32, mode="w+",
                             shape=foundation_paths["moge_shape"])
        if C_cradio > 0:
            foundation_paths["cradio"] = str(foundation_prefix) + "_cradio.f32"
            foundation_paths["cradio_shape"] = (T_all, C_cradio, h64, w64)
            cradio_out = np.memmap(foundation_paths["cradio"], dtype=np.float32, mode="w+",
                                   shape=foundation_paths["cradio_shape"])
        print(f"    frozen VFM prepass: T={T_all}, grid={w64}x{h64}, chunk={foundation_chunk}")
    device = next(model.foundations.parameters()).device

    def shrink(display):
        d = display.unsqueeze(0).to(device, non_blocking=True)
        rvm_in = (F.interpolate(d.float(), size=(H // 4, W // 4), mode="bilinear",
                                align_corners=False) if rvm_out is not None else None)
        radio = moge = None
        if moge_out is not None:
            radio, moge = _build_foundation_inputs(d.unsqueeze(0))
        return rvm_in, radio, moge

    def run_rvm(t0, chunk):
        nonlocal rvm_state
        Tc = len(chunk)
        rgb_rvm = torch.cat([c[0] for c in chunk]).reshape(
            1, Tc, 3, H // 4, W // 4).clamp(0.0, 1.0)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            feats, rvm_state = model.foundations.rvm(rgb_rvm, state=rvm_state)
        feats_cpu = (feats.float().cpu() - mean) / std
        rvm_out[t0:t0 + Tc] = feats_cpu[0].numpy().astype(np.float16)

    def run_foundations(t0, chunk):
        Tc = len(chunk)
        rgb_radio = torch.cat([c[1] for c in chunk], dim=1)
        rgb_moge = torch.cat([c[2] for c in chunk], dim=1)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            ff = model.foundations(
                rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
            )
        moge_out[t0:t0 + Tc] = ff["moge"].float().cpu().numpy()
        if cradio_out is not None:
            cradio_out[t0:t0 + Tc] = ff["cradio"].float().cpu().numpy()

    consumers = []  # [next start, chunk, run]
    if rvm_out is not None:
        consumers.append([0, max(1, rvm_chunk), run_rvm])
    if moge_out is not None:
        consumers.append([0, max(1, foundation_chunk), run_foundations])
    model_views = foundation_views = None
    if model_view_path is not None:
        model_views = np.memmap(model_view_path, dtype=np.int16, mode="w+",
                                shape=(T_all, 3, H, W))
        if foundation_view_path is not None:
            foundation_views = np.memmap(foundation_view_path, dtype=np.int16, mode="w+",
                                         shape=(T_all, 3, H, W))
    on_device = (gpu_prep and model_views is not None and reader_kind == "frame_dir"
                 and tuple(meta["model_native_hw"]) == tuple(meta["native_hw"]))
    frame_iter = (iter_frame_dir_views_on_device(meta, device, workers=workers)
                  if on_device else
                  iter_foundation_frames(str(source), meta, reader_kind=reader_kind,
                                         workers=workers, with_model=model_views is not None))
    frames = {}
    for t, frame in frame_iter:
        if model_views is not None:
            frame, model_view = frame
            model_views[t] = model_view.view(torch.int16).cpu().numpy()
            if foundation_views is not None:
                foundation_views[t] = frame.view(torch.int16).cpu().numpy()
            del model_view
        if consumers:
            frames[t] = shrink(frame)
        del frame
        for c in consumers:
            while c[0] < T_all and (t + 1 - c[0] >= c[1] or t + 1 == T_all):
                t1 = min(c[0] + c[1], T_all)
                c[2](c[0], [frames[k] for k in range(c[0], t1)])
                c[0] = t1
        done = min(c[0] for c in consumers) if consumers else T_all
        for k in [k for k in frames if k < done]:
            del frames[k]
    for out in (rvm_out, moge_out, cradio_out, model_views, foundation_views):
        if out is not None:
            out.flush()
    del rvm_out, moge_out, cradio_out, model_views, foundation_views
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return foundation_paths


def _phase1_setup(model, *, stage1_on_gpu: bool = True):
    device = model_device(model)
    model.foundations.to(device)
    # Low-VRAM mode precomputes foundation features first, so stage 1 waits on CPU.
    if stage1_on_gpu:
        model.stage1.to(device)
    else:
        model.stage1.cpu()
    model.stage2.cpu()
    empty_cache(device)


class _HostCopies:
    """Rendered frames copied to the host without making the GPU wait.

    The default path copies each output with .cpu(): the host waits for the
    GPU to finish everything queued, copies through pageable memory, then
    quantizes and queues the frame, and the GPU idles until the next batch's
    inputs and kernels arrive. Here the renderer quantizes on the GPU, each
    output is copied into a pinned host buffer on the compute stream, and a
    batch is handed to the writers only once the next batch has been queued
    behind it, so the GPU works through the host's share. The writers get the
    same bytes in the same order; a buffer lives as long as the writer holds
    its array.
    """

    def __init__(self, writers):
        self.writers = writers
        self.pending = []

    def submit(self, rendered: dict[str, torch.Tensor]):
        host = {}
        for name, frames in rendered.items():
            buf = torch.empty(frames.shape, dtype=frames.dtype, pin_memory=True)
            buf.copy_(frames, non_blocking=True)
            host[name] = buf
        done = torch.cuda.Event()
        done.record()
        self.pending.append((done, host))

    def flush(self, keep: int = 0):
        while len(self.pending) > keep:
            done, host = self.pending.pop(0)
            done.synchronize()
            for name, buf in host.items():
                self.writers[name].write_many(buf.numpy())


def _phase2_setup(model):
    device = model_device(model)
    model.foundations.cpu()
    model.stage1.cpu()
    empty_cache(device)
    model.stage2.to(device)
    synchronize(device)


@torch.no_grad()
def hann_stage1_stream(model, source, meta, *, reader_cls, reader_kind: str,
                       chunk: int, stride: int, baton_path: Path,
                       initial_hint, initial_hint_quality: float,
                       full_hint_video_path: str | None,
                       full_hint_quality: float,
                       rvm_path: Path | None,
                       foundation_paths,
                       carry_hint_from_aux: bool,
                       carry_hint_quality: float,
                       on_baton_frames,
                       prefetch: bool = False,
                       async_baton: bool = False,
                       lean_reads: bool = False,
                       reader=None):
    """`reader`: a window reader for this clip and these windows that the
    caller already started (stage1_window_reader), used instead of a new one."""
    device = next(model.stage1.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    d_16 = model.stage1.d_16
    h16, w16 = H // 16, W // 16
    starts = window_starts(T_all, chunk, stride)
    overlap = max(0, chunk - stride)
    baton_fp16 = None
    if on_baton_frames is None:
        baton_fp16 = np.memmap(baton_path, dtype=np.float16, mode="w+",
                               shape=(T_all, d_16, h16, w16))
    live_acc = {}
    live_w = {}
    print(f"    Hann stage1: chunk={chunk}, stride={stride}, "
          f"windows={len(starts)}, baton_grid={w16}x{h16}")

    rvm_map = None
    if rvm_path is not None:
        D = model.foundations.rvm.embed_dim
        rvm_map = np.memmap(rvm_path, dtype=np.float16, mode="r",
                            shape=(T_all, D, H // 64, W // 64))
    cradio_map = moge_map = None
    if foundation_paths is not None:
        moge_map = np.memmap(foundation_paths["moge"], dtype=np.float32,
                             mode="r", shape=tuple(foundation_paths["moge_shape"]))
        if model.stage1.cradio_dim > 0:
            cradio_map = np.memmap(foundation_paths["cradio"], dtype=np.float32,
                                   mode="r", shape=tuple(foundation_paths["cradio_shape"]))

    if reader is None:
        reader = stage1_window_reader(
            reader_cls, source, meta, starts, chunk, prefetch=prefetch,
            model_only=lean_reads and foundation_paths is not None)
    hint_reader = make_hint_reader(full_hint_video_path, meta)
    carry_cache = {}
    stage2_frame_cache = {}

    def flush_before(t_limit: int):
        ready = []
        for k in sorted(k for k in live_acc if k < t_limit):
            frame = (live_acc.pop(k) / max(live_w.pop(k), 1e-6)).astype(np.float16)
            if on_baton_frames is None:
                baton_fp16[k] = frame
            else:
                model_frame = stage2_frame_cache.pop(k)
                ready.append((k, frame, model_frame))
        if ready:
            on_baton_frames(ready)

    def accumulate(weighted, w_np, t0: int, t1: int):
        for j, t_abs in enumerate(range(t0, t1)):
            if t_abs not in live_acc:
                live_acc[t_abs] = weighted[j].copy()
                live_w[t_abs] = float(w_np[j])
            else:
                live_acc[t_abs] += weighted[j]
                live_w[t_abs] += float(w_np[j])

    pipelined = async_baton and on_baton_frames is None
    pending = None
    staging = HostBuffer()
    w_dev = ([taper_weights(min(s + chunk, T_all) - s, overlap, is_first=(k == 0),
                            is_last=(k == len(starts) - 1),
                            device="cpu", dtype=torch.float32).to(device)
              for k, s in enumerate(starts)] if pipelined else None)

    def finish(job):
        done, weighted, w_np, j0, j1, j_next = job
        done.synchronize()
        accumulate(weighted.numpy(), w_np, j0, j1)
        flush_before(j_next)

    try:
        for i, t0 in enumerate(starts):
            t1 = min(t0 + chunk, T_all)
            Tc = t1 - t0
            model_chunk, display_chunk, review_chunk = reader.window(t0, t1)
            hint_cpu = None
            if not pipelined or hint_reader is not None or (initial_hint is not None and t0 == 0):
                hint_cpu = torch.zeros(1, Tc, 1, H, W, dtype=torch.bfloat16)
            hint_quality_cpu = torch.zeros(1, Tc, dtype=torch.bfloat16)
            hint_present = torch.zeros(1, Tc, dtype=torch.bool)
            if hint_reader is not None:
                hint_cpu[0] = hint_reader.window(t0, t1)
                hint_quality_cpu.fill_(float(full_hint_quality))
                hint_present.fill_(True)
            for j, t_abs in enumerate(range(t0, t1)):
                if t_abs in carry_cache:
                    if not pipelined:
                        hint_cpu[0, j] = carry_cache[t_abs]
                    hint_quality_cpu[0, j] = float(carry_hint_quality)
                    hint_present[0, j] = True
            if initial_hint is not None and t0 == 0:
                hint_cpu[0, 0] = initial_hint.to(dtype=hint_cpu.dtype)
                hint_quality_cpu[0, 0] = float(initial_hint_quality)
                hint_present[0, 0] = True
            if not bool(hint_present.any().item()):
                hint_quality_cpu.fill_(-1.0)

            if on_baton_frames is not None:
                for j, t_abs in enumerate(range(t0, t1)):
                    stage2_frame_cache[t_abs] = model_chunk[j]

            nb = device.type == "cuda"
            rgb_c = model_chunk.unsqueeze(0).to(device, non_blocking=nb)
            if hint_cpu is not None:
                hint_c = hint_cpu.to(device, non_blocking=nb)
            else:
                hint_c = torch.zeros(1, Tc, 1, H, W, dtype=torch.bfloat16, device=device)
            if pipelined:
                for j, t_abs in enumerate(range(t0, t1)):
                    if t_abs in carry_cache:
                        hint_c[0, j] = carry_cache[t_abs]
            hint_quality_c = hint_quality_cpu.to(device, non_blocking=nb)
            ff = rgb_radio = rgb_moge = None
            if foundation_paths is None:
                rgb_found = display_chunk.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
                rgb_radio, rgb_moge = _build_foundation_inputs(rgb_found)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    ff = model.foundations(
                        rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                        rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
                    )
                cradio_feat = ff["cradio"].reshape(1, Tc, -1, H // 64, W // 64)
                moge_feat = ff["moge"].reshape(1, Tc, -1, H // 64, W // 64)
            else:
                moge_feat = torch.from_numpy(np.asarray(moge_map[t0:t1]).copy())
                moge_feat = moge_feat.unsqueeze(0).to(device, non_blocking=device.type == "cuda").float()
                cradio_feat = torch.from_numpy(np.asarray(cradio_map[t0:t1]).copy())
                cradio_feat = cradio_feat.unsqueeze(0).to(device, non_blocking=device.type == "cuda").float()
            rvm_feat = None
            if rvm_map is not None:
                rvm_feat = torch.from_numpy(np.asarray(rvm_map[t0:t1]).copy())
                rvm_feat = rvm_feat.unsqueeze(0).to(device, non_blocking=nb).float()
            if getattr(model.stage1, "inference_lean", False):
                # Hand the window's inputs over (the dict holds the only
                # references) so stage 1 frees them once consumed.
                stage1_inputs = {"rgb": rgb_c, "hint": hint_c, "moge_feat": moge_feat,
                                 "cradio_feat": cradio_feat, "rvm_feat": rvm_feat}
                rgb_c = hint_c = moge_feat = cradio_feat = rvm_feat = None
                ff = rgb_radio = rgb_moge = rgb_found = None
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    s1 = model.stage1(inputs=stage1_inputs, t_stride=1.0,
                                      hint_quality=hint_quality_c)
            else:
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    s1 = model.stage1(
                        rgb_c, hint_c, moge_feat, cradio_feat=cradio_feat,
                        rvm_feat=rvm_feat, t_stride=1.0,
                        hint_quality=hint_quality_c,
                    )
            baton_gpu = s1["baton"][0].float()
            if carry_hint_from_aux:
                aux_alpha = s1["aux"][0, :, 0:1].float()
                aux_full = F.interpolate(
                    aux_alpha, size=(H, W), mode="bilinear",
                    align_corners=False).clamp(0.0, 1.0)
                aux_bf16 = (aux_full.to(torch.bfloat16) if pipelined
                            else aux_full.to("cpu", dtype=torch.bfloat16))
                next_start = starts[i + 1] if i + 1 < len(starts) else T_all
                for j, t_abs in enumerate(range(t0, t1)):
                    if not pipelined or t_abs >= next_start:
                        carry_cache[t_abs] = aux_bf16[j].clone()
                for old_t in [k for k in carry_cache if k < next_start]:
                    del carry_cache[old_t]
                del aux_alpha, aux_full, aux_bf16
            w = taper_weights(Tc, overlap, is_first=(i == 0),
                              is_last=(i == len(starts) - 1),
                              device="cpu", dtype=torch.float32)
            w_np = w.numpy()
            next_start = starts[i + 1] if i + 1 < len(starts) else T_all
            if pipelined:
                weighted_gpu = baton_gpu * w_dev[i].view(Tc, 1, 1, 1)
            else:
                weighted = (baton_gpu * w.to(device).view(Tc, 1, 1, 1)).to("cpu").numpy()
                accumulate(weighted, w_np, t0, t1)
                del weighted
            del s1, baton_gpu, ff, cradio_feat, moge_feat, rvm_feat
            del rgb_c, hint_c, hint_quality_c, hint_cpu, hint_quality_cpu
            del model_chunk, display_chunk, review_chunk
            if pipelined:
                if pending is not None:
                    finish(pending)
                weighted = staging.view(weighted_gpu.shape, weighted_gpu.dtype)
                weighted.copy_(weighted_gpu, non_blocking=True)
                done = torch.cuda.Event()
                done.record()
                pending = (done, weighted, w_np, t0, t1, next_start)
                del weighted_gpu, weighted
            else:
                flush_before(next_start)
                if device.type == "mps":
                    # Release unused Metal buffers between overlapping windows.
                    empty_cache(device)
            print(f"      window {i + 1}/{len(starts)} [{t0}:{t1}]", flush=True)
        synchronize(device)
        if pending is not None:
            finish(pending)
        flush_before(T_all + 1)
    finally:
        pending = None
        staging.release()
        late_skip = getattr(model.stage1, "_late_skip_host", None)
        if late_skip is not None:
            late_skip.release()
        reader.close()
        if hint_reader is not None:
            hint_reader.close()

    if baton_fp16 is not None:
        baton_fp16.flush()
    del baton_fp16, live_acc, live_w, rvm_map, cradio_map, moge_map
    empty_cache(device)


def stage1_window_reader(reader_cls, source, meta, starts, chunk: int, *,
                         prefetch: bool, model_only: bool = False,
                         parallel: bool = False):
    """The window reader hann_stage1_stream reads its windows with. With
    `prefetch` it starts reading the first window at once, so a caller can
    create it before the prepass to overlap the two."""
    T_all = int(meta["num_frames"])
    reader = reader_cls(str(source), meta, model_only=model_only)
    if prefetch:
        reader = _PrefetchingWindowReader(
            reader, [(t0, min(t0 + chunk, T_all)) for t0 in starts], parallel=parallel)
    return reader


def _crop_native_torch(x: torch.Tensor, meta: dict) -> torch.Tensor:
    h, w = meta["model_native_hw"]
    pad_l, _, pad_t, _ = meta["pad"]
    return x[..., pad_t:pad_t + h, pad_l:pad_l + w]


def _resize_output_torch(x: torch.Tensor, meta: dict) -> torch.Tensor:
    if tuple(meta["model_native_hw"]) == tuple(meta["output_hw"]):
        return x
    raise RuntimeError(
        "optimized runtime render path requires native model_scale=1.0; "
        "non-native scaling would need explicit Lanczos output rendering")


def _linear_adjust_inverse_torch(linear: torch.Tensor, meta: dict) -> torch.Tensor:
    gain = float(meta.get("linear_brightness", 1.0))
    contrast = float(meta.get("linear_contrast", 1.0))
    if abs(gain - 1.0) <= 1e-6 and abs(contrast - 1.0) <= 1e-6:
        return linear
    pivot = float(meta.get("linear_contrast_pivot", 0.18))
    return (linear / gain - pivot) / contrast + pivot


def _despill_average_srgb_torch(fg: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    if strength <= 0.0:
        return fg
    r = fg[:, 0:1]
    g = fg[:, 1:2]
    b = fg[:, 2:3]
    spill = (g - (r + b) * 0.5).clamp(min=0.0)
    full = torch.cat([r + spill * 0.5, g - spill, b + spill * 0.5], dim=1)
    if strength >= 1.0:
        return full
    return fg * (1.0 - strength) + full * strength


def _srgb_to_bgr_uint16_cpu(x: torch.Tensor) -> np.ndarray:
    x = x.clamp(0.0, 1.0)
    x = x[:, [2, 1, 0]].permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return (x * 65535.0).astype(np.uint16)


def _srgb_to_bgra_uint16_cpu(rgb: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    rgb = rgb.clamp(0.0, 1.0)
    alpha = alpha.clamp(0.0, 1.0)
    bgra = torch.cat([rgb[:, [2, 1, 0]], alpha], dim=1)
    bgra = bgra.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return (bgra * 65535.0).astype(np.uint16)


def _quantize_uint16_torch(x: torch.Tensor) -> torch.Tensor:
    """`(x.cpu().numpy() * 65535.0).astype(np.uint16)` on the GPU: the same
    float32 product, truncated toward zero. Identical for every float32 in
    [0, 1], -0.0 and NaN."""
    return (x * 65535.0).to(torch.uint16)


def _srgb_to_bgr_uint16_device(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return _quantize_uint16_torch(x.flip(1).permute(0, 2, 3, 1).contiguous())


def _srgb_to_bgra_uint16_device(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(0.0, 1.0)
    alpha = alpha.clamp(0.0, 1.0)
    bgra = torch.cat([rgb.flip(1), alpha], dim=1)
    return _quantize_uint16_torch(bgra.permute(0, 2, 3, 1).contiguous())


def _chw_to_bgr_f32_device(x: torch.Tensor) -> torch.Tensor:
    return x.flip(1).permute(0, 2, 3, 1).contiguous()


def _chw_to_bgra_f32_device(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    bgra = torch.cat([rgb.flip(1), alpha], dim=1)
    return bgra.permute(0, 2, 3, 1).contiguous()


def _chw_to_bgr_f32_cpu(x: torch.Tensor) -> np.ndarray:
    return (x[:, [2, 1, 0]].permute(0, 2, 3, 1)
            .contiguous().cpu().numpy().astype(np.float32, copy=False))


def _chw_to_bgra_f32_cpu(rgb: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    bgra = torch.cat([rgb[:, [2, 1, 0]], alpha], dim=1)
    return (bgra.permute(0, 2, 3, 1)
            .contiguous().cpu().numpy().astype(np.float32, copy=False))


def _render_predictions_exr_torch(meta: dict, outputs: set[str],
                                  checker: torch.Tensor,
                                  logits: torch.Tensor, *,
                                  on_device: bool = False) -> dict[str, np.ndarray]:
    """Render a stage-2 batch as unclamped scene-linear float32 NHWC arrays.

    HDR contract: `fg`, `cutout`, and `checker` color stays in decoded linear
    light with no [0, 1] clamp and no quantization, so values above 1.0 (and
    negative values the model may emit) survive into the EXR files. Alpha is a
    matte and stays clamped to [0, 1].

    `on_device`: return the same values as GPU tensors instead.
    """
    to_bgr = _chw_to_bgr_f32_device if on_device else _chw_to_bgr_f32_cpu
    to_bgra = _chw_to_bgra_f32_device if on_device else _chw_to_bgra_f32_cpu
    alpha_model = logits[:, 0:1].float()
    pa_model = _crop_native_torch(alpha_model, meta).clamp(0.0, 1.0)
    despill_strength = float(meta.get("despill_strength", 0.5))

    pa = _resize_output_torch(pa_model, meta).clamp(0.0, 1.0)
    rendered: dict[str, np.ndarray] = {}
    if "alpha" in outputs:
        alpha_nhwc = pa.permute(0, 2, 3, 1).contiguous()
        rendered["alpha"] = (alpha_nhwc if on_device else
                             alpha_nhwc.cpu().numpy().astype(np.float32, copy=False))
    if not (outputs & {"fg", "cutout", "checker"}):
        return rendered

    fg_model = logits[:, 1:4].float()
    pf_model = _crop_native_torch(fg_model, meta)
    pf_linear = _linear_adjust_inverse_torch(cu.asinh_to_linear(pf_model), meta)
    pf_linear = _resize_output_torch(pf_linear, meta)
    if "fg" in outputs:
        rendered["fg"] = to_bgr(pf_linear)
    if "cutout" in outputs:
        pf_linear_despilled = _despill_average_srgb_torch(
            pf_linear, strength=despill_strength)
        rendered["cutout"] = to_bgra(pf_linear_despilled, pa)
    if "checker" in outputs:
        checker_b = checker.to(device=pf_linear.device,
                               dtype=pf_linear.dtype).unsqueeze(0)
        bg_lin = cu.srgb_to_linear(checker_b)
        pf_despill = _despill_average_srgb_torch(
            pf_linear, strength=despill_strength)
        comp_lin = pf_despill * pa + bg_lin * (1.0 - pa)
        rendered["checker"] = to_bgr(comp_lin)
    return rendered


def _render_predictions_torch(meta: dict, outputs: set[str],
                              checker: torch.Tensor,
                              logits: torch.Tensor, *,
                              on_device: bool = False) -> dict[str, np.ndarray]:
    """Render a stage-2 batch on GPU, returning uint16 BGR/BGRA NHWC arrays.

    This removes the previous per-frame numpy HDR decode, sRGB encode,
    composition, channel shuffle, and uint8 conversion hot path.

    `on_device`: return the same values as GPU tensors instead.
    """
    to_bgr = _srgb_to_bgr_uint16_device if on_device else _srgb_to_bgr_uint16_cpu
    to_bgra = _srgb_to_bgra_uint16_device if on_device else _srgb_to_bgra_uint16_cpu
    alpha_model = logits[:, 0:1].float()
    pa_model = _crop_native_torch(alpha_model, meta).clamp(0.0, 1.0)
    despill_strength = float(meta.get("despill_strength", 0.5))

    pa = _resize_output_torch(pa_model, meta).clamp(0.0, 1.0)
    rendered: dict[str, np.ndarray] = {}
    if "alpha" in outputs:
        rendered["alpha"] = to_bgr(pa.expand(-1, 3, -1, -1))
    if not (outputs & {"fg", "cutout", "checker"}):
        return rendered

    fg_model = logits[:, 1:4].float()
    pf_model = _crop_native_torch(fg_model, meta)
    pf_linear = _linear_adjust_inverse_torch(cu.asinh_to_linear(pf_model), meta)
    pf_linear_pos = pf_linear.clamp(min=0.0)
    pf = _resize_output_torch(cu.linear_to_srgb(pf_linear_pos), meta).clamp(0.0, 1.0)
    if "fg" in outputs:
        rendered["fg"] = to_bgr(pf)
    if "cutout" in outputs:
        pf_linear_despilled = _despill_average_srgb_torch(pf_linear_pos, strength=despill_strength)
        if meta.get("cutout_linear", False):
            pf_linear_despilled_resized = _resize_output_torch(pf_linear_despilled, meta).clamp(0.0, 1.0)
            rendered["cutout"] = to_bgra(pf_linear_despilled_resized, pa)
        else:
            pf_despill = _despill_average_srgb_torch(pf, strength=despill_strength)
            rendered["cutout"] = to_bgra(pf_despill, pa)
    if "checker" in outputs:
        checker_b = checker.to(device=pf.device, dtype=pf.dtype).unsqueeze(0)
        bg_lin = cu.srgb_to_linear(checker_b)
        alpha = pa
        pf_despill = _despill_average_srgb_torch(pf, strength=despill_strength)
        comp_despill_lin = (
            cu.srgb_to_linear(pf_despill) * alpha + bg_lin * (1.0 - alpha))
        comp_despill = cu.linear_to_srgb(
            comp_despill_lin.clamp(min=0.0)).clamp(0.0, 1.0)
        rendered["checker"] = to_bgr(comp_despill)
    return rendered


def run_inference(model, input_path: str, output_dir: str, cfg: InferenceConfig,
                  writer_cfg: WriterConfig, *, initial_hint_path: str | None,
                  hint_video_path: str | None):
    """Run inference; with cfg.auto_vram on CUDA, an OOM restarts with a cheaper profile."""
    kwargs = {"initial_hint_path": initial_hint_path, "hint_video_path": hint_video_path}
    dev = model_device(model)
    while True:
        try:
            return _run_inference_once(model, input_path, output_dir, cfg, writer_cfg, **kwargs)
        except torch.OutOfMemoryError:
            if not cfg.auto_vram or (cfg.hann_chunk, cfg.low_vram) == (1, True):
                raise
            if dev.type != "cuda":
                raise
            cfg.vram_ceiling_gb = cfg.vram_estimate_gb
            print(f"  [AUTO VRAM] Out of memory at hann_chunk={cfg.hann_chunk}, "
                  f"low_vram={cfg.low_vram}; retrying with a smaller profile.")
        # Past the except block, so the traceback no longer pins GPU tensors.
        gc.collect()
        torch.cuda.empty_cache()


def _run_inference_once(model, input_path: str, output_dir: str, cfg: InferenceConfig,
                        writer_cfg: WriterConfig, *, initial_hint_path: str | None,
                        hint_video_path: str | None):
    device = model_device(model)
    input_path = str(input_path)
    output_dir = str(output_dir)
    cfg.temp_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    is_dir = os.path.isdir(input_path)
    reader_kind = "frame_dir" if is_dir else "video"
    reader_cls = SequentialFrameDirWindowReader if is_dir else SequentialVideoWindowReader
    if is_dir:
        meta = probe_frame_dir_meta(
            input_path, num_frames=cfg.num_frames,
            shape_quantum=cfg.shape_quantum, fps=cfg.frame_dir_fps,
            exr_linear=cfg.frame_dir_linear, model_scale=cfg.model_scale)
        if (not cfg.frame_dir_linear
                and meta["frame_paths"]
                and meta["frame_paths"][0].lower().endswith(".exr")):
            print("  [WARNING] EXR input frames are being treated as "
                  "display-coded and clipped to [0, 1]. For scene-linear HDR "
                  "EXR plates, pass --frame_dir_linear.")
    else:
        meta = probe_video_meta(
            input_path, num_frames=cfg.num_frames,
            shape_quantum=cfg.shape_quantum, native_resolution=True,
            start_mode=cfg.start_mode, seed=cfg.seed,
            model_scale=cfg.model_scale,
            linear_brightness=cfg.linear_brightness,
            linear_contrast=cfg.linear_contrast,
            linear_contrast_pivot=cfg.linear_contrast_pivot)
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    if cfg.auto_vram:
        apply_auto_vram(cfg, (H, W), T_all, device)
    set_region_index_cache(cfg.cache_region_indices)
    H_out, W_out = meta["output_hw"]
    meta["cutout_linear"] = cfg.cutout_linear
    meta["despill_strength"] = cfg.despill_strength
    fps = float(meta["native_fps"])
    clip_name = raw_clip_name(input_path)
    checker = make_checker(H_out, W_out)
    checker_t = torch.from_numpy(checker)
    writers = {}
    suffix = {
        "alpha": "alpha",
        "fg": "fg",
        "checker": f"checker_comp_despill{int(cfg.despill_strength * 100):03d}",
        "cutout": "cutout",
    }
    try:
        use_exr = cfg.output_format == "exr"
        if cfg.output_format not in ("video", "exr"):
            raise ValueError(f"unsupported output format: {cfg.output_format}")
        for name in sorted(cfg.outputs):
            if name not in suffix:
                raise ValueError(f"unsupported output choice: {name}")
            if use_exr:
                channels = 4 if name == "cutout" else (1 if name == "alpha" else 3)
                writers[name] = ExrSequenceWriter(
                    os.path.join(output_dir, f"{clip_name}_{suffix[name]}_exr"),
                    fps, (W_out, H_out), channels=channels,
                    dtype=cfg.exr_dtype, frame_name=suffix[name],
                    workers=cfg.exr_workers)
            else:
                ext = ".mov" if name == "cutout" else ".mp4"
                channels = 4 if name == "cutout" else 3
                color_trc = "linear" if (name == "cutout" and cfg.cutout_linear) else "iec61966-2-1"
                writers[name] = ThreadedVideoWriter(
                    os.path.join(output_dir, f"{clip_name}_{suffix[name]}{ext}"),
                    fps, (W_out, H_out), writer_cfg, channels=channels, color_trc=color_trc)

        # Write settings log file in the output folder
        settings_path = os.path.join(output_dir, f"{clip_name}_settings.txt")
        try:
            with open(settings_path, "w") as sf:
                sf.write(f"CorridorKey v2 Inference Settings Log\n")
                sf.write(f"====================================\n")
                sf.write(f"Input: {input_path}\n")
                sf.write(f"Output Directory: {output_dir}\n")
                sf.write(f"Resolution: {W_out}x{H_out}\n")
                sf.write(f"FPS: {fps}\n")
                sf.write(f"Outputs: {', '.join(sorted(cfg.outputs))}\n")
                sf.write(f"Output Format: {cfg.output_format}\n")
                if cfg.output_format == "exr":
                    sf.write(f"EXR Pixel Type: {cfg.exr_dtype}\n")
                sf.write(f"Hint First Frame: {initial_hint_path}\n")
                sf.write(f"Hint Video: {hint_video_path}\n")
                sf.write(f"Hint Quality (Trust): {cfg.hint_quality}\n")
                sf.write(f"Carry Hint: {cfg.carry_hint}\n")
                sf.write(f"Carry Hint Quality: {cfg.carry_hint_quality}\n")
                sf.write(f"Despill Strength: {cfg.despill_strength}\n")
                sf.write(f"Cutout Linear: {cfg.cutout_linear}\n")
                sf.write(f"Frame Dir Linear: {cfg.frame_dir_linear}\n")
                sf.write(f"Hann Chunk: {cfg.hann_chunk}\n")
                sf.write(f"Hann Stride: {cfg.hann_stride}\n")
                sf.write(f"Low VRAM: {cfg.low_vram}\n")
                sf.write(f"Model Scale: {cfg.model_scale}\n")
                sf.write(f"Num Frames: {cfg.num_frames}\n")
                sf.write(f"Start Mode: {cfg.start_mode}\n")
                sf.write(f"Seed: {cfg.seed}\n")
                sf.write(f"Linear Brightness: {cfg.linear_brightness}\n")
                sf.write(f"Linear Contrast: {cfg.linear_contrast}\n")
                sf.write(f"Linear Contrast Pivot: {cfg.linear_contrast_pivot}\n")
        except Exception as e:
            print(f"[WARNING] Failed to write settings file: {e}")

        initial_hint = (
            load_initial_hint_frame(initial_hint_path, meta)
            if none_path(initial_hint_path) is not None else None
        )
        _phase1_setup(model, stage1_on_gpu=not cfg.low_vram)
        if cfg.low_vram:
            configure_low_vram_inference(model, cfg)
        configure_lean_activations(model, cfg.lean_activations)
        rvm_path = cfg.temp_dir / f"{clip_name}_rvm_feat.f16"
        baton_path = cfg.temp_dir / f"{clip_name}_baton.f16"
        foundation_prefix = cfg.temp_dir / f"{clip_name}_foundation"
        for p in (
            rvm_path, baton_path, Path(str(foundation_prefix) + "_cradio.f32"),
            Path(str(foundation_prefix) + "_moge.f32"),
        ):
            safe_remove(p)
        rvm_feat_path = rvm_path if model.foundations.rvm is not None else None
        foundation_paths = None
        stage1_reader = None
        model_view_path = foundation_view_path = None
        if cfg.decode_once and cfg.prefetch_frames:
            need = T_all * 3 * H * W * 2 * (1 if cfg.low_vram else 2)
            free = shutil.disk_usage(cfg.temp_dir).free
            # On MPS, memmaps live in unified memory and compete with GPU
            # allocations for the same physical pool. Only use decode_once
            # if the memmap fits within the MPS headroom left after inference.
            if device.type == "mps":
                inference_gb = estimate_peak_vram_gb(
                    H * W / 1e6, min(int(cfg.hann_chunk), T_all),
                    cfg.low_vram, _VRAM_PEAK_GB_OPTIMIZED if cfg.auto_vram_optimized else _VRAM_PEAK_GB,
                ) * _MPS_ESTIMATE_SCALE
                mps_headroom = memory_budget_gb(device)
                mps_ok = (need / 2**30) < max(0.0, mps_headroom - inference_gb)
            else:
                mps_ok = True
            if need <= free // 2 and mps_ok:
                model_view_path = cfg.temp_dir / f"{clip_name}_model.bf16"
                if not cfg.low_vram:
                    foundation_view_path = cfg.temp_dir / f"{clip_name}_foundation_view.bf16"
            else:
                print(f"  decode_once: the clip's model views ({need / 2**30:.1f} GB) do not "
                      f"fit in half the temp folder's free space; reading frames as usual")
        if cfg.prefetch_frames and cfg.read_ahead and model_view_path is None:
            # Stage 1's first window is read (on the worker threads) while
            # the prepass runs; the low-VRAM prepass makes the foundation
            # features, so stage 1 then needs only the model view.
            stage1_reader = stage1_window_reader(
                reader_cls, input_path, meta,
                window_starts(T_all, int(cfg.hann_chunk), int(cfg.hann_stride)),
                int(cfg.hann_chunk), prefetch=True,
                model_only=cfg.lean_reads and cfg.low_vram, parallel=True)
        if cfg.prefetch_frames:
            # One read of the clip feeds both prepasses (same chunks, same memmaps).
            foundation_paths = precompute_prepasses_to_memmap(
                model, input_path, meta, rvm_path=rvm_feat_path,
                foundation_prefix=foundation_prefix if cfg.low_vram else None,
                rvm_chunk=max(1, min(int(cfg.hann_chunk), 8)),
                foundation_chunk=max(1, int(cfg.low_vram_foundation_chunk)),
                reader_kind=reader_kind, workers=max(1, int(cfg.read_workers)),
                model_view_path=model_view_path, foundation_view_path=foundation_view_path,
                gpu_prep=cfg.gpu_prep)
            if model_view_path is not None:
                stage1_reader = _ModelViewReader(model_view_path, meta,
                                                 foundation_path=foundation_view_path)
        else:
            if model.foundations.rvm is not None:
                precompute_rvm_to_memmap(
                    model, input_path, meta, rvm_path,
                    chunk=max(1, min(int(cfg.hann_chunk), 8)),
                    reader_kind=reader_kind)
            if cfg.low_vram:
                foundation_paths = precompute_foundations_to_memmap(
                    model, input_path, meta, foundation_prefix,
                    chunk=max(1, int(cfg.low_vram_foundation_chunk)),
                    reader_kind=reader_kind)
        if cfg.low_vram:
            model.foundations.cpu()
            model.stage2.cpu()
            model.stage1.to(device)
            empty_cache(device)
        else:
            model.stage2.to(device)
            model.stage2.eval()
            empty_cache(device)

        if cfg.activation_arena:
            estimate = estimate_peak_vram_gb(H * W / 1e6, min(int(cfg.hann_chunk), T_all),
                                             cfg.low_vram, _VRAM_PEAK_GB_OPTIMIZED)
            arena = reserve_activation_arena(
                int(_ARENA_SCALE[cfg.low_vram]
                    * (estimate * 2**30 - torch.cuda.memory_allocated())))
            if arena:
                print(f"  Activation arena: {arena / 2**30:.1f} GB reserved as one block")

        stage2_batch = max(1, int(cfg.stage2_batch))
        host_copies = _HostCopies(writers) if cfg.pinned_io else None

        def write_prediction_batches(items):
            device = next(model.stage2.parameters()).device
            for offset in range(0, len(items), stage2_batch):
                sub = items[offset:offset + stage2_batch]
                rgb_t = torch.stack([item[2] for item in sub], dim=0)
                if host_copies is not None:
                    rgb_t = rgb_t.pin_memory()
                rgb_t = rgb_t.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
                baton_np = np.stack([item[1] for item in sub], axis=0)
                baton_t = torch.from_numpy(baton_np).unsqueeze(0)
                if host_copies is not None:
                    baton_t = baton_t.pin_memory()
                baton_t = baton_t.to(device, non_blocking=device.type == "cuda").float()
                with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
                    logits = model.stage2(rgb_t, baton_t)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                render_fn = (_render_predictions_exr_torch if use_exr
                             else _render_predictions_torch)
                rendered = render_fn(meta, cfg.outputs, checker_t, logits[0],
                                     on_device=host_copies is not None)
                if host_copies is not None:
                    host_copies.submit(rendered)
                    host_copies.flush(keep=1)
                else:
                    for name, frames in rendered.items():
                        writers[name].write_many(frames)
                del logits, baton_t, rgb_t, baton_np, rendered

        hann_stage1_stream(
            model, input_path, meta, reader_cls=reader_cls,
            reader_kind=reader_kind, chunk=cfg.hann_chunk, stride=cfg.hann_stride,
            baton_path=baton_path, initial_hint=initial_hint,
            initial_hint_quality=cfg.hint_quality,
            full_hint_video_path=hint_video_path,
            full_hint_quality=cfg.hint_quality,
            rvm_path=rvm_feat_path, foundation_paths=foundation_paths,
            carry_hint_from_aux=cfg.carry_hint,
            carry_hint_quality=cfg.carry_hint_quality,
            on_baton_frames=None if cfg.low_vram else write_prediction_batches,
            prefetch=cfg.prefetch_frames, async_baton=cfg.async_baton,
            lean_reads=cfg.lean_reads, reader=stage1_reader)
        if host_copies is not None:
            host_copies.flush()

        if cfg.low_vram:
            _phase2_setup(model)
            baton = np.memmap(
                baton_path, dtype=np.float16, mode="r",
                shape=(T_all, model.stage1.d_16, H // 16, W // 16))
            reader = (_ModelViewReader(model_view_path, meta) if model_view_path is not None
                      else reader_cls(input_path, meta, model_only=cfg.lean_reads))
            # read_ahead: frames come off the reader eight (a multiple of the
            # batch) at a time, read on the worker threads while the GPU runs
            # the previous eight; the batches are the same.
            read_chunk = stage2_batch
            if cfg.prefetch_frames and cfg.read_ahead:
                read_chunk = stage2_batch * max(1, -(-8 // stage2_batch))
            if cfg.prefetch_frames:
                reader = _PrefetchingWindowReader(
                    reader, [(t0, min(T_all, t0 + read_chunk))
                             for t0 in range(0, T_all, read_chunk)],
                    parallel=cfg.read_ahead)
            try:
                for c0 in range(0, T_all, read_chunk):
                    c1 = min(T_all, c0 + read_chunk)
                    model_chunk, _display_chunk, _review_chunk = reader.window(c0, c1)
                    for t0 in range(c0, c1, stage2_batch):
                        t1 = min(c1, t0 + stage2_batch)
                        items = [
                            (t, np.asarray(baton[t]).copy(), model_chunk[t - c0])
                            for t in range(t0, t1)
                        ]
                        write_prediction_batches(items)
                        del items
                    del model_chunk, _display_chunk, _review_chunk
                if host_copies is not None:
                    host_copies.flush()
            finally:
                reader.close()
                del baton
    finally:
        for writer in writers.values():
            writer.release()
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_rvm_feat.f16")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_baton.f16")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_foundation_cradio.f32")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_foundation_moge.f32")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_model.bf16")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_foundation_view.bf16")
        empty_cache(device)
    print(f"  wrote {len(writers)} output video(s) for {clip_name} "
          f"({T_all} frames @ {W_out}x{H_out})")
