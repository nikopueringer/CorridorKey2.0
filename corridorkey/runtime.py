from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# CorridorKey's MPS path is GPU-only. Unsupported operators must fail loudly
# instead of silently crossing to the CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
import torch.nn.functional as F

from . import color_utils as cu
from .model import CorridorKeyV3
from .video_io import (
    ExrSequenceWriter,
    ThreadedVideoWriter,
    WriterConfig,
    apply_linear_adjust,
    frame_paths,
    has_linear_adjust,
    iter_frame_dir_chunks,
    iter_video_frame_chunks,
    load_initial_hint_frame,
    make_checker,
    make_hint_reader,
    none_path,
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
_VRAM_MARGIN_GB = 0.3
# MPS currently materializes attention score buffers that CUDA SDPA avoids.
# Calibrated conservatively from the 61-frame 1920x1024 reference run:
# 19.2 GB CUDA-model estimate versus approximately 52 GB MPS peak.
_MPS_PEAK_SCALE = 2.75


class _InlineProgress:
    """Dependency-free progress display that only rewrites the current line."""

    def __init__(self, *, total: int, desc: str, unit: str):
        self.total = max(0, int(total))
        self.desc = desc.strip()
        self.unit = unit
        self.n = 0
        self.started = time.perf_counter()
        self.last_rendered = 0.0
        self.detail = ""
        self.visible = False

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return (f"{hours:d}:{minutes:02d}:{seconds:02d}"
                if hours else f"{minutes:02d}:{seconds:02d}")

    def _render(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_rendered < 0.2:
            return
        elapsed = now - self.started
        fraction = min(1.0, self.n / self.total) if self.total else 1.0
        rate = self.n / elapsed if elapsed > 0 else 0.0
        remaining = ((self.total - self.n) / rate
                     if rate > 0 and self.total > self.n else 0.0)
        message = (
            f"{self.desc}: {fraction * 100:5.1f}% "
            f"({self.n}/{self.total} {self.unit}) | "
            f"elapsed {self._duration(elapsed)}"
        )
        if self.total > self.n and rate > 0:
            message += f" | ETA {self._duration(remaining)}"
        if self.detail:
            message += f" | {self.detail}"
        try:
            width = os.get_terminal_size(sys.stdout.fileno()).columns
            message = message[:max(1, width - 1)]
        except (OSError, ValueError):
            pass
        # CR returns to column zero; EL2 clears the entire current line. Unlike
        # tqdm's nested cursor movement, this cannot create additional rows.
        sys.stdout.write(f"\r\033[2K{message}")
        sys.stdout.flush()
        self.last_rendered = now
        self.visible = True

    def update(self, amount: int = 1) -> None:
        self.n = min(self.total, self.n + int(amount))
        self._render(force=self.n >= self.total)

    def set_postfix_str(self, detail: str) -> None:
        self.detail = str(detail)
        self._render()

    def close(self) -> None:
        if self.visible:
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()
            self.visible = False


def _inline_progress(*, total: int, desc: str, unit: str):
    return _InlineProgress(total=total, desc=desc, unit=unit)


def select_accelerator(requested: str = "auto") -> torch.device:
    """Select a GPU backend. CPU execution and MPS CPU fallback are forbidden."""
    requested = str(requested).lower()
    if requested not in ("auto", "cuda", "mps"):
        raise ValueError(f"unsupported accelerator {requested!r}")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in ("auto", "mps") and torch.backends.mps.is_available():
        # Setting this after torch import still documents and enforces the
        # policy for subprocesses; PyTorch's default is already no fallback.
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        return torch.device("mps")
    if requested == "cuda":
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available.")
    if requested == "mps":
        built = torch.backends.mps.is_built()
        raise RuntimeError(
            "MPS was requested, but it is unavailable "
            f"(PyTorch built with MPS support: {built}).")
    raise RuntimeError("CorridorKey requires a CUDA or MPS GPU; CPU fallback is disabled.")


def _inference_dtype(device: torch.device) -> torch.dtype:
    # Match the trained CUDA runtime's dynamic range. Current MPS supports
    # bf16 autocast, which avoids fp16 overflow on scene-linear HDR plates.
    return torch.bfloat16


def _autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=_inference_dtype(device))


def _empty_accelerator_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _interp(x: float, points) -> float:
    """Piecewise-linear through (x, y) points, extrapolating past the ends."""
    points = list(points)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1 or (x1, y1) == points[-1]:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[0][1]


def estimate_peak_vram_gb(megapixels: float, frames: int, low_vram: bool) -> float:
    """Estimated peak GPU memory for a Hann window of `frames` frames."""
    chunks, rows = _VRAM_PEAK_GB[low_vram]
    frames = max(frames, chunks[0])
    per_chunk = [
        max(_interp(megapixels, [(mp, peaks[i]) for mp, peaks in rows]), rows[0][1][i])
        for i in range(len(chunks))
    ]
    return _interp(frames, zip(chunks, per_chunk))


def select_vram_profile(shape_hw, num_frames: int, budget_gb: float):
    """Largest Hann window that fits `budget_gb`, as (chunk, stride, low_vram, estimate).

    Bigger windows give better temporal consistency, and low-VRAM mode is
    exact, so a larger low-VRAM window is preferred over a smaller normal one.
    Falls back to a single-frame window when nothing is estimated to fit.
    """
    megapixels = shape_hw[0] * shape_hw[1] / 1e6
    for chunk in reversed(_VRAM_PEAK_GB[True][0]):
        for low_vram in (False, True):
            if chunk not in _VRAM_PEAK_GB[low_vram][0]:
                continue
            estimate = estimate_peak_vram_gb(megapixels, min(chunk, num_frames), low_vram)
            if estimate * _VRAM_MARGIN_SCALE + _VRAM_MARGIN_GB <= budget_gb:
                return chunk, max(1, chunk // 2), low_vram, estimate
    return 1, 1, True, estimate


def apply_auto_vram(cfg: InferenceConfig, shape_hw, num_frames: int,
                    device: torch.device) -> None:
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        # Memory this process already holds (the loaded model) is part of the budget.
        budget_bytes = free_bytes + torch.cuda.memory_reserved(device)
    elif device.type == "mps":
        recommended = getattr(torch.mps, "recommended_max_memory", None)
        if recommended is None:
            raise RuntimeError(
                "This PyTorch build cannot query the recommended MPS memory limit; "
                "pass --low_vram and an explicit --hann_chunk.")
        budget_bytes = recommended()
    else:
        raise RuntimeError(f"unsupported inference device: {device}")
    budget_gb = budget_bytes / 2**30
    profile_budget_gb = (
        budget_gb / _MPS_PEAK_SCALE if device.type == "mps" else budget_gb)
    chunk, stride, low_vram, estimate = select_vram_profile(
        shape_hw, num_frames, profile_budget_gb)
    displayed_estimate = estimate * (_MPS_PEAK_SCALE if device.type == "mps" else 1.0)
    cfg.hann_chunk, cfg.hann_stride, cfg.low_vram = chunk, stride, low_vram
    print(f"  Auto VRAM: {budget_gb:.1f} GB available at {shape_hw[1]}x{shape_hw[0]} -> "
          f"hann_chunk={chunk}, hann_stride={stride}, low_vram={low_vram} "
          f"(estimated peak {displayed_estimate:.1f} GB"
          f"{' on MPS' if device.type == 'mps' else ''})")
    if displayed_estimate * _VRAM_MARGIN_SCALE + _VRAM_MARGIN_GB > budget_gb:
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


def load_model(paths: RuntimePaths, device: str | torch.device = "cuda") -> CorridorKeyV3:
    device = torch.device(device)
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
    model = CorridorKeyV3(**cfg)
    model._accelerator_device = device
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    train_missing = [k for k in missing if not k.startswith("foundations.")]
    train_unexpected = [k for k in unexpected if not k.startswith("foundations.")]
    if train_missing or train_unexpected:
        raise RuntimeError(
            "runtime checkpoint did not match model: "
            f"missing={train_missing[:8]} unexpected={train_unexpected[:8]}")
    print(f"  loaded weights; omitted frozen foundation tensors={len(missing)}")
    model.eval()
    del ckpt
    return model


def configure_low_vram_inference(model, cfg: InferenceConfig,
                                 device: torch.device | None = None):
    device = torch.device(device) if device is not None else model._accelerator_device
    # CUDA SDPA does not materialize the score matrix, but current MPS SDPA
    # does. A batch of 16 temporal windows at T=61 and 16x16 spatial tokens
    # requests ~125 GB, while processing one window at a time requests ~7.8 GB.
    window_batch = max(0, int(cfg.low_vram_swin_window_batch))
    if device.type == "mps":
        window_batch = 1
    model.stage1.low_vram_enc_frame_chunk = max(0, int(cfg.low_vram_enc_frame_chunk))
    for module in model.modules():
        name = module.__class__.__name__
        if name in ("SwiGLU", "FusionLinSwiglu"):
            module.inference_chunk_tokens = max(0, int(cfg.low_vram_mlp_chunk_tokens))
        if name == "SwinAttention3D":
            module.inference_window_batch = window_batch
        if name == "CSwinAttention3D":
            module.inference_strip_batch = window_batch
        if name in ("FullSpatialAttention2D", "FullAttention3D"):
            module.inference_query_chunk_tokens = max(
                0, int(cfg.low_vram_full_attn_query_chunk_tokens))


def _area_resize_mps(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """MPS implementation of PyTorch's adaptive-average `area` resize.

    MPS adaptive_avg_pool2d currently rejects non-divisible input/output
    dimensions.  Constructing the same integer pooling bins as two separable
    matrix multiplies keeps this operation entirely on the GPU and preserves
    area-mode semantics.
    """
    out_h, out_w = size
    in_h, in_w = x.shape[-2:]

    def weights(in_size: int, out_size: int) -> torch.Tensor:
        out_idx = torch.arange(out_size, device=x.device)
        # Integer arithmetic avoids boundary drift when a rational bin edge is
        # exactly integral (for example 7 * 128 / 28 == 32).
        starts = torch.div(out_idx * in_size, out_size, rounding_mode="floor")
        ends = torch.div(
            (out_idx + 1) * in_size + out_size - 1,
            out_size, rounding_mode="floor")
        in_idx = torch.arange(in_size, device=x.device).unsqueeze(0)
        included = (in_idx >= starts.unsqueeze(1)) & (in_idx < ends.unsqueeze(1))
        return included.to(x.dtype) / (ends - starts).to(x.dtype).unsqueeze(1)

    weight_h = weights(in_h, out_h)
    weight_w = weights(in_w, out_w)
    y = torch.einsum("oh,nchw->ncow", weight_h, x)
    return torch.einsum("pw,ncow->ncop", weight_w, y)


def _area_resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if (x.device.type == "mps"
            and (x.shape[-2] % size[0] or x.shape[-1] % size[1])):
        return _area_resize_mps(x, size)
    return F.interpolate(x, size=size, mode="area")


def _build_foundation_inputs(rgb_bt: torch.Tensor):
    B, T, _, H, W = rgb_bt.shape
    flat = rgb_bt.reshape(B * T, 3, H, W)
    rgb_radio = _area_resize(flat, (H // 4, W // 4))
    H_moge, W_moge = (H * 7) // 32, (W * 7) // 32
    rgb_moge = _area_resize(flat, (H_moge, W_moge))
    return (
        rgb_radio.reshape(B, T, 3, H // 4, W // 4),
        rgb_moge.reshape(B, T, 3, H_moge, W_moge),
    )


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


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
    progress = _inline_progress(
        total=T_all, desc="    RVM features", unit="frame")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_next_or_none, chunks)
        while True:
            item = future.result()
            if item is None:
                break
            future = pool.submit(_next_or_none, chunks)
            t0, _model, display = item
            Tc = display.shape[0]
            rgb = display.unsqueeze(0).to(device, non_blocking=True).float()
            rgb_rvm = F.interpolate(
                rgb.reshape(Tc, 3, H, W), size=(H_e, W_e),
                mode="bilinear", align_corners=False,
            ).reshape(1, Tc, 3, H_e, W_e).clamp(0.0, 1.0)
            rgb_rvm = rgb_rvm.to(dtype=_inference_dtype(device))
            with _autocast(device):
                feats, state = model.foundations.rvm(rgb_rvm, state=state)
            feats_cpu = (feats.float().cpu() - mean) / std
            out[t0:t0 + Tc] = feats_cpu[0].numpy().astype(np.float16)
            progress.update(Tc)
            del rgb, rgb_rvm, feats, feats_cpu, display
    progress.close()
    out.flush()
    del out
    _empty_accelerator_cache(device)


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
    progress = _inline_progress(
        total=T_all, desc="    Foundation features", unit="frame")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_next_or_none, chunks)
        while True:
            item = future.result()
            if item is None:
                break
            future = pool.submit(_next_or_none, chunks)
            t0, _model, display = item
            Tc = display.shape[0]
            rgb = display.unsqueeze(0).to(
                device, dtype=_inference_dtype(device), non_blocking=True)
            rgb_radio, rgb_moge = _build_foundation_inputs(rgb)
            with _autocast(device):
                ff = model.foundations(
                    rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                    rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
                )
            moge_out[t0:t0 + Tc] = ff["moge"].float().cpu().numpy()
            if cradio_out is not None:
                cradio_out[t0:t0 + Tc] = ff["cradio"].float().cpu().numpy()
            progress.update(Tc)
            del rgb, rgb_radio, rgb_moge, ff, display
    progress.close()
    moge_out.flush()
    del moge_out
    if cradio_out is not None:
        cradio_out.flush()
        del cradio_out
    _empty_accelerator_cache(device)
    return paths


def _phase1_setup(model, *, stage1_on_gpu: bool = True):
    device = model._accelerator_device
    model.foundations.to(device)
    # Low-VRAM mode precomputes foundation features first, so stage 1 waits on CPU.
    if stage1_on_gpu:
        model.stage1.to(device)
    else:
        model.stage1.cpu()
    model.stage2.cpu()
    _empty_accelerator_cache(device)


def _phase2_setup(model):
    device = model._accelerator_device
    model.foundations.cpu()
    model.stage1.cpu()
    _empty_accelerator_cache(device)
    model.stage2.to(device)
    _synchronize(device)


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
                       on_baton_frames):
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

    tracked_blocks = list(model.stage1.enc_16) + list(model.stage1.body_64)
    tracked_blocks += list(model.stage1.mid_16) + list(model.stage1.dec_16)
    enc_blocks = len(model.stage1.enc_16)
    other_blocks = len(tracked_blocks) - enc_blocks
    frame_chunk = int(getattr(model.stage1, "low_vram_enc_frame_chunk", 0) or 0)

    def block_calls(tc: int) -> int:
        enc_passes = (
            (tc + frame_chunk - 1) // frame_chunk
            if frame_chunk > 0 and tc > frame_chunk else 1)
        return enc_blocks * enc_passes + other_blocks

    stage1_progress = _inline_progress(
        total=sum(block_calls(min(t0 + chunk, T_all) - t0) for t0 in starts),
        desc="    Stage 1 blocks", unit="block")
    def update_stage1_progress(_module, _inputs, _output):
        stage1_progress.update(1)
        # Forward hooks replace a module's output when they return a value.
        # tqdm.update() returns a bool on refresh, so return None explicitly.
        return None

    hook_handles = [
        block.register_forward_hook(update_stage1_progress)
        for block in tracked_blocks
    ]

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

    reader = reader_cls(str(source), meta)
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

    try:
        for i, t0 in enumerate(starts):
            t1 = min(t0 + chunk, T_all)
            Tc = t1 - t0
            stage1_progress.set_postfix_str(
                f"window {i + 1}/{len(starts)} frames {t0}:{t1}")
            model_chunk, display_chunk, review_chunk = reader.window(t0, t1)
            hint_cpu = torch.zeros(1, Tc, 1, H, W, dtype=torch.bfloat16)
            hint_quality_cpu = torch.zeros(1, Tc, dtype=torch.bfloat16)
            hint_present = torch.zeros(1, Tc, dtype=torch.bool)
            if hint_reader is not None:
                hint_cpu[0] = hint_reader.window(t0, t1)
                hint_quality_cpu.fill_(float(full_hint_quality))
                hint_present.fill_(True)
            for j, t_abs in enumerate(range(t0, t1)):
                if t_abs in carry_cache:
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

            compute_dtype = _inference_dtype(device)
            rgb_c = model_chunk.unsqueeze(0).to(
                device, dtype=compute_dtype, non_blocking=True)
            hint_c = hint_cpu.to(device, dtype=compute_dtype, non_blocking=True)
            hint_quality_c = hint_quality_cpu.to(
                device, dtype=compute_dtype, non_blocking=True)
            ff = rgb_radio = rgb_moge = None
            if foundation_paths is None:
                rgb_found = display_chunk.unsqueeze(0).to(
                    device, dtype=compute_dtype, non_blocking=True)
                rgb_radio, rgb_moge = _build_foundation_inputs(rgb_found)
                with _autocast(device):
                    ff = model.foundations(
                        rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                        rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
                    )
                cradio_feat = ff["cradio"].reshape(1, Tc, -1, H // 64, W // 64)
                moge_feat = ff["moge"].reshape(1, Tc, -1, H // 64, W // 64)
            else:
                moge_feat = torch.from_numpy(np.asarray(moge_map[t0:t1]).copy())
                moge_feat = moge_feat.unsqueeze(0).to(device, non_blocking=True).float()
                cradio_feat = torch.from_numpy(np.asarray(cradio_map[t0:t1]).copy())
                cradio_feat = cradio_feat.unsqueeze(0).to(device, non_blocking=True).float()
            rvm_feat = None
            if rvm_map is not None:
                rvm_feat = torch.from_numpy(np.asarray(rvm_map[t0:t1]).copy())
                rvm_feat = rvm_feat.unsqueeze(0).to(device, non_blocking=True).float()
            with _autocast(device):
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
                aux_cpu = aux_full.to("cpu", dtype=torch.bfloat16)
                for j, t_abs in enumerate(range(t0, t1)):
                    carry_cache[t_abs] = aux_cpu[j].clone()
                next_start = starts[i + 1] if i + 1 < len(starts) else T_all
                for old_t in [k for k in carry_cache if k < next_start]:
                    del carry_cache[old_t]
                del aux_alpha, aux_full, aux_cpu
            w = taper_weights(Tc, overlap, is_first=(i == 0),
                              is_last=(i == len(starts) - 1),
                              device="cpu", dtype=torch.float32)
            weighted = (baton_gpu * w.to(device).view(Tc, 1, 1, 1)).to("cpu").numpy()
            w_np = w.numpy()
            for j, t_abs in enumerate(range(t0, t1)):
                if t_abs not in live_acc:
                    live_acc[t_abs] = weighted[j].copy()
                    live_w[t_abs] = float(w_np[j])
                else:
                    live_acc[t_abs] += weighted[j]
                    live_w[t_abs] += float(w_np[j])
            next_start = starts[i + 1] if i + 1 < len(starts) else T_all
            del s1, baton_gpu, weighted, ff, cradio_feat, moge_feat, rvm_feat
            del rgb_c, hint_c, hint_quality_c, hint_cpu, hint_quality_cpu
            del model_chunk, display_chunk, review_chunk
            flush_before(next_start)
        _synchronize(device)
        flush_before(T_all + 1)
    finally:
        for handle in hook_handles:
            handle.remove()
        stage1_progress.close()
        reader.close()
        if hint_reader is not None:
            hint_reader.close()

    if baton_fp16 is not None:
        baton_fp16.flush()
    del baton_fp16, live_acc, live_w, rvm_map, cradio_map, moge_map
    _empty_accelerator_cache(device)


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


def _chw_to_bgr_f32_cpu(x: torch.Tensor) -> np.ndarray:
    return (x[:, [2, 1, 0]].permute(0, 2, 3, 1)
            .contiguous().cpu().numpy().astype(np.float32, copy=False))


def _chw_to_bgra_f32_cpu(rgb: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    bgra = torch.cat([rgb[:, [2, 1, 0]], alpha], dim=1)
    return (bgra.permute(0, 2, 3, 1)
            .contiguous().cpu().numpy().astype(np.float32, copy=False))


def _render_predictions_exr_torch(meta: dict, outputs: set[str],
                                  checker: torch.Tensor,
                                  logits: torch.Tensor) -> dict[str, np.ndarray]:
    """Render a stage-2 batch as unclamped scene-linear float32 NHWC arrays.

    HDR contract: `fg`, `cutout`, and `checker` color stays in decoded linear
    light with no [0, 1] clamp and no quantization, so values above 1.0 (and
    negative values the model may emit) survive into the EXR files. Alpha is a
    matte and stays clamped to [0, 1].
    """
    alpha_model = logits[:, 0:1].float()
    pa_model = _crop_native_torch(alpha_model, meta).clamp(0.0, 1.0)
    despill_strength = float(meta.get("despill_strength", 0.5))

    pa = _resize_output_torch(pa_model, meta).clamp(0.0, 1.0)
    rendered: dict[str, np.ndarray] = {}
    if "alpha" in outputs:
        rendered["alpha"] = (pa.permute(0, 2, 3, 1)
                             .contiguous().cpu().numpy()
                             .astype(np.float32, copy=False))
    if not (outputs & {"fg", "cutout", "checker"}):
        return rendered

    fg_model = logits[:, 1:4].float()
    pf_model = _crop_native_torch(fg_model, meta)
    pf_linear = _linear_adjust_inverse_torch(cu.asinh_to_linear(pf_model), meta)
    pf_linear = _resize_output_torch(pf_linear, meta)
    if "fg" in outputs:
        rendered["fg"] = _chw_to_bgr_f32_cpu(pf_linear)
    if "cutout" in outputs:
        pf_linear_despilled = _despill_average_srgb_torch(
            pf_linear, strength=despill_strength)
        rendered["cutout"] = _chw_to_bgra_f32_cpu(pf_linear_despilled, pa)
    if "checker" in outputs:
        checker_b = checker.to(device=pf_linear.device,
                               dtype=pf_linear.dtype).unsqueeze(0)
        bg_lin = cu.srgb_to_linear(checker_b)
        pf_despill = _despill_average_srgb_torch(
            pf_linear, strength=despill_strength)
        comp_lin = pf_despill * pa + bg_lin * (1.0 - pa)
        rendered["checker"] = _chw_to_bgr_f32_cpu(comp_lin)
    return rendered


def _render_predictions_torch(meta: dict, outputs: set[str],
                              checker: torch.Tensor,
                              logits: torch.Tensor) -> dict[str, np.ndarray]:
    """Render a stage-2 batch on GPU, returning uint16 BGR/BGRA NHWC arrays.

    This removes the previous per-frame numpy HDR decode, sRGB encode,
    composition, channel shuffle, and uint8 conversion hot path.
    """
    alpha_model = logits[:, 0:1].float()
    pa_model = _crop_native_torch(alpha_model, meta).clamp(0.0, 1.0)
    despill_strength = float(meta.get("despill_strength", 0.5))

    pa = _resize_output_torch(pa_model, meta).clamp(0.0, 1.0)
    rendered: dict[str, np.ndarray] = {}
    if "alpha" in outputs:
        rendered["alpha"] = _srgb_to_bgr_uint16_cpu(pa.expand(-1, 3, -1, -1))
    if not (outputs & {"fg", "cutout", "checker"}):
        return rendered

    fg_model = logits[:, 1:4].float()
    pf_model = _crop_native_torch(fg_model, meta)
    pf_linear = _linear_adjust_inverse_torch(cu.asinh_to_linear(pf_model), meta)
    pf_linear_pos = pf_linear.clamp(min=0.0)
    pf = _resize_output_torch(cu.linear_to_srgb(pf_linear_pos), meta).clamp(0.0, 1.0)
    if "fg" in outputs:
        rendered["fg"] = _srgb_to_bgr_uint16_cpu(pf)
    if "cutout" in outputs:
        pf_linear_despilled = _despill_average_srgb_torch(pf_linear_pos, strength=despill_strength)
        if meta.get("cutout_linear", False):
            pf_linear_despilled_resized = _resize_output_torch(pf_linear_despilled, meta).clamp(0.0, 1.0)
            rendered["cutout"] = _srgb_to_bgra_uint16_cpu(pf_linear_despilled_resized, pa)
        else:
            pf_despill = _despill_average_srgb_torch(pf, strength=despill_strength)
            rendered["cutout"] = _srgb_to_bgra_uint16_cpu(pf_despill, pa)
    if "checker" in outputs:
        checker_b = checker.to(device=pf.device, dtype=pf.dtype).unsqueeze(0)
        bg_lin = cu.srgb_to_linear(checker_b)
        alpha = pa
        pf_despill = _despill_average_srgb_torch(pf, strength=despill_strength)
        comp_despill_lin = (
            cu.srgb_to_linear(pf_despill) * alpha + bg_lin * (1.0 - alpha))
        comp_despill = cu.linear_to_srgb(
            comp_despill_lin.clamp(min=0.0)).clamp(0.0, 1.0)
        rendered["checker"] = _srgb_to_bgr_uint16_cpu(comp_despill)
    return rendered


def run_inference(model, input_path: str, output_dir: str, cfg: InferenceConfig,
                  writer_cfg: WriterConfig, *, initial_hint_path: str | None,
                  hint_video_path: str | None):
    inference_started = time.perf_counter()
    device = model._accelerator_device
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
    H_out, W_out = meta["output_hw"]
    meta["cutout_linear"] = cfg.cutout_linear
    meta["despill_strength"] = cfg.despill_strength
    fps = float(meta["native_fps"])
    clip_name = raw_clip_name(input_path)
    checker = make_checker(H_out, W_out)
    checker_t = torch.from_numpy(checker)
    writers = {}
    stage2_progress = None
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
                    dtype=cfg.exr_dtype, frame_name=suffix[name])
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
        # MPS scaled-dot-product attention may materialize its full attention
        # matrix. At HD resolutions, batching the per-frame /16 encoder over a
        # long Hann window can therefore request well over 100 GB even when the
        # actual model/activation working set is much smaller. These chunking
        # knobs are mathematically exact and keep every compute chunk on MPS;
        # they are independent of the CPU module staging used by --low_vram.
        if cfg.low_vram or device.type == "mps":
            configure_low_vram_inference(model, cfg, device=device)
            if device.type == "mps" and not cfg.low_vram:
                print("    MPS safe attention chunking enabled "
                      "(GPU execution remains enabled)")
        rvm_path = cfg.temp_dir / f"{clip_name}_rvm_feat.f16"
        baton_path = cfg.temp_dir / f"{clip_name}_baton.f16"
        foundation_prefix = cfg.temp_dir / f"{clip_name}_foundation"
        for p in (
            rvm_path, baton_path, Path(str(foundation_prefix) + "_cradio.f32"),
            Path(str(foundation_prefix) + "_moge.f32"),
        ):
            safe_remove(p)
        if model.foundations.rvm is not None:
            precompute_rvm_to_memmap(
                model, input_path, meta, rvm_path,
                chunk=max(1, min(int(cfg.hann_chunk), 8)),
                reader_kind=reader_kind)
            rvm_feat_path = rvm_path
        else:
            rvm_feat_path = None
        foundation_paths = None
        if cfg.low_vram:
            foundation_paths = precompute_foundations_to_memmap(
                model, input_path, meta, foundation_prefix,
                chunk=max(1, int(cfg.low_vram_foundation_chunk)),
                reader_kind=reader_kind)
            model.foundations.cpu()
            model.stage2.cpu()
            model.stage1.to(device)
            _empty_accelerator_cache(device)
        else:
            model.stage2.to(device)
            model.stage2.eval()
            _empty_accelerator_cache(device)

        stage2_batch = max(1, int(cfg.stage2_batch))
        stage2_progress = _inline_progress(
            total=T_all, desc="    Stage 2 + output", unit="frame")

        def write_prediction_batches(items):
            device = next(model.stage2.parameters()).device
            for offset in range(0, len(items), stage2_batch):
                sub = items[offset:offset + stage2_batch]
                rgb_t = torch.stack([item[2] for item in sub], dim=0)
                rgb_t = rgb_t.unsqueeze(0).to(
                    device, dtype=_inference_dtype(device), non_blocking=True)
                baton_np = np.stack([item[1] for item in sub], axis=0)
                baton_t = torch.from_numpy(baton_np).unsqueeze(0)
                baton_t = baton_t.to(device, non_blocking=True).float()
                with torch.inference_mode(), _autocast(device):
                    logits = model.stage2(rgb_t, baton_t)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                render_fn = (_render_predictions_exr_torch if use_exr
                             else _render_predictions_torch)
                rendered = render_fn(meta, cfg.outputs, checker_t, logits[0])
                for name, frames in rendered.items():
                    writers[name].write_many(frames)
                stage2_progress.update(len(sub))
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
            on_baton_frames=None if cfg.low_vram else write_prediction_batches)

        if cfg.low_vram:
            _phase2_setup(model)
            baton = np.memmap(
                baton_path, dtype=np.float16, mode="r",
                shape=(T_all, model.stage1.d_16, H // 16, W // 16))
            reader = reader_cls(input_path, meta)
            try:
                for t0 in range(0, T_all, stage2_batch):
                    t1 = min(T_all, t0 + stage2_batch)
                    model_chunk, _display_chunk, _review_chunk = reader.window(t0, t1)
                    items = [
                        (t, np.asarray(baton[t]).copy(), model_chunk[t - t0])
                        for t in range(t0, t1)
                    ]
                    write_prediction_batches(items)
                    del model_chunk, _display_chunk, _review_chunk, items
            finally:
                reader.close()
                del baton
    finally:
        if stage2_progress is not None:
            stage2_progress.close()
        for writer in writers.values():
            writer.release()
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_rvm_feat.f16")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_baton.f16")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_foundation_cradio.f32")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_foundation_moge.f32")
        _empty_accelerator_cache(device)
    print(f"  wrote {len(writers)} output video(s) for {clip_name} "
          f"({T_all} frames @ {W_out}x{H_out})")
    elapsed = time.perf_counter() - inference_started
    print(f"  inference runtime: {elapsed / 60:.1f} min ({elapsed:.1f} s), "
          f"{T_all / elapsed:.2f} frames/s")
