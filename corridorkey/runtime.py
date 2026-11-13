from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import color_utils as cu
from .model import CorridorKeyV3
from .device import area_resize, empty_cache, memory_budget_gb, model_device, synchronize
from .mps_attention import available as native_attention_available
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
        # Packaged apps keep model downloads outside the application bundle.
        root = Path(os.environ.get("CORRIDORKEY_ASSET_ROOT") or package_root).expanduser().resolve()
        return cls(
            package_root=Path(package_root).resolve(),
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
    input_color_space: str = "legacy"
    color_output: str = "processed_working"
    input_range: str = "auto"


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


def apply_auto_vram(cfg: InferenceConfig, shape_hw, num_frames: int, device="cuda") -> None:
    budget_gb = memory_budget_gb(device)
    chunk, stride, low_vram, estimate = select_vram_profile(shape_hw, num_frames, budget_gb)
    cfg.hann_chunk, cfg.hann_stride, cfg.low_vram = chunk, stride, low_vram
    if torch.device(device).type == "mps" and low_vram:
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
            # These controls preserve attention token sets and model precision.
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
    estimate_label = "CUDA reference peak" if torch.device(device).type == "mps" else "estimated peak"
    print(f"  Auto VRAM: {budget_gb:.1f} GB available at {shape_hw[1]}x{shape_hw[0]} -> "
          f"hann_chunk={chunk}, hann_stride={stride}, low_vram={low_vram} "
          f"({estimate_label} {estimate:.1f} GB)")
    if estimate * _VRAM_MARGIN_SCALE + _VRAM_MARGIN_GB > budget_gb:
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


def load_model(paths: RuntimePaths, device: str = "cuda") -> CorridorKeyV3:
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
    # PR #9's storage optimization was validated on CUDA. Preserve PR #15's
    # original storage policy on Metal until same-device parity is established.
    if torch.device(device).type == "cuda":
        for module in model.modules():
            if isinstance(module, _BF16_WEIGHT_MODULES):
                for p in module.parameters(recurse=False):
                    p.data = p.data.to(torch.bfloat16)
    del ckpt
    return model


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

            rgb_c = model_chunk.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
            hint_c = hint_cpu.to(device, non_blocking=device.type == "cuda")
            hint_quality_c = hint_quality_cpu.to(device, non_blocking=device.type == "cuda")
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
                rvm_feat = rvm_feat.unsqueeze(0).to(device, non_blocking=device.type == "cuda").float()
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
            if device.type == "mps":
                # Release unused Metal buffers between overlapping windows.
                empty_cache(device)
            print(f"      window {i + 1}/{len(starts)} [{t0}:{t1}]", flush=True)
        synchronize(device)
        flush_before(T_all + 1)
    finally:
        reader.close()
        if hint_reader is not None:
            hint_reader.close()

    if baton_fp16 is not None:
        baton_fp16.flush()
    del baton_fp16, live_acc, live_w, rvm_map, cradio_map, moge_map
    empty_cache(device)


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


def _replace_rendered_rgb_with_source(rendered: dict[str, np.ndarray],
                                      source_frames: list[np.ndarray]):
    """Preserve native decoded source RGB while keeping inferred alpha straight."""
    source_bgr = np.stack([frame[..., ::-1] for frame in source_frames])
    if "fg" in rendered:
        rendered["fg"] = source_bgr.astype(np.float32, copy=False)
    if "cutout" in rendered:
        alpha = rendered["cutout"][..., 3:4]
        rendered["cutout"] = np.concatenate((source_bgr, alpha), axis=-1).astype(
            np.float32, copy=False)
    return rendered


def _write_color_manifest(path: str | os.PathLike, manifest: dict) -> None:
    """Atomically update selected output descriptions, preserving old files."""
    path = Path(path)
    merged = dict(manifest)
    outputs = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing.get("outputs"), dict):
                outputs.update(existing["outputs"])
        except (OSError, ValueError, TypeError):
            pass
    outputs.update(manifest.get("outputs", {}))
    merged["outputs"] = outputs
    temp_path = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with open(temp_path, "w", encoding="utf-8") as mf:
            json.dump(merged, mf, indent=2)
            mf.write("\n")
            mf.flush()
            os.fsync(mf.fileno())
        os.replace(temp_path, path)
    finally:
        safe_remove(temp_path)


def _encode_rendered_camera_source(rendered: dict[str, np.ndarray],
                                   profile_id: str, *, video: bool,
                                   legacy_linear: bool = False):
    """Convert unclamped working-linear render arrays to camera-log output.

    EXR retains signed and over-range camera code values. Video output must
    quantize to its representable range, but its camera primaries/TRC tags are
    intentionally omitted by the writer.
    """
    if profile_id == "legacy" and not legacy_linear:
        encode_rgb = lambda rgb: cu.linear_to_srgb(np.maximum(rgb, 0.0))
    elif profile_id == "legacy":
        encode_rgb = lambda rgb: rgb
    else:
        from .camera_color import encode_from_working
        encode_rgb = lambda rgb: encode_from_working(rgb, profile_id)
    if "fg" in rendered:
        rendered["fg"] = np.ascontiguousarray(
            encode_rgb(rendered["fg"][..., ::-1])[..., ::-1])
    if "cutout" in rendered:
        cutout = rendered["cutout"]
        rgb = encode_rgb(cutout[..., 2::-1])
        rendered["cutout"] = np.concatenate(
            (rgb[..., ::-1], cutout[..., 3:4]), axis=-1)
    if video:
        for name in ("fg", "cutout"):
            if name in rendered:
                rendered[name] = (
                    np.clip(rendered[name], 0.0, 1.0) * 65535.0
                ).astype(np.uint16)
        if "checker" in rendered:
            checker_rgb = rendered["checker"][..., ::-1]
            checker_srgb = np.clip(
                cu.linear_to_srgb(np.maximum(checker_rgb, 0.0)), 0.0, 1.0)
            rendered["checker"] = (
                checker_srgb[..., ::-1] * 65535.0
            ).astype(np.uint16)
        if "alpha" in rendered:
            alpha = np.clip(rendered["alpha"], 0.0, 1.0)
            alpha_rgb = np.repeat(alpha, 3, axis=-1)
            rendered["alpha"] = (alpha_rgb * 65535.0).astype(np.uint16)
    return rendered


def run_inference(model, input_path: str, output_dir: str, cfg: InferenceConfig,
                  writer_cfg: WriterConfig, *, initial_hint_path: str | None,
                  hint_video_path: str | None):
    # Half EXR cannot retain adjacent 16-bit source codes. Apply the same
    # fidelity guarantee to CLI/API callers as to the desktop output control.
    if cfg.color_output == "source_rgb" and cfg.exr_dtype != "float":
        cfg = replace(cfg, exr_dtype="float")
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
        if (cfg.input_color_space == "legacy" and not cfg.frame_dir_linear
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
    if cfg.input_range not in ("auto", "full", "legal"):
        raise ValueError(f"unsupported input range {cfg.input_range!r}")
    if cfg.color_output not in ("processed_working", "processed_source", "source_rgb"):
        raise ValueError(f"unsupported color output {cfg.color_output!r}")
    effective_color_output = cfg.color_output
    if cfg.input_color_space != "legacy":
        from .camera_color import PROFILES
        if cfg.input_color_space not in PROFILES:
            raise ValueError(f"unknown input color space {cfg.input_color_space!r}")
    if effective_color_output == "source_rgb" and cfg.output_format != "exr":
        raise ValueError("source_rgb requires --output_format exr to preserve source codes")
    elif (effective_color_output == "processed_source" and is_dir
          and cfg.frame_dir_linear and cfg.output_format != "exr"):
        raise ValueError(
            "processed_source for linear frame sequences requires --output_format exr")
    meta["input_color_space"] = cfg.input_color_space
    meta["color_output"] = effective_color_output
    meta["input_range"] = cfg.input_range
    meta["ffmpeg_bin"] = writer_cfg.ffmpeg_bin
    if is_dir and cfg.input_color_space != "legacy":
        meta["linear_brightness"] = float(cfg.linear_brightness)
        meta["linear_contrast"] = float(cfg.linear_contrast)
        meta["linear_contrast_pivot"] = float(cfg.linear_contrast_pivot)
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
    source_reader = None
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
                    fps, (W_out, H_out), writer_cfg, channels=channels,
                    color_trc=color_trc,
                    tag_color=not (
                        cfg.input_color_space != "legacy"
                        and effective_color_output == "processed_source"
                        and name in ("fg", "cutout")))

        if effective_color_output == "source_rgb" and not is_dir:
            from .camera_video import CameraVideoReader
            source_reader = CameraVideoReader(
                input_path, width=int(meta["native_hw"][1]),
                height=int(meta["native_hw"][0]),
                input_range=cfg.input_range, ffmpeg_bin=writer_cfg.ffmpeg_bin)

        output_profiles = {}
        source_output_profile = (
            cfg.input_color_space if cfg.input_color_space != "legacy" else
            ("linear_rec709" if is_dir and cfg.frame_dir_linear else "srgb"))
        for name in cfg.outputs:
            if name == "alpha":
                output_profiles[name] = {"profile": "matte", "data": "matte"}
            elif name == "checker":
                output_profiles[name] = {
                    "profile": "linear_rec709" if use_exr else "srgb"}
            elif effective_color_output in ("processed_source", "source_rgb"):
                output_profiles[name] = {
                    "profile": source_output_profile,
                    **({"alpha": "straight"} if name == "cutout" else {}),
                }
            else:
                output_profiles[name] = {
                    "profile": ("linear_rec709" if use_exr else
                                ("linear_rec709" if name == "cutout" and cfg.cutout_linear
                                 else "srgb")),
                    **({"alpha": "straight"} if name == "cutout" else {}),
                }
            extension = "_exr" if use_exr else (".mov" if name == "cutout" else ".mp4")
            output_profiles[name]["path"] = f"{clip_name}_{suffix[name]}{extension}"
            output_profiles[name]["native_fps"] = fps
        color_manifest = {
            "schema_version": 1,
            "native_fps": fps,
            "input_color_space": cfg.input_color_space,
            "color_output": effective_color_output,
            "input_range": cfg.input_range,
            "outputs": output_profiles,
        }
        if (cfg.input_color_space != "legacy"
                and effective_color_output == "processed_source"
                and not use_exr):
            color_manifest["video_note"] = (
                "Camera-log RGB was clipped to [0,1] and encoded through a lossy "
                "video codec. Camera primaries/TRC tags are omitted; Y'CbCr "
                "matrix metadata is BT.709. Use EXR for unclipped float output.")
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
                sf.write(f"Input Color Space: {cfg.input_color_space}\n")
                sf.write(f"Color Output: {effective_color_output}\n")
                sf.write(f"Input Range: {cfg.input_range}\n")
                if (cfg.input_color_space != "legacy"
                        and effective_color_output == "processed_source"
                        and not use_exr):
                    sf.write("Camera-log video output is clipped/quantized and lossy; "
                             "camera primaries/TRC tags are omitted.\n")
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
            empty_cache(device)
        else:
            model.stage2.to(device)
            model.stage2.eval()
            empty_cache(device)

        stage2_batch = max(1, int(cfg.stage2_batch))

        def original_source_codes(relative_index: int) -> np.ndarray:
            absolute_index = int(meta["start_frame"]) + int(relative_index)
            if is_dir:
                from .camera_video import read_frame_dir_codes
                return read_frame_dir_codes(meta["frame_paths"][absolute_index])
            if source_reader is None:
                raise RuntimeError("source RGB reader was not initialized")
            return source_reader.read_code_frame(absolute_index)

        def apply_color_output(rendered, sub):
            if effective_color_output == "processed_working":
                return rendered
            if effective_color_output == "processed_source":
                return _encode_rendered_camera_source(
                    rendered, cfg.input_color_space, video=not use_exr,
                    legacy_linear=is_dir and cfg.frame_dir_linear)

            # source_rgb keeps the decoded source codes at native dimensions;
            # only the inferred straight-alpha matte comes from the model.
            if not (cfg.outputs & {"fg", "cutout", "checker"}):
                return rendered
            source_frames = [original_source_codes(item[0]) for item in sub]
            source_alpha = (rendered["cutout"][..., 3:4]
                            if "cutout" in rendered else rendered.get("alpha"))
            rendered = _replace_rendered_rgb_with_source(rendered, source_frames)
            if "checker" in rendered and source_alpha is not None:
                bg_srgb = checker_t.permute(1, 2, 0).cpu().numpy()[None]
                bg_linear = cu.srgb_to_linear(bg_srgb)
                checker_frames = []
                for source_codes, alpha in zip(source_frames, source_alpha):
                    if cfg.input_color_space == "legacy":
                        source_linear = (source_codes if is_dir and cfg.frame_dir_linear
                                         else cu.srgb_to_linear(source_codes))
                    else:
                        from .camera_color import decode_to_working
                        source_linear = decode_to_working(
                            source_codes, cfg.input_color_space)
                    comp = source_linear * alpha + bg_linear[0] * (1.0 - alpha)
                    checker_frames.append(comp[..., ::-1])
                rendered["checker"] = np.stack(checker_frames).astype(np.float32)
            return rendered

        def write_prediction_batches(items):
            device = next(model.stage2.parameters()).device
            for offset in range(0, len(items), stage2_batch):
                sub = items[offset:offset + stage2_batch]
                rgb_t = torch.stack([item[2] for item in sub], dim=0)
                rgb_t = rgb_t.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
                baton_np = np.stack([item[1] for item in sub], axis=0)
                baton_t = torch.from_numpy(baton_np).unsqueeze(0)
                baton_t = baton_t.to(device, non_blocking=device.type == "cuda").float()
                with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
                    logits = model.stage2(rgb_t, baton_t)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                render_fn = (_render_predictions_exr_torch
                             if use_exr or effective_color_output == "processed_source"
                             else _render_predictions_torch)
                render_outputs = set(cfg.outputs)
                if (effective_color_output == "source_rgb"
                        and "checker" in cfg.outputs
                        and not (cfg.outputs & {"alpha", "cutout"})):
                    render_outputs.add("alpha")
                rendered = render_fn(meta, render_outputs, checker_t, logits[0])
                rendered = apply_color_output(rendered, sub)
                for hidden in set(rendered) - set(cfg.outputs):
                    del rendered[hidden]
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
        camera_close_error = None
        if source_reader is not None:
            try:
                source_reader.close()
            except Exception as exc:
                camera_close_error = exc
        for writer in writers.values():
            writer.release()
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_rvm_feat.f16")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_baton.f16")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_foundation_cradio.f32")
        safe_remove(cfg.temp_dir / f"{raw_clip_name(input_path)}_foundation_moge.f32")
        empty_cache(device)
        if camera_close_error is not None:
            if sys.exc_info()[0] is None:
                raise camera_close_error
            print(f"[WARNING] Camera source decoder close reported: {camera_close_error}")
    # Publish the interpretation only after every selected output is closed
    # successfully; a failed attempt must not relabel an older completed render.
    _write_color_manifest(
        os.path.join(output_dir, f"{clip_name}_color.json"), color_manifest)
    print(f"  wrote {len(writers)} output(s) for {clip_name} "
          f"({T_all} frames @ {W_out}x{H_out})")
