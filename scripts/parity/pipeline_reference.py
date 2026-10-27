#!/usr/bin/env python3
"""End-to-end fp32 reference of the product's `runtime.run_inference`, for the Rust pipeline.

Runs the product's own driver on a short PNG frame sequence (the portrait with horizontal motion)
with two overlapping Stage 1 windows (`hann_chunk` 8, `hann_stride` 4 over 12 frames), carried
hints, every output as float32 EXR frames. Three patches make it an fp32 reference without changing
its logic:

- `torch.autocast` is disabled (the product runs the backbones and stages under bf16 autocast);
- `video_io.preprocess_frame_dir_path` keeps the model and display views in float32 (the product
  rounds them to bf16 on the CPU);
- `runtime.ExrSequenceWriter` is replaced by a collector of the exact arrays handed to the writer.

The product's other roundings stay, being part of its logic: batons to float16 after the Hann
blend, carried hints and hint quality to bf16.

Writes `frames/NNNN.png`, `frames_rgb8.u8` (the same frames, raw RGB), `<output>.f32` (`[T, h, w, C]` as written: BGR / BGRA / alpha) and
`meta.json`.

With `--exr-linear`: the frames are scene-linear EXR (the portrait's linear values × 1.6, so
highlights go above 1) keyed with `frame_dir_linear`, writing `frames/NNNN.exr` and
`frames_rgbf32.f32` (`[T, h, w, 3]` float RGB) instead of the PNG and byte files.

With `--hints`: also a first-frame hint (`hint_first.png`, 8-bit, alpha channel holding the matte)
and a hint frame directory (`hints/NNNN.png`, 8-bit RGB mattes the product turns to gray), passed
as `initial_hint_path` / `hint_video_path`; their values as read by the product are written to
`hint_first.f32` (`[h, w]`) and `hints.f32` (`[T, h, w]`). The mattes are rough green-difference
keys of the frames, enough to exercise every hint path.

With `--hints-exr`: the same hints as EXR files, the hint frames at 3/4 of the frames' size and in
float (they go through the product's EXR reader, float gray conversion and Lanczos resize).

usage: pipeline_reference.py <out_dir> [--exr-linear] [--hints | --hints-exr]
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from corridorkey import color_utils as cu  # noqa: E402
from corridorkey import exr_io, runtime, video_io  # noqa: E402
from corridorkey.runtime import InferenceConfig, RuntimePaths, load_model  # noqa: E402
from corridorkey.video_io import WriterConfig  # noqa: E402

FRAMES, CHUNK, STRIDE = 12, 8, 4
OUTPUTS = ("alpha", "fg", "cutout", "checker")


def preprocess_frame_dir_path_f32(path: str, meta: dict, *, need_model: bool):
    """`video_io.preprocess_frame_dir_path` without the bf16 rounding of its views."""
    h_out, w_out = meta["shape_hw"]
    h_model, w_model = meta["model_native_hw"]
    display, linear = video_io._read_frame_dir_views(path, exr_linear=bool(meta.get("exr_linear", False)))
    display = video_io.resize_hwc_lanczos(display, (h_model, w_model))
    linear = video_io.resize_hwc_lanczos(linear, (h_model, w_model))
    display, _ = video_io.center_pad_to_shape(np.clip(display, 0.0, 1.0), h_out, w_out)
    display_t = torch.from_numpy(display.transpose(2, 0, 1).copy()).float()
    if not need_model:
        return None, display_t, display_t
    linear, _ = video_io.center_pad_to_shape(linear, h_out, w_out)
    model_t = torch.from_numpy(cu.linear_to_asinh(linear).transpose(2, 0, 1).copy()).float()
    return model_t, display_t, display_t


def rough_matte(frame_bgr: np.ndarray) -> np.ndarray:
    """A soft green-difference key, uint8."""
    f = frame_bgr.astype(np.float32)
    spill = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    alpha = np.clip(1.0 - spill / 60.0, 0.0, 1.0)
    return (cv2.GaussianBlur(alpha, (0, 0), 3.0) * 255.0 + 0.5).astype(np.uint8)


def write_hints(out: Path, frames_bgr: list[np.ndarray], *, exr: bool) -> tuple[str, str]:
    """Writes the hint inputs and the values the product reads from them (see the module docs)."""
    mattes = [rough_matte(f) for f in frames_bgr]
    noise = np.random.default_rng(1).integers(0, 256, mattes[0].shape, dtype=np.uint8)
    hint_dir = out / "hints"
    hint_dir.mkdir(exist_ok=True)
    if exr:
        # BGRA float with the matte in alpha and noise in the color: `read_hint_image` must pick alpha
        first = out / "hint_first.exr"
        n = noise.astype(np.float32) / 255.0
        exr_io.write_exr(first, np.dstack([n, n, n, mattes[0].astype(np.float32) / 255.0]), half=False)
        h, w = mattes[0].shape
        small = (w * 3 // 4, h * 3 // 4)
        for i, m in enumerate(mattes):
            # Unequal float channels at 3/4 size: gray conversion and Lanczos resize both matter
            m = cv2.resize(m.astype(np.float32) / 255.0, small, interpolation=cv2.INTER_AREA)
            exr_io.write_exr(hint_dir / f"{i:04d}.exr", np.dstack([m * 0.5, m, np.minimum(m + 0.08, 1.2)]), half=False)
    else:
        first = out / "hint_first.png"
        # BGRA with the matte in alpha and noise in B: `read_hint_image` must pick alpha
        cv2.imwrite(str(first), np.dstack([noise, noise, noise, mattes[0]]))
        for i, m in enumerate(mattes):
            # Unequal channels, so the gray conversion matters
            bgr = np.dstack([m // 2, m, np.minimum(m.astype(np.uint16) + 20, 255).astype(np.uint8)])
            cv2.imwrite(str(hint_dir / f"{i:04d}.png"), bgr)
    video_io.read_hint_image(first).astype(np.float32).tofile(out / "hint_first.f32")
    meta = {"shape_hw": frames_bgr[0].shape[:2], "model_native_hw": frames_bgr[0].shape[:2]}
    hints = [video_io.preprocess_hint_frame(video_io.imread_unchanged(p), meta)[0].float().numpy()
             for p in sorted(hint_dir.iterdir())]
    np.stack(hints).astype(np.float32).tofile(out / "hints.f32")
    return str(first), str(hint_dir)


class Collector:
    """Stands in for `ExrSequenceWriter`: keeps every frame handed to it."""

    frames: dict[str, list[np.ndarray]] = {}

    def __init__(self, dir_path, fps, size, *, channels, dtype, frame_name):
        self.name = frame_name
        Collector.frames.setdefault(frame_name, [])

    def write_many(self, frames):
        Collector.frames[self.name].extend(np.asarray(f, dtype=np.float32) for f in frames)

    def write(self, frame):
        self.write_many([frame])

    def release(self):
        pass


def main() -> None:
    out = Path(sys.argv[1])
    exr_linear = "--exr-linear" in sys.argv[2:]
    hints_exr = "--hints-exr" in sys.argv[2:]
    with_hints = "--hints" in sys.argv[2:] or hints_exr
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    portrait = cv2.imread(str(ROOT / "data" / "green_screen_portrait.png"), cv2.IMREAD_COLOR)
    frames_bgr = [np.roll(portrait, 6 * i, axis=1) for i in range(FRAMES)]
    if exr_linear:
        frames_lin = [(cu.srgb_to_linear(f.astype(np.float32) / 255.0) * 1.6).astype(np.float32) for f in frames_bgr]
        for i, frame in enumerate(frames_lin):
            exr_io.write_exr(frames_dir / f"{i:04d}.exr", frame, half=False)
        np.stack([f[..., ::-1] for f in frames_lin]).astype(np.float32).tofile(out / "frames_rgbf32.f32")
    else:
        for i, frame in enumerate(frames_bgr):
            cv2.imwrite(str(frames_dir / f"{i:04d}.png"), frame)
        # The same frames as raw RGB bytes ([T, h, w, 3] uint8), the Rust tests' input
        np.stack([f[..., ::-1] for f in frames_bgr]).tofile(out / "frames_rgb8.u8")

    torch.autocast = lambda *a, **k: contextlib.nullcontext()  # fp32 reference
    video_io.preprocess_frame_dir_path = preprocess_frame_dir_path_f32
    runtime.ExrSequenceWriter = Collector
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = load_model(RuntimePaths.from_package(ROOT), "cuda")
    cfg = InferenceConfig(
        hann_chunk=CHUNK, hann_stride=STRIDE, outputs=set(OUTPUTS), output_format="exr", exr_dtype="float",
        temp_dir=out / "tmp", frame_dir_linear=exr_linear,
    )
    initial_hint_path = hint_video_path = None
    if with_hints:
        initial_hint_path, hint_video_path = write_hints(out, frames_bgr, exr=hints_exr)
    runtime.run_inference(model, str(frames_dir), str(out / "unused"), cfg, WriterConfig(),
                          initial_hint_path=initial_hint_path, hint_video_path=hint_video_path)

    shapes = {}
    for name, frames in Collector.frames.items():
        arr = np.stack(frames).astype(np.float32)
        arr = arr[..., None] if arr.ndim == 3 else arr
        arr.tofile(out / f"{name}.f32")
        shapes[name] = list(arr.shape)
    meta = {"frames": FRAMES, "exr_linear": exr_linear, "hints": with_hints, "chunk": CHUNK, "stride": STRIDE, "shapes": shapes,
            "carry_hint_quality": cfg.carry_hint_quality, "despill": cfg.despill_strength}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
