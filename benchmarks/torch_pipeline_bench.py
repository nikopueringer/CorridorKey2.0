#!/usr/bin/env python3
"""The PyTorch product's whole pipeline, timed like `corridorkey bench` times the Rust engine.

Runs the product's own `runtime.run_inference` (RVM, backbones, Stage 1 windows with the Hann
blend, Stage 2, output rendering) on the same clip `corridorkey bench` keys: the test portrait
resized to `--size`, moving 6 px per frame. For a like-for-like comparison:

- frames are decoded into memory before timing (the Rust bench keys from memory): the product's
  frame reader is served from that cache, its preprocessing itself (on the CPU) stays;
- outputs (alpha, fg) are collected, not encoded (the Rust bench returns them to the host);
- peak device memory is this process's, as the driver reports it (NVML, sampled every 10 ms),
  the same instrument `corridorkey bench` uses;
- one window of `--frames` frames (`hann_chunk = hann_stride·2 = frames`), as the Rust bench.

Precision modes: `fp32` (autocast off), `bf16` (the product as shipped: bf16 autocast) and
`fastest` (the GPU's fastest native type: bf16 on sm >= 8.0, fp16 on 7.x, where bf16 is emulated).
Each mode runs `--runs` timed passes after one untimed pass; out-of-memory is recorded as such.

Appends one JSON line per mode to `--json`.

usage: uv run --with nvidia-ml-py python benchmarks/torch_pipeline_bench.py --size 768x512 --frames 8 \\
           --modes fp32,bf16,fastest --json benchmarks/results/torch.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pynvml
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridorkey import color_utils as cu  # noqa: E402
from corridorkey import runtime, video_io  # noqa: E402
from corridorkey.runtime import InferenceConfig, RuntimePaths, load_model  # noqa: E402
from corridorkey.video_io import WriterConfig  # noqa: E402


class PeakSampler:
    """This process's device memory (NVML, compute and graphics contexts), sampled on a thread."""

    def __init__(self, period: float = 0.01):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.pid = os.getpid()
        self.peak = 0
        self.period = period
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def used(self) -> int:
        procs = list(pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle))
        procs += list(pynvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle))
        return max((p.usedGpuMemory or 0 for p in procs if p.pid == self.pid), default=0)

    def run(self) -> None:
        while not self.stop_flag.is_set():
            self.peak = max(self.peak, self.used())
            time.sleep(self.period)

    def reset(self) -> None:
        self.peak = self.used()

    def stop(self) -> int:
        self.stop_flag.set()
        self.thread.join()
        return self.peak


class Collector:
    """Stands in for the product's writers: counts the frames handed to it, and keeps the alpha
    frames when `keep_alpha`."""

    frames: dict[str, int] = {}
    keep_alpha = False
    alpha: list[np.ndarray] = []

    def __init__(self, *args, frame_name: str = "frame", **kwargs):
        self.name = frame_name
        Collector.frames.setdefault(frame_name, 0)

    def write_many(self, frames):
        for f in frames:
            a = np.asarray(f)
            if Collector.keep_alpha and self.name == "alpha":
                Collector.alpha.append(np.array(a, dtype=np.float32).reshape(a.shape[:2]))
            Collector.frames[self.name] += 1

    def write(self, frame):
        self.write_many([frame])

    def release(self):
        pass


def preprocess_frame_dir_path_f32(path: str, meta: dict, *, need_model: bool):
    """`video_io.preprocess_frame_dir_path` keeping its views in float32 (it rounds them to bf16,
    which an fp32 model without autocast cannot take); as `scripts/parity/pipeline_reference.py`."""
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


PREPROCESS = video_io.preprocess_frame_dir_path


def fastest_dtype() -> torch.dtype:
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def patch_autocast(mode: str) -> None:
    original = getattr(patch_autocast, "original", torch.autocast)
    patch_autocast.original = original
    video_io.preprocess_frame_dir_path = preprocess_frame_dir_path_f32 if mode == "fp32" else PREPROCESS
    if mode == "fp32":
        torch.autocast = lambda *a, **k: contextlib.nullcontext()
    elif mode == "fastest":
        dtype = fastest_dtype()

        def autocast(device_type, dtype_arg=None, **kwargs):
            kwargs.pop("dtype", None)
            return original(device_type, dtype=dtype, **kwargs)

        torch.autocast = autocast
    else:  # bf16: the product as shipped
        torch.autocast = original


def make_clip(size: tuple[int, int], frames: int, out_dir: Path) -> list[Path]:
    w, h = size
    portrait = cv2.imread(str(ROOT / "data" / "green_screen_portrait.png"), cv2.IMREAD_COLOR)
    portrait = cv2.resize(portrait, (w, h), interpolation=cv2.INTER_LANCZOS4)
    paths = []
    for i in range(frames):
        p = out_dir / f"{i:04d}.png"
        cv2.imwrite(str(p), np.roll(portrait, 6 * i, axis=1))
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="768x512")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--modes", default="fp32,bf16,fastest")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--json", required=True)
    ap.add_argument("--dump-alpha", help="directory for each mode's alpha of the last run, `<mode>.f32` [T, h, w]")
    ap.add_argument("--dump-frames", help="file for the clip's frames, raw RGB uint8 [T, h, w, 3]")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sampler = PeakSampler()
    work = Path(tempfile.mkdtemp(prefix="ck-torch-bench-"))
    frame_dir = work / "frames"
    frame_dir.mkdir()
    make_clip((w, h), args.frames, frame_dir)
    if args.dump_frames:
        rgb = [cv2.imread(str(p), cv2.IMREAD_COLOR)[..., ::-1] for p in sorted(frame_dir.glob("*.png"))]
        np.stack(rgb).tofile(args.dump_frames)

    # Frames decoded once, before timing; the product's preprocessing of them stays timed.
    read_views = video_io._read_frame_dir_views
    cache = {str(p): read_views(str(p), exr_linear=False) for p in sorted(frame_dir.glob("*.png"))}
    video_io._read_frame_dir_views = lambda path, *, exr_linear: cache[str(path)]
    runtime.ExrSequenceWriter = Collector
    runtime.ThreadedVideoWriter = Collector

    model = load_model(RuntimePaths.from_package(ROOT), "cuda")
    major, minor = torch.cuda.get_device_capability()
    for mode in args.modes.split(","):
        patch_autocast(mode)
        cfg = InferenceConfig(
            hann_chunk=args.frames, hann_stride=max(1, args.frames // 2), outputs={"alpha", "fg"},
            output_format="exr", exr_dtype="float", temp_dir=work / "tmp",
        )
        result = {
            "engine": "torch", "runtime": "cuda", "precision": mode,
            "dtype": {"fp32": "fp32", "bf16": "bf16", "fastest": str(fastest_dtype()).split(".")[-1]}[mode],
            "gpu": torch.cuda.get_device_name(), "sm": f"{major}.{minor}",
            "width": w, "height": h, "frames": args.frames, "chunk": args.frames,
        }
        times = []
        try:
            torch.cuda.empty_cache()
            sampler.reset()
            for i in range(args.runs + 1):
                Collector.keep_alpha = bool(args.dump_alpha) and i == args.runs
                Collector.alpha = []
                torch.cuda.synchronize()
                t = time.perf_counter()
                with contextlib.redirect_stdout(open(os.devnull, "w")):
                    runtime.run_inference(model, str(frame_dir), str(work / "out"), cfg, WriterConfig(),
                                          initial_hint_path=None, hint_video_path=None)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            if args.dump_alpha:
                Path(args.dump_alpha).mkdir(parents=True, exist_ok=True)
                np.stack(Collector.alpha).astype(np.float32).tofile(Path(args.dump_alpha) / f"{mode}.f32")
            steady = times[1:]
            result |= {"first_run_s": times[0], "runs_s": steady, "best_s": min(steady),
                       "mean_s": sum(steady) / len(steady), "fps": args.frames / min(steady),
                       "peak_vram_bytes": sampler.peak}
        except torch.cuda.OutOfMemoryError as e:
            result |= {"error": "out of memory", "detail": str(e).splitlines()[0][:200], "peak_vram_bytes": sampler.peak}
        torch.cuda.empty_cache()
        line = json.dumps(result)
        print(line, flush=True)
        with open(args.json, "a") as f:
            f.write(line + "\n")
    sampler.stop()


if __name__ == "__main__":
    main()
