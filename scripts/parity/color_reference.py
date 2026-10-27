#!/usr/bin/env python3
"""Reference data and PyTorch timings for the colour transfer kernels.

Writes, for each frame size, the input frames, the f64 NumPy reference and the
PyTorch (CUDA, f32) results of every conversion as raw little-endian f32 files,
plus `torch_timings.json`. The Rust benchmark
(`crates/corridorkey-base/tests/bench_color.rs`) reads them, compares and times
its own kernels on the same data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from corridorkey.color_utils import (  # noqa: E402
    HDR_ASINH_KNEE, HDR_ASINH_REF, asinh_to_linear, linear_to_asinh,
    linear_to_srgb, srgb_to_linear,
)

SIZES = {"1080p": (1080, 1920), "4k": (2160, 3840)}
ADJUST = dict(gain=1.2, contrast=0.9, pivot=0.18)


def adjust(x, gain, contrast, pivot):
    return gain * (pivot + contrast * (x - pivot))


# name -> (input encoding, function)
CONVERSIONS = {
    "srgb_to_linear": ("srgb", srgb_to_linear),
    "linear_to_srgb": ("linear", linear_to_srgb),
    "linear_to_asinh": ("linear", linear_to_asinh),
    "asinh_to_linear": ("asinh", asinh_to_linear),
    # The model's input chain: decode sRGB, linear adjust, asinh
    "srgb_adjust_asinh": ("srgb", lambda x: linear_to_asinh(adjust(srgb_to_linear(x), **ADJUST))),
}


def inputs(h: int, w: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Frames in each encoding: HDR scene-linear with some negatives, and its encodings."""
    linear = rng.uniform(-0.05, 1.0, (h, w, 3)) ** 2 * 4.0 - 0.02
    linear[rng.random((h, w, 3)) < 0.01] *= -1.0
    srgb = np.clip(rng.random((h, w, 3)), 0.0, 1.0)
    return {
        "linear": linear.astype(np.float32),
        "srgb": srgb.astype(np.float32),
        "asinh": linear_to_asinh(linear).astype(np.float32),
    }


def cuda_median_ms(fn, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    timings = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cases": {}}

    for size, (h, w) in SIZES.items():
        frames = inputs(h, w, rng)
        for enc, x in frames.items():
            x.tofile(args.out / f"{size}_in_{enc}.f32")
        for name, (enc, fn) in CONVERSIONS.items():
            x32 = frames[enc]
            # Ground truth: the same function in f64 on the f32 input
            np.asarray(fn(x32.astype(np.float64)), dtype=np.float32).tofile(args.out / f"{size}_{name}_ref.f32")
            dev = torch.from_numpy(x32).cuda()
            out = fn(dev)
            out.cpu().numpy().astype(np.float32).tofile(args.out / f"{size}_{name}_torch.f32")
            resident = cuda_median_ms(lambda: fn(dev), args.iters)
            end_to_end = cuda_median_ms(lambda: fn(torch.from_numpy(x32).cuda()).cpu(), args.iters)
            timings["cases"][f"{size}/{name}"] = {"resident_ms": resident, "end_to_end_ms": end_to_end}
            print(f"{size:6} {name:18} torch cuda: resident {resident:7.3f} ms, end to end {end_to_end:7.3f} ms")

    meta = {"sizes": SIZES, "adjust": ADJUST, "knee": HDR_ASINH_KNEE, "ref": HDR_ASINH_REF,
            "conversions": {k: v[0] for k, v in CONVERSIONS.items()}}
    (args.out / "torch_timings.json").write_text(json.dumps({**timings, **meta}, indent=2))


if __name__ == "__main__":
    main()
