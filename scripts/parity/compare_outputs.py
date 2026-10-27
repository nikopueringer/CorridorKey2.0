#!/usr/bin/env python3
"""Error of Rust runner outputs against the PyTorch modes of `production_reference.py`, in the
units the product writes: alpha clamped to [0, 1], foreground decoded asinh -> linear -> sRGB,
cropped to the native (unpadded) region, plus how many 8- and 16-bit output codes differ.

usage: compare_outputs.py <ref_dir> <rust_out_dir>...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from corridorkey import color_utils as cu  # noqa: E402


def load(path: Path, ref: dict) -> tuple[np.ndarray, np.ndarray]:
    t, h, w = ref["t"], ref["h"], ref["w"]
    pad_l, _, pad_t, _ = ref["pad"]
    nh, nw = ref["native_hw"]
    x = np.fromfile(path, dtype=np.float32).reshape(t, h, w, 4)[:, pad_t:pad_t + nh, pad_l:pad_l + nw]
    alpha = np.clip(x[..., 0], 0, 1)
    with np.errstate(over="ignore"):
        fg = np.clip(cu.linear_to_srgb(np.maximum(cu.asinh_to_linear(x[..., 1:4]), 0)), 0, 1)
    return alpha, fg


def stats(name: str, got: np.ndarray, want: np.ndarray) -> None:
    diff = np.abs(got.astype(np.float64) - want)
    u16 = np.mean(np.round(got * 65535) != np.round(want * 65535)) * 100
    u8 = np.mean(np.round(got * 255) != np.round(want * 255)) * 100
    print(f"    {name:5} mean={diff.mean():.3e} max={diff.max():.3e} p99.99={np.quantile(diff, 0.9999):.3e} "
          f"u16 codes {u16:.2f}%  u8 codes {u8:.3f}%")


def main() -> None:
    ref_dir = Path(sys.argv[1])
    ref = json.loads((ref_dir / "torch_ref.json").read_text())
    want = load(ref_dir / "torch_fp32_logits_bthwc.f32", ref)
    print(f"{ref['device']} (sm_{''.join(map(str, ref['compute_capability']))}), T={ref['t']}, "
          f"{ref['native_hw'][1]}x{ref['native_hw'][0]} padded to {ref['w']}x{ref['h']}; error vs PyTorch fp32:")
    for name, mode in ref["modes"].items():
        if name == "fp32":
            continue
        if "oom" in mode:
            print(f"  torch {name} ({mode['dtype']}): out of memory")
            continue
        print(f"  torch {name} ({mode['dtype']}):")
        a, f = load(ref_dir / f"torch_{name}_logits_bthwc.f32", ref)
        stats("alpha", a, want[0])
        stats("fg", f, want[1])
    for out in map(Path, sys.argv[2:]):
        timing = json.loads((out / "rust_timing.json").read_text())
        print(f"  rust {timing['target']} {timing.get('dtype', 'f32')} ({out.name}):")
        a, f = load(out / "rust_logits_bthwc.f32", ref)
        stats("alpha", a, want[0])
        stats("fg", f, want[1])


if __name__ == "__main__":
    main()
