#!/usr/bin/env python3
"""Alpha accuracy of every engine and precision against the PyTorch product in fp32, on the clip
`torch_pipeline_bench.py --dump-frames/--dump-alpha` keyed (one window of all its frames).

Rust runs through `corridorkey_rs`, one runtime and precision per process (CubeCL keeps its memory
pools for the life of a process). Errors are
absolute alpha differences, also in 16-bit output codes (1 code = 1/65535).

usage: CORRIDORKEY_RUNTIME=cuda crates/corridorkey-py/.venv/bin/python benchmarks/accuracy.py \\
           benchmarks/results/final 768x512 8 f16      (or: ... 768x512 8 --torch)
Appends JSON lines to `<dir>/accuracy.jsonl`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def stats(got: np.ndarray, want: np.ndarray) -> dict:
    err = np.abs(got.astype(np.float64) - want.astype(np.float64))
    return {"mean_abs": float(err.mean()), "p99_abs": float(np.quantile(err, 0.99)), "max_abs": float(err.max()),
            "mean_codes16": float(err.mean() * 65535)}


def main() -> None:
    out_dir, size, frames = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    w, h = (int(v) for v in size.split("x"))
    acc = out_dir / f"acc_{size}"
    want = np.fromfile(acc / "fp32.f32", np.float32).reshape(frames, h, w)
    rows = []
    if "--torch" in sys.argv:
        for mode, label in (("fastest", "fp16"), ("bf16", "bf16")):
            p = acc / f"{mode}.f32"
            if p.exists():
                got = np.fromfile(p, np.float32).reshape(frames, h, w)
                rows.append({"engine": "torch", "runtime": "cuda", "precision": label} | stats(got, want))
    else:
        import corridorkey_rs as ck

        rgb = np.fromfile(out_dir / f"acc_{size}_frames.u8", np.uint8).reshape(frames, h, w, 3)
        precision = sys.argv[4]
        keyer = ck.Keyer(root=str(ROOT), precision=precision)
        got = keyer.key_frames(rgb, outputs=["alpha"], hann_chunk=frames, hann_stride=max(1, frames // 2))["alpha"]
        rows.append({"engine": "rust", "runtime": keyer.runtime, "precision": precision} | stats(got, want))
    with open(out_dir / "accuracy.jsonl", "a") as f:
        for r in rows:
            r |= {"size": size, "frames": frames, "reference": "torch fp32"}
            print(json.dumps(r))
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
