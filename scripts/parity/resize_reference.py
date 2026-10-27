#!/usr/bin/env python3
"""OpenCV references for `corridorkey-io`'s `resize.rs`: `cv2.resize(INTER_LANCZOS4)` of single-
channel float32 images (down, up, mixed, one axis, tiny) and `cv2.cvtColor(COLOR_BGR2GRAY)` of
float32 BGR and BGRA pixels, the two operations the product applies to hint frames.

Writes `<out>/cases.json` and raw float32 inputs and outputs.

usage: resize_reference.py <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

RESIZE = [((420, 664), (210, 332)), ((420, 664), (840, 1328)), ((97, 131), (64, 200)), ((100, 80), (100, 57)),
          ((5, 7), (13, 3)), ((1080, 1920), (420, 664)), ((33, 33), (33, 34))]


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    cases = []
    for i, ((h, w), (oh, ow)) in enumerate(RESIZE):
        src = rng.uniform(-0.5, 1.5, (h, w)).astype(np.float32)
        # Smooth content as well as noise: a real matte is mostly smooth with sharp edges
        if i % 2:
            src = cv2.GaussianBlur(src, (0, 0), 3.0)
        dst = cv2.resize(src, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
        src.tofile(out / f"resize{i}_in.f32")
        dst.astype(np.float32).tofile(out / f"resize{i}_out.f32")
        cases.append({"kind": "resize", "id": i, "hw": [h, w], "out_hw": [oh, ow]})
    for i, ch in enumerate((3, 4)):
        src = rng.uniform(-0.5, 2.0, (64, 96, ch)).astype(np.float32)
        dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY if ch == 3 else cv2.COLOR_BGRA2GRAY)
        src.tofile(out / f"gray{i}_in.f32")
        dst.astype(np.float32).tofile(out / f"gray{i}_out.f32")
        cases.append({"kind": "gray", "id": i, "hw": [64, 96], "channels": ch})
    (out / "cases.json").write_text(json.dumps({"opencv": cv2.__version__, "cases": cases}, indent=1))
    print(f"wrote {len(cases)} cases (OpenCV {cv2.__version__}) to {out}")


if __name__ == "__main__":
    main()
