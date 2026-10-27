#!/usr/bin/env python3
"""Reference of the product's output rendering, for `crates/corridorkey-base/tests/parity_render.rs`.

Runs `runtime._render_predictions_torch` (video: uint16 BGR / BGRA) and
`runtime._render_predictions_exr_torch` (EXR: float32) on random Stage 2 logits with uneven
padding, a linear adjustment and despill 0.5, for every output and both cutout modes. Dumps:

- `logits_nhwc.f32`               [T, H, W, 4]   alpha logit, then asinh fg RGB (padded size)
- `video_<out>[_linear].u16`      [T, h, w, C]   C = 3 (alpha, fg, checker) or 4 (cutout)
- `exr_<out>.f32`                 [T, h, w, C]   C = 1 (alpha), 3 (fg, checker), 4 (cutout)

usage: render_reference.py <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from corridorkey.runtime import _render_predictions_exr_torch, _render_predictions_torch  # noqa: E402
from corridorkey.video_io import make_checker  # noqa: E402

T, H, W = 2, 256, 384
NATIVE = (205, 301)
PAD = (40, 43, 25, 26)  # left, right, top, bottom
OUTPUTS = ("alpha", "fg", "cutout", "checker")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    alpha = rng.uniform(-0.2, 1.2, size=(T, 1, H, W))
    fg = rng.uniform(-0.3, 1.6, size=(T, 3, H, W))
    logits = torch.from_numpy(np.concatenate([alpha, fg], axis=1).astype(np.float32))
    meta = {
        "model_native_hw": NATIVE, "output_hw": NATIVE, "pad": PAD,
        "linear_brightness": 1.2, "linear_contrast": 0.85, "linear_contrast_pivot": 0.18,
        "despill_strength": 0.5,
    }
    checker = torch.from_numpy(make_checker(*NATIVE))
    logits.permute(0, 2, 3, 1).contiguous().numpy().tofile(out / "logits_nhwc.f32")
    for linear in (False, True):
        r = _render_predictions_torch({**meta, "cutout_linear": linear}, set(OUTPUTS), checker, logits)
        for name in OUTPUTS:
            if linear and name != "cutout":
                continue
            suffix = "_linear" if linear else ""
            np.ascontiguousarray(r[name]).astype("<u2").tofile(out / f"video_{name}{suffix}.u16")
    r = _render_predictions_exr_torch(meta, set(OUTPUTS), checker, logits)
    for name in OUTPUTS:
        np.ascontiguousarray(r[name]).astype("<f4").tofile(out / f"exr_{name}.f32")
    info = {"t": T, "h": H, "w": W, "native_hw": NATIVE, "pad": PAD,
            "linear_brightness": 1.2, "linear_contrast": 0.85, "linear_contrast_pivot": 0.18, "despill": 0.5,
            "shapes": {name: list(np.asarray(r[name]).shape) for name in OUTPUTS}}
    (out / "meta.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
