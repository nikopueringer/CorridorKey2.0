#!/usr/bin/env python3
"""Reference of the product's frame preprocessing, for `crates/corridorkey-base/tests/parity_frame.rs`.

Uses the product's own functions (`corridorkey/video_io.py`, `color_utils.py`,
`runtime._build_foundation_inputs`, `runtime.precompute_rvm_to_memmap`'s resize) on random 8-bit RGB
frames of an odd size, with a non-identity linear adjustment, in f32 (the product then rounds the
display and model views to bf16; the reference does not). Dumps little-endian files:

- `frames_u8`     [T, h0, w0, 3]  uint8 RGB as decoded
- `display_nhwc`     [T, H, W, 3]    sRGB in [0, 1], reflect-101 centre-padded to the shape quantum
- `model_nhwc`       [T, H, W, 3]    asinh(adjust(srgb_to_linear(display)))
- `foundation_nhwc`  [T, H, W, 3]    the backbones' view: `linear_to_vfm_sdr(adjust(linear))` when an
                                     adjustment is set, else the display (`preprocess_video_frame_views`)
- `cradio_nhwc`      [T, H/4, W/4, 3]      area resize of the foundation view (C-RADIO input)
- `moge_nhwc`        [T, 7H/32, 7W/32, 3]  area resize of the foundation view (MoGe input)
- `rvm_nhwc`         [T, H/4, W/4, 3]      bilinear resize of the foundation view, clamped (RVM input)

usage: frame_reference.py <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from corridorkey import color_utils as cu  # noqa: E402
from corridorkey.runtime import _build_foundation_inputs  # noqa: E402
from corridorkey.video_io import apply_linear_adjust, ceil_to_multiple, center_pad_to_shape  # noqa: E402

T, H0, W0, QUANTUM = 2, 421, 667, 128
ADJUST = {"linear_brightness": 1.3, "linear_contrast": 0.9, "linear_contrast_pivot": 0.18}


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261026)
    frames = rng.integers(0, 256, size=(T, H0, W0, 3), dtype=np.uint8)
    h, w = ceil_to_multiple(H0, QUANTUM), ceil_to_multiple(W0, QUANTUM)

    display, model, foundation = [], [], []
    for f in frames:
        d, pad = center_pad_to_shape(f.astype(np.float32) / 255.0, h, w)
        lin = apply_linear_adjust(cu.srgb_to_linear(d), ADJUST)
        display.append(d)
        model.append(cu.linear_to_asinh(lin).astype(np.float32))
        foundation.append(cu.linear_to_vfm_sdr(lin).astype(np.float32))
    display = np.stack(display).astype(np.float32)
    model = np.stack(model)
    foundation = np.stack(foundation)

    found_t = torch.from_numpy(foundation.transpose(0, 3, 1, 2).copy())[None]  # [1, T, 3, H, W]
    radio, moge = _build_foundation_inputs(found_t)
    rvm = F.interpolate(found_t[0], size=(h // 4, w // 4), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    nhwc = lambda x: x.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)  # noqa: E731
    frames.tofile(out / "frames_u8.u8")
    display.tofile(out / "display_nhwc.f32")
    model.tofile(out / "model_nhwc.f32")
    foundation.tofile(out / "foundation_nhwc.f32")
    nhwc(radio[0]).tofile(out / "cradio_nhwc.f32")
    nhwc(moge[0]).tofile(out / "moge_nhwc.f32")
    nhwc(rvm).tofile(out / "rvm_nhwc.f32")
    meta = {"t": T, "h0": H0, "w0": W0, "h": h, "w": w, "pad": list(pad), "moge_hw": list(moge.shape[-2:]), **ADJUST}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
