#!/usr/bin/env python3
"""Layer-by-layer fp32 reference of the MoGe-2 encoder as CorridorKey runs it, for
`crates/corridorkey-ml/tests/parity_moge.rs`.

Builds MoGe the way `FrozenFoundations` does (`build_moge_vitb` on `vfms/moge/model.pt`), feeds it the
product's input (`_build_foundation_inputs`: area resize of the display frame to 7H/32 x 7W/32, here
two different frames so batching is exercised) and dumps, channels-last little-endian f32:

- `input_nhwc`            [N, Hm, Wm, 3]          display sRGB at 7H/32 x 7W/32
- `pos_embed`             [1 + h*w, 768]          position embedding interpolated to the token grid
- `tokens`                [N, 1 + h*w, 768]       after patch embedding + position embedding
- `block_5`, `block_11`   [N, 1 + h*w, 768]       block outputs (before the final norm)
- `features_nhwc`         [N, h, w, 768]          encoder output (summed projections)
- `normalized_nhwc`       [N, h, w, 768]          after the moge mean / std normalisation

usage: moge_reference.py <out_dir> [--height 512 --width 768]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "MoGe"))

from corridorkey.foundations import build_moge_vitb  # noqa: E402
from corridorkey.runtime import _build_foundation_inputs  # noqa: E402
from corridorkey.video_io import center_pad_to_shape  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402


def dump(out: Path, name: str, x: torch.Tensor) -> None:
    x.detach().float().cpu().contiguous().numpy().astype(np.float32).tofile(out / f"{name}.f32")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    args = ap.parse_args()
    out, hp, wp = args.out_dir, args.height, args.width
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    srgb = np.asarray(Image.open(ROOT / "data" / "green_screen_portrait.png").convert("RGB"), np.float32) / 255
    display, _ = center_pad_to_shape(srgb, hp, wp)
    frames = np.stack([display, display[:, ::-1]])  # two different frames
    found = torch.from_numpy(frames.transpose(0, 3, 1, 2).copy()).to(dev)[None]  # [1, 2, 3, H, W]
    _, rgb_moge = _build_foundation_inputs(found)
    rgb_moge = rgb_moge[0]  # [N, 3, Hm, Wm]
    n, _, hm, wm = rgb_moge.shape
    rows, cols = hm // 14, wm // 14

    moge = build_moge_vitb(device="cuda", checkpoint=str(ROOT / "vfms" / "moge" / "model.pt"))
    enc = moge.encoder
    vit = enc.backbone
    stats = load_file(ROOT / "weights" / "foundation_stats.safetensors")

    with torch.no_grad():
        image_14 = torch.nn.functional.interpolate(
            rgb_moge, (rows * 14, cols * 14), mode="bilinear", align_corners=False, antialias=True)
        assert torch.equal(image_14, rgb_moge), "7H/32 x 7W/32 should already be a multiple of 14"
        x = (image_14 - enc.image_mean) / enc.image_std
        tokens = vit.prepare_tokens_with_masks(x)
        pos = vit.interpolate_pos_encoding(tokens, rows * 14, cols * 14)[0]
        taken = {}
        h = tokens
        for i, blk in enumerate(vit.blocks):
            h = blk(h)
            if i in enc.intermediate_layers:
                taken[i] = h
        features = enc(rgb_moge, rows, cols)  # [N, 768, rows, cols]
        mean = torch.from_numpy(stats["moge_mean"]).to(dev).view(1, -1, 1, 1)
        std = torch.from_numpy(stats["moge_std"]).to(dev).view(1, -1, 1, 1)
        normalized = (features.float() - mean) / std.clamp(min=1e-6)

    dump(out, "input_nhwc", rgb_moge.permute(0, 2, 3, 1))
    dump(out, "pos_embed", pos)
    dump(out, "tokens", tokens)
    for i, t in taken.items():
        dump(out, f"block_{i}", t)
    dump(out, "features_nhwc", features.permute(0, 2, 3, 1))
    dump(out, "normalized_nhwc", normalized.permute(0, 2, 3, 1))
    meta = {"n": n, "hm": hm, "wm": wm, "rows": rows, "cols": cols, "dim": vit.embed_dim,
            "layers": list(enc.intermediate_layers), "torch": torch.__version__}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
