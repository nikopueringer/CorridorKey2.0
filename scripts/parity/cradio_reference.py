#!/usr/bin/env python3
"""fp32 reference of C-RADIOv4-SO400M as CorridorKey runs it, for
`crates/corridorkey-ml/tests/parity_cradio.rs`.

Builds the model the way `FrozenFoundations` does (`build_cradio_so400m` on the bundled snapshot),
feeds the product's input (`_build_foundation_inputs`: area resize of the display frame to
H/4 x W/4, two different frames) and dumps, channels-last little-endian f32:

- `input_nhwc`        [N, He, We, 3]
- `pos_embed`         [rows * cols, 1152]     the CPE position table for the token grid
- `tokens`            [N, skip + rows*cols, 1152]   patch generator output (prefix + patches)
- `block_<i>`         [N, skip + rows*cols, 1152]   outputs of blocks 0, 13 and 26
- `normalized_nhwc`   [N, rows, cols, 1152]   `.features` after the cradio mean / std

usage: cradio_reference.py <out_dir> [--height 512 --width 768]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from corridorkey.foundations import build_cradio_so400m  # noqa: E402
from corridorkey.runtime import _build_foundation_inputs  # noqa: E402
from corridorkey.video_io import center_pad_to_shape  # noqa: E402

DUMP_BLOCKS = (0, 13, 26)


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
    frames = np.stack([display, display[:, ::-1]])
    found = torch.from_numpy(frames.transpose(0, 3, 1, 2).copy()).to(dev)[None]
    rgb_radio, _ = _build_foundation_inputs(found)
    rgb = rgb_radio[0]  # [N, 3, He, We]
    n, _, he, we = rgb.shape

    radio = build_cradio_so400m(repo=str(ROOT / "vfms" / "cradio" / "C-RADIOv4-SO400M"), device="cuda")
    rm = getattr(radio, "radio_model", radio)
    vit = rm.model
    pg = vit.patch_generator
    stats = load_file(ROOT / "weights" / "foundation_stats.safetensors")
    rows, cols = he // pg.patch_size, we // pg.patch_size

    with torch.no_grad():
        x = rm.input_conditioner(rgb)
        tokens = pg(x)
        pos = pg._get_pos_embeddings(n, (rows, cols)).reshape(-1, pos_dim := tokens.shape[-1])
        h = tokens
        blocks = {}
        for i, blk in enumerate(vit.blocks):
            h = blk(h)
            if i in DUMP_BLOCKS:
                blocks[i] = h
        # As `FrozenFoundations.forward`: the HF wrapper may not take `feature_fmt`; then the
        # features come back as [N, L, C] and are reshaped to the token grid.
        try:
            features = radio(rgb, feature_fmt="NCHW").features
        except TypeError:
            features = radio(rgb).features
        if features.dim() == 3:
            features = features.transpose(1, 2).reshape(n, -1, rows, cols)
        features = features.float()  # [N, 1152, rows, cols]
        mean = torch.from_numpy(stats["cradio_mean"]).to(dev).view(1, -1, 1, 1)
        std = torch.from_numpy(stats["cradio_std"]).to(dev).view(1, -1, 1, 1).clamp(min=1e-6)
        normalized = (features - mean) / std

    dump(out, "input_nhwc", rgb.permute(0, 2, 3, 1))
    dump(out, "pos_embed", pos)
    dump(out, "tokens", tokens)
    for i, b in blocks.items():
        dump(out, f"block_{i}", b)
    dump(out, "normalized_nhwc", normalized.permute(0, 2, 3, 1))
    meta = {"n": n, "he": he, "we": we, "rows": rows, "cols": cols, "dim": pos_dim,
            "skip": pg.num_skip, "depth": len(vit.blocks), "heads": vit.blocks[0].attn.num_heads,
            "torch": torch.__version__}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
