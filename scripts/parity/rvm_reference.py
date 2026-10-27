#!/usr/bin/env python3
"""fp32 reference of RVM-small as CorridorKey runs it, for `crates/corridorkey-ml/tests/parity_rvm.rs`.

Input as `runtime.precompute_rvm_to_memmap` prepares it: the display frame resized bilinearly to
H/4 x W/4 (not the area resize C-RADIO gets) and clamped to [0, 1]. Frames run in sequence with
the recurrent state carried from frame to frame, starting from zeros. Dumps, little-endian f32:

- `input_nhwc`       [T, He, We, 3]
- `encoded`          [T, 1 + h*w, 384]   encoder output (after its final norm), per frame
- `state_<t>`        [1 + h*w, 384]      recurrent state after frame t
- `normalized_nhwc`  [T, h, w, 384]      features (state without CLS) after rvm mean / std

usage: rvm_reference.py <out_dir> [--height 512 --width 768 --frames 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from corridorkey.rvm import RVMVideoSiamMAE, load_rvm_npz  # noqa: E402
from corridorkey.video_io import center_pad_to_shape  # noqa: E402


def dump(out: Path, name: str, x: torch.Tensor) -> None:
    x.detach().float().cpu().contiguous().numpy().astype(np.float32).tofile(out / f"{name}.f32")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()
    out, hp, wp, t = args.out_dir, args.height, args.width, args.frames
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    srgb = np.asarray(Image.open(ROOT / "data" / "green_screen_portrait.png").convert("RGB"), np.float32) / 255
    display, _ = center_pad_to_shape(srgb, hp, wp)
    # Different frames: the image, mirrored, and shifted, so the recurrence sees motion.
    frames = [display, display[:, ::-1], np.roll(display, 40, axis=1)][:t]
    rgb = torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2).copy()).to(dev)
    he, we = hp // 4, wp // 4
    rgb_rvm = F.interpolate(rgb, size=(he, we), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    model = RVMVideoSiamMAE(variant="small").to(dev).eval()
    load_rvm_npz(model, str(next((ROOT / "vfms" / "rvm").glob("*.npz"))), strict_load=True, verbose=False)
    stats = load_file(ROOT / "weights" / "foundation_stats.safetensors")
    mean = torch.from_numpy(stats["rvm_mean"]).to(dev)
    std = torch.from_numpy(stats["rvm_std"]).to(dev).clamp(min=1e-6)

    with torch.no_grad():
        encoded = model._run_encoder(model._tokenize(rgb_rvm))  # [T, N, D]
        state = encoded.new_zeros(1, encoded.shape[1], model.embed_dim)
        feats = []
        for i in range(t):
            o, state = model.rnn_core(encoded[i:i + 1], state)
            dump(out, f"state_{i}", state[0])
            feats.append(o[0, 1:])
        h, w = he // 16, we // 16
        normalized = ((torch.stack(feats) - mean) / std).view(t, h, w, model.embed_dim)

    dump(out, "input_nhwc", rgb_rvm.permute(0, 2, 3, 1))
    dump(out, "encoded", encoded)
    dump(out, "normalized_nhwc", normalized)
    meta = {"t": t, "he": he, "we": we, "rows": h, "cols": w, "dim": model.embed_dim, "torch": torch.__version__}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
