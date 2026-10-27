#!/usr/bin/env python3
"""Reference data and PyTorch CUDA timings for Stage 2 (`corridorkey/stage2.py`).

Loads `stage2.*` weights directly from `weights/corridorkey_v2.safetensors`,
generates deterministic NHWC inputs (`rgb_nhwc` and `baton_nhwc`) at `(H=384, W=640)`
(so `/16` is `24×40`, exercising non-multiple-of-16 Swin window padding and shifted
region groups), writes raw little-endian f32 files, and records PyTorch CUDA latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from corridorkey.stage2 import Stage2  # noqa: E402


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
    ap.add_argument("--h", type=int, default=384)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt_path = ROOT / "weights/corridorkey_v2.safetensors"
    with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
        header = f.metadata() or {}
        cfg = json.loads(header["model_config"])
        state_dict = {
            k[len("stage2."):]: f.get_tensor(k)
            for k in f.keys()
            if k.startswith("stage2.")
        }

    stage2 = Stage2(
        d=cfg["d_stage2"],
        d_baton=cfg["d_16"],
        num_heads=cfg["num_heads_stage2"],
        num_blocks=cfg["n_blocks_stage2"],
        num_final_blocks=cfg["n_final_blocks_stage2"],
        window_size=cfg["window_size_stage2"],
        mlp_mult=cfg.get("mlp_mult", 2.5),
        overlap_patchify=cfg.get("overlap_patchify", False),
        overlap_mult=cfg.get("overlap_mult", 2),
        depatch_fourier_features=cfg.get("depatch_fourier_features", 0),
        depatch_fourier_kernel=cfg.get("depatch_fourier_kernel", 3),
        local_mix_kernel=cfg.get("stage2_local_mix_kernel", 0),
        mlp_dw_kernel=cfg.get("mlp_dw_kernel", 0),
        qkv_dw_kernel=cfg.get("qkv_dw_kernel", 0),
        qkv_dw_fraction=cfg.get("qkv_dw_fraction", 1.0),
        refine4_blocks=cfg.get("stage2_refine4_blocks", 0),
        refine4_width=cfg.get("stage2_refine4_width", 96),
        refine4_token_channels=cfg.get("stage2_refine4_token_channels", 48),
        refine4_kernel=cfg.get("stage2_refine4_kernel", 7),
        refine4_expansion=cfg.get("stage2_refine4_expansion", 2.0),
        rope_base=cfg.get("rope_base", 100.0),
    )
    stage2.load_state_dict(state_dict, strict=True)
    stage2.eval().cuda()

    h, w = args.h, args.w
    h16, w16 = h // 16, w // 16
    d_baton = cfg["d_16"]

    rng = np.random.default_rng(42)
    rgb_nhwc = rng.uniform(-0.25, 0.25, (1, h, w, 3)).astype(np.float32)
    baton_nhwc = rng.uniform(-0.15, 0.15, (1, h16, w16, d_baton)).astype(np.float32)

    rgb_nhwc.tofile(args.out / "rgb_nhwc.f32")
    baton_nhwc.tofile(args.out / "baton_nhwc.f32")

    # PyTorch Stage2 expects NTHW channels-first: rgb (B, T, 3, H, W), baton (B, T, d_baton, H/16, W/16)
    rgb_pt = torch.from_numpy(rgb_nhwc).permute(0, 3, 1, 2).unsqueeze(1).contiguous().cuda()
    baton_pt = torch.from_numpy(baton_nhwc).permute(0, 3, 1, 2).unsqueeze(1).contiguous().cuda()

    with torch.no_grad():
        out_pt = stage2(rgb_pt, baton_pt)  # (1, 1, 4, H, W)
        out_nhwc = out_pt.squeeze(1).permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
        out_nhwc.tofile(args.out / "out_nhwc_torch.f32")

        resident_ms = cuda_median_ms(lambda: stage2(rgb_pt, baton_pt), args.iters)

    meta = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "h": h,
        "w": w,
        "d_baton": d_baton,
        "resident_ms": resident_ms,
    }
    (args.out / "stage2_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Stage2 ({h}x{w}) torch cuda resident: {resident_ms:.3f} ms")


if __name__ == "__main__":
    main()
