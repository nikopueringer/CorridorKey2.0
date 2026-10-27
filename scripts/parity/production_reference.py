#!/usr/bin/env python3
"""PyTorch reference for the Rust Stage 1 + Stage 2 runner, prepared like the product.

Inputs follow `corridorkey/video_io.py` (sRGB -> linear -> asinh for the stages, display sRGB for
the foundations, centre reflect-101 pad to `shape_quantum`) and `runtime.hann_stage1_stream`
(no hint: `hint_quality = -1`). Foundations run once, as the product does, and their features are
staged for Rust.

Stage 1 + Stage 2 then run in every precision mode worth comparing on this GPU:

- `fp32`: the parity reference (always).
- `product`: bf16 autocast, what `runtime.py` runs.
- `fastest`: the fastest native mode of this GPU: bf16 autocast on compute capability >= 8.0
  (Ampere and newer), fp16 autocast on 7.x (Volta / Turing, where bf16 is emulated: slower than
  fp32, and its math attention fallback runs out of memory sooner), fp32 below.
  Skipped when it is the product mode. Stage 1/2 activations stay far inside fp16's range (max
  |x| 13.7 over all 593 module outputs, measured 2026-10-26), so fp16 is safe for this model.

A mode that runs out of memory is recorded as such instead of aborting the run.

`--free-foundations` drops the backbones once their features are computed and prints
`STAGES_BEGIN`, so `peak_vram.py` measures Stage 1 + Stage 2 alone, like the Rust runner.

usage: production_reference.py <image> <out_dir> [--frames T] [--modes fp32,fastest] [--free-foundations]
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

from corridorkey import color_utils as cu  # noqa: E402
from corridorkey.runtime import RuntimePaths, _build_foundation_inputs, load_model  # noqa: E402
from corridorkey.video_io import ceil_to_multiple, center_pad_to_shape  # noqa: E402

SHAPE_QUANTUM = 128
ITERS = 5


def precision_modes(capability: tuple[int, int]) -> dict[str, torch.dtype | None]:
    """Mode name -> autocast dtype (None = fp32) for a GPU of this compute capability."""
    major, _ = capability
    fastest = torch.bfloat16 if major >= 8 else torch.float16 if major >= 7 else None
    modes = {"fp32": None, "product": torch.bfloat16}
    if fastest not in (None, torch.bfloat16):
        modes["fastest"] = fastest
    return modes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--frames", type=int, default=1, help="window length T (the frame repeated)")
    ap.add_argument("--modes", default=None, help="comma-separated subset of fp32,product,fastest")
    ap.add_argument("--free-foundations", action="store_true",
                    help="free the backbones before the stages (for peak_vram.py)")
    args = ap.parse_args()
    out, t = args.out_dir, args.frames
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    capability = torch.cuda.get_device_capability(dev)

    srgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255.0
    h, w = srgb.shape[:2]
    hp, wp = ceil_to_multiple(h, SHAPE_QUANTUM), ceil_to_multiple(w, SHAPE_QUANTUM)
    display, pad = center_pad_to_shape(srgb, hp, wp)
    model_in = cu.linear_to_asinh(cu.srgb_to_linear(display)).astype(np.float32)

    model = load_model(RuntimePaths.from_package(ROOT), "cuda").to(dev).eval()
    rgb = torch.from_numpy(model_in.transpose(2, 0, 1)).to(dev)[None, None].repeat(1, t, 1, 1, 1)
    found = torch.from_numpy(display.transpose(2, 0, 1)).to(dev)[None, None].repeat(1, t, 1, 1, 1)
    hint = torch.zeros(1, t, 1, hp, wp, device=dev)
    hint_quality = torch.full((1, t), -1.0, device=dev)

    with torch.no_grad():
        rgb_radio, rgb_moge = _build_foundation_inputs(found)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ff = model.foundations(
                rgb_cradio=rgb_radio.reshape(t, 3, *rgb_radio.shape[-2:]),
                rgb_moge=rgb_moge.reshape(t, 3, *rgb_moge.shape[-2:]),
                rgb_rvm=rgb_radio,
            )
        feats = {k: ff[k].float().reshape(1, t, -1, hp // 64, wp // 64)
                 for k in ("cradio", "moge", "rvm") if ff.get(k) is not None}

    if args.free_foundations:
        del model.foundations
        torch.cuda.empty_cache()
        print("STAGES_BEGIN", flush=True)

    def forward():
        s1 = model.stage1(rgb, hint, feats["moge"], cradio_feat=feats.get("cradio"),
                          rvm_feat=feats.get("rvm"), hint_quality=hint_quality, t_stride=1.0)
        s2 = model.stage2(rgb, s1["baton"].float())
        return s1, s2[0] if isinstance(s2, tuple) else s2

    def run_mode(dtype: torch.dtype | None) -> dict:
        autocast = torch.autocast("cuda", dtype=dtype) if dtype else torch.autocast("cuda", enabled=False)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), autocast:
            forward()
            torch.cuda.synchronize()
            s1_ms, s2_ms = [], []
            for _ in range(ITERS):
                ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                ev[0].record()
                s1 = model.stage1(rgb, hint, feats["moge"], cradio_feat=feats.get("cradio"),
                                  rvm_feat=feats.get("rvm"), hint_quality=hint_quality, t_stride=1.0)
                ev[1].record()
                s2 = model.stage2(rgb, s1["baton"].float())
                logits = s2[0] if isinstance(s2, tuple) else s2
                ev[2].record()
                torch.cuda.synchronize()
                s1_ms.append(ev[0].elapsed_time(ev[1]))
                s2_ms.append(ev[1].elapsed_time(ev[2]))
        stage1, stage2 = float(np.median(s1_ms)), float(np.median(s2_ms))
        return {"stage1_ms": stage1, "stage2_ms": stage2, "total_ms": stage1 + stage2,
                "peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20, "logits": logits.float()}

    report = {
        "device": torch.cuda.get_device_name(dev),
        "compute_capability": list(capability),
        "torch": torch.__version__,
        "h": hp, "w": wp, "t": t, "pad": list(pad), "native_hw": [h, w],
        "modes": {},
    }
    modes = precision_modes(capability)
    if args.modes:
        wanted = set(args.modes.split(","))
        modes = {k: v for k, v in modes.items() if k in wanted}
    for name, dtype in modes.items():
        label = str(dtype).removeprefix("torch.") if dtype else "float32"
        try:
            result = run_mode(dtype)
        except torch.OutOfMemoryError as e:
            report["modes"][name] = {"dtype": label, "oom": str(e).split(".")[0]}
            torch.cuda.empty_cache()
            print(f"{name:8} ({label}): out of memory")
            continue
        logits = result.pop("logits")
        logits.permute(0, 1, 3, 4, 2).contiguous().cpu().numpy().astype(np.float32).tofile(
            out / f"torch_{name}_logits_bthwc.f32")
        report["modes"][name] = {"dtype": label, **result}
        print(f"{name:8} ({label}): stage 1 {result['stage1_ms']:.1f} ms, stage 2 {result['stage2_ms']:.1f} ms, "
              f"total {result['total_ms']:.1f} ms, peak {result['peak_alloc_mb']:.0f} MB")
        del logits

    vfm = torch.cat([feats[k] for k in ("cradio", "moge", "rvm") if k in feats], dim=2)
    staged = {
        "rgb_hint_bthwc": torch.cat([rgb, hint], 2),
        "rgb_bthwc": rgb,
        "vfm_64_bthwc": vfm,
    }
    for name, x in staged.items():
        x.permute(0, 1, 3, 4, 2).contiguous().cpu().numpy().astype(np.float32).tofile(out / f"{name}.f32")
    hint_quality.reshape(-1, 1).cpu().numpy().astype(np.float32).tofile(out / "hint_quality_bt.f32")
    (out / "meta.json").write_text(json.dumps({"b": 1, "t": t, "h": hp, "w": wp, "vfm_channels": vfm.shape[2]}))
    (out / "torch_ref.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
