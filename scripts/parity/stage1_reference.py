#!/usr/bin/env python3
"""Generate Stage 1 reference inputs, outputs, and PyTorch CUDA resident timing from
weights/corridorkey_v2.safetensors for parity testing against corridorkey-ml.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from corridorkey.stage1 import Stage1  # noqa: E402


def load_safetensors_stage1(ckpt_path: Path):
    with open(ckpt_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_offset = 8 + header_len

        meta = header["__metadata__"]
        cfg = json.loads(meta["model_config"])
        state_dict = {}
        for k, info in header.items():
            if k == "__metadata__" or not k.startswith("stage1."):
                continue
            start, end = info["data_offsets"]
            f.seek(data_offset + start)
            raw = f.read(end - start)
            arr = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"]).copy()
            state_dict[k[len("stage1.") :]] = torch.from_numpy(arr)
    return cfg, state_dict


def main() -> None:
    out_dir = Path("/tmp/corridorkey-parity-stage1")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ROOT / "weights" / "corridorkey_v2.safetensors"

    cfg, state_dict = load_safetensors_stage1(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rvm_dim = 384 if cfg.get("rvm_variant") != "none" else 0
    moge_dim = 768 if cfg.get("use_moge", True) else 0
    cradio_dim = 1152 if cfg.get("use_cradio", False) else 0

    model = Stage1(
        d_16=cfg["d_16"],
        d_64=cfg["d_64"],
        cradio_dim=cradio_dim,
        moge_dim=moge_dim,
        rvm_dim=rvm_dim,
        n_enc_16=cfg["n_enc_16"],
        n_body_64=cfg["n_body_64"],
        n_dec_16=cfg["n_dec_16"],
        num_heads_16=cfg["num_heads_16"],
        num_heads_64=cfg["num_heads_64"],
        window_HW=cfg["window_HW"],
        local_mix_kernel=cfg["stage1_local_mix_kernel"],
        local_mix_init_std=cfg["stage1_local_mix_init_std"],
        body64_local_mix_kernel=cfg["stage1_body64_local_mix_kernel"],
        body64_local_mix_init_std=cfg["stage1_body64_local_mix_init_std"],
        body64_mlp_hidden=cfg["stage1_body64_mlp_hidden"],
        body64_direct_concat=cfg["stage1_body64_direct_concat"],
        recurrent_64_sites=tuple(cfg.get("stage1_recurrent_64_sites") or ()),
        hint_adaln_blocks=cfg["stage1_hint_adaln_blocks"],
        mid_16_blocks=cfg["stage1_mid_16_blocks"],
        mlp_dw_kernel=cfg["mlp_dw_kernel"],
        qkv_dw_kernel=cfg["qkv_dw_kernel"],
        qkv_dw_fraction=cfg["qkv_dw_fraction"],
        mlp_mult=cfg["mlp_mult"],
        overlap_patchify=cfg["overlap_patchify"],
        overlap_mult=cfg["overlap_mult"],
        depatch_fourier_features=cfg["depatch_fourier_features"],
        depatch_fourier_kernel=cfg["depatch_fourier_kernel"],
        enable_mae_mask_tokens=cfg["enable_mae_mask_tokens"],
        bottleneck_head_stride=cfg["bottleneck_head_stride"],
        rope_base=cfg["rope_base"],
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")
    model.eval().to(device)

    # 192x320 (/16 = 12x20, non-multiple of window_HW=16; /64 = 3x5) across T=2 frames
    b, t, h, w = 1, 2, 192, 320
    h64, w64 = h // 64, w // 64
    rng = np.random.default_rng(20261023)
    rgb_np = rng.standard_normal((b, t, 3, h, w), dtype=np.float32) * 0.25
    hint_np = rng.uniform(0.0, 1.0, size=(b, t, 1, h, w)).astype(np.float32)
    cradio_np = rng.standard_normal((b, t, cradio_dim, h64, w64), dtype=np.float32) * 0.2
    moge_np = rng.standard_normal((b, t, moge_dim, h64, w64), dtype=np.float32) * 0.2
    rvm_np = rng.standard_normal((b, t, rvm_dim, h64, w64), dtype=np.float32) * 0.2
    hq_np = np.array([[1.0, 0.0]], dtype=np.float32)

    rgb = torch.from_numpy(rgb_np).to(device)
    hint = torch.from_numpy(hint_np).to(device)
    cradio = torch.from_numpy(cradio_np).to(device) if cradio_dim > 0 else None
    moge = torch.from_numpy(moge_np).to(device)
    rvm = torch.from_numpy(rvm_np).to(device) if rvm_dim > 0 else None
    hq = torch.from_numpy(hq_np).to(device)

    # Register intermediate hooks to isolate layer-by-layer parity
    intermediates = {}
    def hook(name):
        def fn(m, inp, out):
            if isinstance(out, torch.Tensor):
                intermediates[name] = out.detach().cpu()
            elif isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                intermediates[name] = out[0].detach().cpu()
        return fn

    model.stem.register_forward_hook(hook("stem"))
    for i, blk in enumerate(model.enc_16):
        blk.register_forward_hook(hook(f"enc_16_{i}"))
    model.down.register_forward_hook(hook("down"))
    if model.fuse is not None:
        model.fuse.register_forward_hook(hook("fuse"))
    for i, blk in enumerate(model.body_64):
        blk.register_forward_hook(hook(f"body_64_{i}"))
    model.mid_up.register_forward_hook(hook("mid_up"))
    for i, blk in enumerate(model.mid_16):
        blk.register_forward_hook(hook(f"mid_16_{i}"))
    model.mid_down.register_forward_hook(hook("mid_down"))
    model.up.register_forward_hook(hook("up"))
    for i, blk in enumerate(model.dec_16):
        blk.register_forward_hook(hook(f"dec_16_{i}"))

    with torch.no_grad():
        for _ in range(3):
            out = model(
                rgb, hint, moge, cradio_feat=cradio, rvm_feat=rvm, hint_quality=hq, t_stride=1.0
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
            times = []
            for _ in range(10):
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                out = model(
                    rgb,
                    hint,
                    moge,
                    cradio_feat=cradio,
                    rvm_feat=rvm,
                    hint_quality=hq,
                    t_stride=1.0,
                )
                ev1.record()
                torch.cuda.synchronize()
                times.append(ev0.elapsed_time(ev1))
            times.sort()
            resident_ms = float(times[len(times) // 2])
        else:
            resident_ms = 0.0

    # Save channels-last BTHWC tensors for Rust
    rgb_hint_bthwc = (
        np.concatenate([rgb_np, hint_np], axis=2).transpose(0, 1, 3, 4, 2).astype(np.float32)
    )
    vfm_parts = [p for p in (cradio_np, moge_np, rvm_np) if p.shape[2] > 0]
    vfm_64_bthwc = (
        np.concatenate(vfm_parts, axis=2).transpose(0, 1, 3, 4, 2).astype(np.float32)
    )
    baton_bthwc = out["baton"].detach().cpu().permute(0, 1, 3, 4, 2).contiguous().numpy()
    aux_bthwc = out["aux"].detach().cpu().permute(0, 1, 3, 4, 2).contiguous().numpy()

    (out_dir / "rgb_hint_bthwc.f32").write_bytes(rgb_hint_bthwc.tobytes())
    (out_dir / "vfm_64_bthwc.f32").write_bytes(vfm_64_bthwc.tobytes())
    (out_dir / "hint_quality_bt.f32").write_bytes(hq_np.reshape(-1, 1).tobytes())
    (out_dir / "baton_bthwc_torch.f32").write_bytes(baton_bthwc.tobytes())
    (out_dir / "aux_bthwc_torch.f32").write_bytes(aux_bthwc.tobytes())

    for k, tensor in intermediates.items():
        arr = tensor.numpy()
        # Convert to BTHWC if it's 5D (B, T, C, H, W) or 4D (BT, C, H, W) or (B, T, H, W, C)
        if arr.ndim == 5:
            # Check if channels are at dim 2 (B, T, C, H, W) or dim 4 (B, T, H, W, C)
            if arr.shape[2] in (cfg["d_16"], cfg["d_64"], cfg.get("d_down_aux", 256)):
                arr = arr.transpose(0, 1, 3, 4, 2)
        elif arr.ndim == 4:
            # (BT, C, H, W) -> (B, T, H, W, C)
            arr = arr.reshape(b, t, arr.shape[1], arr.shape[2], arr.shape[3]).transpose(0, 1, 3, 4, 2)
        (out_dir / f"{k}_bthwc.f32").write_bytes(arr.astype(np.float32).tobytes())

    summary = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "b": b,
        "t": t,
        "h": h,
        "w": w,
        "d_16": cfg["d_16"],
        "d_64": cfg["d_64"],
        "vfm_channels": moge_dim + rvm_dim + cradio_dim,
        "resident_ms": resident_ms,
    }
    (out_dir / "stage1_meta.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
