#!/usr/bin/env python3
"""Run end-to-end PyTorch and Rust inference on a green screen image and

generate an 8-panel visual comparison grid PNG.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from corridorkey import color_utils as cu
from corridorkey.runtime import InferenceConfig, RuntimePaths, load_model
from corridorkey.video_io import ceil_to_multiple, make_checker


def pad_to_multiple(img_np: np.ndarray, quantum: int = 128):
    """img_np: [H, W, C] in [0, 1]. Returns (padded_bchw, pad_tuple)."""
    h, w, c = img_np.shape
    h_out = ceil_to_multiple(h, quantum)
    w_out = ceil_to_multiple(w, quantum)
    pad_l = (w_out - w) // 2
    pad_r = w_out - w - pad_l
    pad_t = (h_out - h) // 2
    pad_b = h_out - h - pad_t

    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    padded = F.pad(t, (pad_l, pad_r, pad_t, pad_b), mode="reflect")
    return padded, (pad_l, pad_r, pad_t, pad_b), (h, w)


def unpad(img_np: np.ndarray, pad: tuple[int, int, int, int], orig_hw: tuple[int, int]):
    pad_l, pad_r, pad_t, pad_b = pad
    h, w = orig_hw
    return img_np[pad_t : pad_t + h, pad_l : pad_l + w]


def main():
    root = Path(__file__).resolve().parent.parent
    img_path = root / "data" / "green_screen_portrait.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Missing {img_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading image {img_path}...")
    pil_in = Image.open(img_path).convert("RGB")
    rgb_orig = np.array(pil_in, dtype=np.float32) / 255.0  # [H, W, 3]
    h_orig, w_orig, _ = rgb_orig.shape

    padded_t, pad, (h_orig, w_orig) = pad_to_multiple(rgb_orig, quantum=128)
    _, _, H_pad, W_pad = padded_t.shape
    print(f"Native: {w_orig}x{h_orig} -> Padded: {W_pad}x{H_pad} (pad={pad})")

    # Load CorridorKey runtime model in PyTorch
    paths = RuntimePaths.from_package(root)
    print("Loading PyTorch CorridorKey model and foundations...")
    model = load_model(paths, str(device)).to(device)
    model.eval()

    # Prepare foundation inputs
    print("Preparing foundation inputs (C-RADIO, MoGe, RVM)...")
    rgb_dev = padded_t.to(device)  # [1, 3, H, W]
    rgb_5d = rgb_dev.unsqueeze(1)  # [1, 1, 3, H, W]
    hint = torch.zeros(1, 1, 1, H_pad, W_pad, device=device)
    hq = torch.zeros(1, 1, device=device)

    rgb_cradio = F.interpolate(rgb_dev, size=(H_pad // 4, W_pad // 4), mode="area").unsqueeze(1)
    rgb_moge = F.interpolate(rgb_dev, size=(7 * H_pad // 32, 7 * W_pad // 32), mode="area").unsqueeze(1)

    print("Running PyTorch end-to-end model...")
    with torch.no_grad():
        out_dict = model(rgb_5d, hint, rgb_cradio, rgb_moge, hint_quality=hq)
        alpha_torch = out_dict["alpha"][0, 0, 0].clamp(0.0, 1.0).cpu().numpy()
        fg_torch = out_dict["fg"][0, 0].permute(1, 2, 0).cpu().numpy()

        # Extract normalized foundations for Rust
        foundation_feats = model.foundations(
            rgb_moge=rgb_moge.reshape(1, 3, *rgb_moge.shape[-2:]),
            rgb_cradio=rgb_cradio.reshape(1, 3, *rgb_cradio.shape[-2:]),
            rgb_rvm=rgb_cradio.reshape(1, 1, 3, *rgb_cradio.shape[-2:]),
        )
        vfm_parts = []
        if "cradio" in foundation_feats and foundation_feats["cradio"] is not None:
            vfm_parts.append(foundation_feats["cradio"].reshape(1, 1, -1, H_pad // 64, W_pad // 64))
        if "moge" in foundation_feats and foundation_feats["moge"] is not None:
            vfm_parts.append(foundation_feats["moge"].reshape(1, 1, -1, H_pad // 64, W_pad // 64))
        if "rvm" in foundation_feats and foundation_feats["rvm"] is not None:
            vfm_parts.append(foundation_feats["rvm"].reshape(1, 1, -1, H_pad // 64, W_pad // 64))
        vfm_cat = torch.cat(vfm_parts, dim=2)  # [1, 1, C_vfm, H/64, W/64]

    # Save inputs for Rust
    rust_in_dir = Path("/tmp/greenscreen_test/for_rust")
    rust_out_dir = Path("/tmp/greenscreen_test/from_rust")
    rust_in_dir.mkdir(parents=True, exist_ok=True)
    rust_out_dir.mkdir(parents=True, exist_ok=True)

    # Save meta.json
    vfm_c = vfm_cat.shape[2]
    meta = {
        "b": 1,
        "t": 1,
        "h": H_pad,
        "w": W_pad,
        "vfm_channels": vfm_c,
    }
    (rust_in_dir / "meta.json").write_text(json.dumps(meta))

    rgb_hint_bthwc = (
        torch.cat([rgb_5d, hint], dim=2)
        .permute(0, 1, 3, 4, 2)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    rgb_bthwc = (
        rgb_5d.permute(0, 1, 3, 4, 2)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    vfm_64_bthwc = (
        vfm_cat.permute(0, 1, 3, 4, 2)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    (rust_in_dir / "rgb_hint_bthwc.f32").write_bytes(rgb_hint_bthwc.tobytes())
    (rust_in_dir / "rgb_bthwc.f32").write_bytes(rgb_bthwc.tobytes())
    (rust_in_dir / "vfm_64_bthwc.f32").write_bytes(vfm_64_bthwc.tobytes())
    (rust_in_dir / "hint_quality_bt.f32").write_bytes(hq.cpu().numpy().astype(np.float32).tobytes())

    # CRITICAL: Free PyTorch VRAM before running Rust!
    # PyTorch and Rust cannot both hold 3 GB models simultaneously on an 8 GB GPU.
    print("Freeing PyTorch VRAM before running Rust engine...")
    del model, rgb_dev, rgb_5d, hint, hq, rgb_cradio, rgb_moge, vfm_cat
    torch.cuda.empty_cache()

    # Step 4: Run Rust Stage 1 + Stage 2 binary
    print("Executing Rust Stage 1 + Stage 2 (CubeCL / Burn CUDA)...")
    cmd = [
        "cargo", "run", "--release", "--features", "cuda",
        "--bin", "run_stage1_stage2", "--",
        str(rust_in_dir), str(rust_out_dir)
    ]
    res = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    print("Rust output:\n", res.stdout)
    if res.returncode != 0:
        print("Rust stderr:\n", res.stderr)
        raise RuntimeError("Rust binary execution failed")

    # Load Rust logits
    raw_bytes = (rust_out_dir / "rust_logits_bthwc.f32").read_bytes()
    rust_logits = np.frombuffer(raw_bytes, dtype=np.float32).reshape(1, H_pad, W_pad, 4)
    alpha_rust = np.clip(rust_logits[0, :, :, 0], 0.0, 1.0)
    fg_rust = rust_logits[0, :, :, 1:4]

    # Unpad both to native dimensions
    alpha_torch_unpad = unpad(alpha_torch, pad, (h_orig, w_orig))
    alpha_rust_unpad = unpad(alpha_rust, pad, (h_orig, w_orig))
    fg_torch_unpad = unpad(fg_torch, pad, (h_orig, w_orig))
    fg_rust_unpad = unpad(fg_rust, pad, (h_orig, w_orig))

    # Diff metrics
    diff_alpha = np.abs(alpha_torch_unpad - alpha_rust_unpad)
    mean_err_a = diff_alpha.mean()
    max_err_a = diff_alpha.max()
    print(f"End-to-End Alpha Parity: mean={mean_err_a:.3e}, max={max_err_a:.3e}")

    diff_fg = np.abs(fg_torch_unpad - fg_rust_unpad)
    mean_err_fg = diff_fg.mean()
    max_err_fg = diff_fg.max()
    print(f"End-to-End FG Parity: mean={mean_err_fg:.3e}, max={max_err_fg:.3e}")

    # Despill & Checkerboard composite
    checker = make_checker(h_orig, w_orig).transpose(1, 2, 0)
    a_t = alpha_torch_unpad[:, :, None]
    a_r = alpha_rust_unpad[:, :, None]

    # Simple green despill for visualization
    def simple_despill(fg):
        # clamp green to max(red, blue)
        res = fg.copy()
        rb_max = np.maximum(res[:, :, 0], res[:, :, 2])
        res[:, :, 1] = np.minimum(res[:, :, 1], rb_max * 1.05)
        return np.clip(res, 0.0, 1.0)

    fg_t_despill = simple_despill(fg_torch_unpad)
    fg_r_despill = simple_despill(fg_rust_unpad)

    comp_torch = a_t * fg_t_despill + (1.0 - a_t) * checker
    comp_rust = a_r * fg_r_despill + (1.0 - a_r) * checker

    # Generate 8-panel comparison grid (2 rows x 4 cols)
    # [0, 0] Input Photo         [0, 1] Torch Alpha         [0, 2] Rust Alpha         [0, 3] |Alpha Diff| x 100
    # [1, 0] Checker Reference   [1, 1] Torch Composite     [1, 2] Rust Composite     [1, 3] |Comp Diff| x 100
    cell_w, cell_h = w_orig, h_orig
    margin_top = 40
    grid_w = cell_w * 4
    grid_h = (cell_h + margin_top) * 2

    canvas = Image.new("RGB", (grid_w, grid_h), color=(28, 28, 30))
    draw = ImageDraw.Draw(canvas)

    def to_pil(arr, is_gray=False):
        u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if is_gray:
            return Image.fromarray(u8, mode="L").convert("RGB")
        return Image.fromarray(u8, mode="RGB")

    def diff_heatmap(diff, scale=100.0):
        # Hot colormap emulation: black -> red -> yellow -> white
        val = np.clip(diff * scale, 0.0, 1.0)
        h, w = val.shape[:2]
        if val.ndim == 3:
            val = val.mean(axis=2)
        r = np.clip(val * 2.0, 0.0, 1.0)
        g = np.clip(val * 2.0 - 0.7, 0.0, 1.0)
        b = np.clip(val * 3.0 - 2.0, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=2)
        return to_pil(rgb)

    panels = [
        # Row 1
        ("1. Input Green Screen Photo", to_pil(rgb_orig), 0, 0),
        ("2. PyTorch Alpha Matte", to_pil(alpha_torch_unpad, is_gray=True), 1, 0),
        ("3. Rust (CubeCL CUDA) Alpha Matte", to_pil(alpha_rust_unpad, is_gray=True), 2, 0),
        (f"4. |Alpha Diff| x 100 (max={max_err_a:.2e})", diff_heatmap(diff_alpha, 100.0), 3, 0),
        # Row 2
        ("5. Checkerboard Background", to_pil(checker), 0, 1),
        ("6. PyTorch Despilled Composite", to_pil(comp_torch), 1, 1),
        ("7. Rust (CubeCL CUDA) Composite", to_pil(comp_rust), 2, 1),
        ("8. |Comp Diff| x 100 (FP32 Identity)", diff_heatmap(np.abs(comp_torch - comp_rust), 100.0), 3, 1),
    ]

    for title, img, col, row in panels:
        x = col * cell_w
        y = row * (cell_h + margin_top)
        draw.text((x + 10, y + 10), title, fill=(240, 240, 240))
        canvas.paste(img, (x, y + margin_top))

    # Save individual results in data/
    to_pil(alpha_torch_unpad, is_gray=True).save(root / "data" / "torch_alpha.png")
    to_pil(alpha_rust_unpad, is_gray=True).save(root / "data" / "rust_alpha.png")
    to_pil(comp_torch).save(root / "data" / "torch_composite.png")
    to_pil(comp_rust).save(root / "data" / "rust_composite.png")

    out_grid_path = root / "data" / "greenscreen_comparison_grid.png"
    canvas.save(out_grid_path, quality=95)
    print(f"\nSuccessfully generated comparison grid: {out_grid_path}")


if __name__ == "__main__":
    main()
