#!/usr/bin/env python3
"""Convert the frozen foundation checkpoints the Rust runtime reads into safetensors.

Only the parts CorridorKey runs are kept:

- MoGe-2 ViT-B (`vfms/moge/model.pt`, a pickled `{model_config, model}`): the DINOv2 encoder and its
  two output projections (`encoder.*`, 88 M parameters). MoGe's neck and heads are never called
  (`FrozenFoundations.forward` uses `moge.encoder` only).
- RVM-small (`vfms/rvm/*.npz`, a JAX tree): the encoder and the gated recurrent core (33 M of the
  checkpoint's 67 M parameters; the MAE decoder is dropped), mapped to PyTorch names and layouts by
  `corridorkey.rvm.load_rvm_npz`, the mapping the Python runtime itself uses.

Tensors stay float32 (the runtime rounds to its compute type when loading); the source config is
kept in the safetensors metadata.

usage: convert_foundations.py [--root <package root>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_moge(root: Path, out_dir: Path) -> Path:
    src = root / "vfms" / "moge" / "model.pt"
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]["encoder"]
    tensors = {
        k[len("encoder."):]: v.float().contiguous()
        for k, v in ckpt["model"].items()
        if k.startswith("encoder.") and not k.endswith("mask_token")
    }
    out = out_dir / "moge2_vitb_encoder.safetensors"
    save_file(tensors, out, metadata={
        "format": "corridorkey_foundation",
        "model": "moge2_vitb_encoder",
        "config": json.dumps(cfg),
        "source": str(src.relative_to(root)),
    })
    params = sum(t.numel() for t in tensors.values())
    print(f"{out}: {len(tensors)} tensors, {params / 1e6:.1f} M parameters, config {cfg}")
    return out


def convert_rvm(root: Path, out_dir: Path) -> Path:
    import sys
    sys.path.insert(0, str(root))
    from corridorkey.rvm import RVM_CFG, RVMVideoSiamMAE, load_rvm_npz

    src = next((root / "vfms" / "rvm").glob("*.npz"))
    model = RVMVideoSiamMAE(variant="small")
    load_rvm_npz(model, str(src), strict_load=True, verbose=False)
    tensors = {k: v.float().contiguous() for k, v in model.state_dict().items()}
    out = out_dir / "rvm_small.safetensors"
    save_file(tensors, out, metadata={
        "format": "corridorkey_foundation",
        "model": "rvm_small",
        "config": json.dumps(RVM_CFG["small"]),
        "source": str(src.relative_to(root)),
    })
    params = sum(t.numel() for t in tensors.values())
    print(f"{out}: {len(tensors)} tensors, {params / 1e6:.1f} M parameters, config {RVM_CFG['small']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    out_dir = args.root / "weights" / "foundations"
    out_dir.mkdir(parents=True, exist_ok=True)
    convert_moge(args.root, out_dir)
    convert_rvm(args.root, out_dir)


if __name__ == "__main__":
    main()
