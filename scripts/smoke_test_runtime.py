#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package_root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--load_model", action="store_true")
    ap.add_argument("--gpu", choices=("auto", "nvidia", "amd"), default="auto")
    args = ap.parse_args()

    root = Path(args.package_root).resolve()
    sys.path.insert(0, str(root))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from corridorkey.device import describe_accelerator, resolve_accelerator
    from corridorkey.runtime import RuntimePaths, load_model, load_runtime_checkpoint

    paths = RuntimePaths.from_package(root)
    paths.validate()
    ckpt = load_runtime_checkpoint(paths.weights)
    assert ckpt["format"] == "corridorkey_v2_runtime_checkpoint"
    assert ckpt["metadata"]["pre_recurrence"] is True
    state = ckpt["state_dict"]
    bad = [k for k in state if "recurrent" in k.lower() or "state_attn" in k.lower()]
    if bad:
        raise RuntimeError(f"unexpected recurrence tensors: {bad[:8]}")
    forbidden = {"optimizer", "scheduler", "ema", "wandb_run_id"}
    present = forbidden.intersection(ckpt)
    if present:
        raise RuntimeError(f"runtime checkpoint contains training-only keys: {present}")
    print(f"checkpoint ok: {paths.weights} ({paths.weights.stat().st_size / 1e9:.2f} GB)")
    print(f"state tensors: {len(state)}")
    if args.load_model:
        accelerator = resolve_accelerator(args.gpu)
        print(describe_accelerator(accelerator))
        model = load_model(paths, device=accelerator.device)
        print(f"model ok: d16={model.stage1.d_16}, "
              f"cradio={model.stage1.cradio_dim}, "
              f"moge={model.stage1.moge_dim}, "
              f"rvm={model.stage1.rvm_dim}")


if __name__ == "__main__":
    main()
