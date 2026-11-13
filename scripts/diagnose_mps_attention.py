#!/usr/bin/env python3
"""Collect MPS SDPA layout/mask diagnostics against CPU references.

Inputs are generated deterministically on CPU and copied to MPS. Every case
records MPS SDPA versus both CPU SDPA and an explicit fp32 softmax reference.
This is diagnostic-only: backend errors and numerical differences are recorded
without stopping later cases, and the script exits successfully after a run.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


def make_mask(kind: str, encoding: str, layout: str, *, batch: int,
              heads: int, tokens: int, device: torch.device) -> torch.Tensor | None:
    if kind == "none":
        return None
    if kind == "key_padding":
        base = torch.ones(batch, 1, 1, tokens, dtype=torch.bool, device=device)
        base[:, :, :, -max(1, tokens // 8):] = False
    else:
        base = torch.ones(1, 1, tokens, tokens, dtype=torch.bool, device=device)
        rows = torch.arange(tokens, device=device).view(1, 1, tokens, 1)
        cols = torch.arange(tokens, device=device).view(1, 1, 1, tokens)
        base = cols <= (rows + max(1, tokens // 3)) % tokens
        # Ensure at least one legal key for every query.
        base[..., 0] = True

    if layout == "dense_per_head":
        base = base.expand(batch, heads, tokens, tokens).contiguous()
    if encoding == "additive":
        return torch.zeros_like(base, dtype=torch.float32).masked_fill_(~base, float("-inf"))
    return base


def explicit_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: torch.Tensor | None) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask.float()
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    finite = bool(torch.isfinite(actual).all().item())
    if not finite:
        return {"finite": False, "max_abs": None, "allclose_1e-5_1e-4": False}
    delta = (actual.float() - expected.float()).abs()
    return {
        "finite": True,
        "max_abs": float(delta.max().item()),
        "allclose_1e-5_1e-4": bool(torch.allclose(actual.float(), expected.float(), atol=1e-5, rtol=1e-4)),
    }


def tensor_layout(tensor: torch.Tensor) -> dict:
    return {
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "is_contiguous": tensor.is_contiguous(),
    }


def run_case(*, packed_cpu: torch.Tensor, packed_mps: torch.Tensor,
             v_layout: str, mask_kind: str, mask_encoding: str,
             mask_layout: str) -> dict:
    batch, _, heads, tokens, dim = packed_cpu.shape
    # Q/K are dense in the production path after RMSNorm + RoPE. Keep only V
    # layout as the independent variable in this diagnostic.
    q = packed_cpu[:, 0].clone(memory_format=torch.contiguous_format)
    k = packed_cpu[:, 1].clone(memory_format=torch.contiguous_format)
    raw_v = packed_cpu[:, 2]
    q_mps = packed_mps[:, 0].clone(memory_format=torch.contiguous_format)
    k_mps = packed_mps[:, 1].clone(memory_format=torch.contiguous_format)
    raw_v_mps = packed_mps[:, 2]
    if v_layout == "packed_view":
        v, v_mps = raw_v, raw_v_mps
    elif v_layout == "contiguous":
        v, v_mps = raw_v.contiguous(), raw_v_mps.contiguous()
    else:
        v = raw_v.clone(memory_format=torch.contiguous_format)
        v_mps = raw_v_mps.clone(memory_format=torch.contiguous_format)

    mask = make_mask(mask_kind, mask_encoding, mask_layout, batch=batch,
                     heads=heads, tokens=tokens, device=torch.device("cpu"))
    cpu_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    cpu_explicit = explicit_fp32(q, k, v, mask)

    result = {
        "shape": {"batch": batch, "heads": heads, "tokens": tokens, "head_dim": dim},
        "v_layout": v_layout,
        "q_cpu": tensor_layout(q),
        "k_cpu": tensor_layout(k),
        "v_cpu": tensor_layout(v),
        "mask": {"kind": mask_kind, "encoding": mask_encoding, "layout": mask_layout,
                 "shape": None if mask is None else list(mask.shape),
                 "layout_details": None if mask is None else tensor_layout(mask)},
        "cpu_sdpa_vs_explicit_fp32": compare(cpu_sdpa, cpu_explicit),
    }

    device = torch.device("mps")
    mask_mps = None
    try:
        mask_mps = None if mask is None else mask.to(device)
        result["q_mps"] = tensor_layout(q_mps)
        result["k_mps"] = tensor_layout(k_mps)
        result["v_mps"] = tensor_layout(v_mps)
        result["mask"]["layout_details_mps"] = None if mask_mps is None else tensor_layout(mask_mps)
    except Exception as exc:
        result["mps_input_setup_error"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        mps_explicit = explicit_fp32(q_mps, k_mps, v_mps, mask_mps).cpu()
        result["mps_explicit_vs_cpu_explicit_fp32"] = compare(mps_explicit, cpu_explicit)
    except Exception as exc:
        result["mps_explicit_error"] = f"{type(exc).__name__}: {exc}"

    try:
        actual = F.scaled_dot_product_attention(q_mps, k_mps, v_mps, attn_mask=mask_mps).cpu()
        result["mps_sdpa_vs_cpu_sdpa"] = compare(actual, cpu_sdpa)
        result["mps_sdpa_vs_explicit_fp32"] = compare(actual, cpu_explicit)
    except Exception as exc:
        result["mps_sdpa_error"] = f"{type(exc).__name__}: {exc}"
    return result


def collect() -> dict:
    report = {
        "metadata": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "macos_version": platform.mac_ver()[0] or None,
            "torch": torch.__version__,
            "mps_built": bool(getattr(torch.backends.mps, "is_built", lambda: False)()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cpu_threads": torch.get_num_threads(),
        },
        "results": [],
    }
    if not report["metadata"]["mps_available"]:
        report["status"] = "mps_unavailable"
        return report

    cases = []
    for batch, heads, tokens, dim in ((1, 4, 16, 8), (5, 4, 16, 8), (5, 4, 256, 64)):
        generator = torch.Generator(device="cpu").manual_seed(1729 + tokens)
        # Keep identical Q/K/V for every mask and layout variant. V is the
        # third view in a packed [B,3,H,N,D] fixture; selecting it after the
        # device copy preserves its nonzero offset and 3x outer stride on MPS.
        packed_cpu = torch.randn(batch, 3, heads, tokens, dim, generator=generator)
        packed_cpu[:, :2].mul_(0.25)
        packed_mps = packed_cpu.to("mps")
        for v_layout in ("packed_view", "contiguous", "zero_offset_clone"):
            for mask_kind in ("none", "key_padding", "query_specific"):
                if mask_kind == "none":
                    cases.append((packed_cpu, packed_mps, v_layout, "none", "none", "broadcast"))
                    continue
                for encoding in ("bool", "additive"):
                    for layout in ("broadcast", "dense_per_head"):
                        cases.append((packed_cpu, packed_mps, v_layout, mask_kind, encoding, layout))

    for packed_cpu, packed_mps, v_layout, mask_kind, encoding, layout in cases:
        report["results"].append(run_case(
            packed_cpu=packed_cpu,
            packed_mps=packed_mps,
            v_layout=v_layout,
            mask_kind=mask_kind,
            mask_encoding=encoding,
            mask_layout=layout,
        ))
    report["status"] = "complete"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    args = parser.parse_args()
    report = collect()
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
