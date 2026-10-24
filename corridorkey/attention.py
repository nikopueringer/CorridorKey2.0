"""Memory-bounded scaled dot-product attention for GPUs without fused kernels.

On NVIDIA (CUDA builds of torch), and on AMD GPUs with flash or
memory-efficient SDPA kernels, this is a direct call to
F.scaled_dot_product_attention. Some AMD GPUs (e.g. RDNA1, gfx1010) only have
the math backend, which materializes the full (L_q x L_k) score matrix for
every batch entry and head at once; at video resolutions that is tens of GB.
There the call is split along the batch and query axes. Each softmax row
depends on a single query, so the result is mathematically equivalent to one
full call (up to floating-point reduction order).
"""
from __future__ import annotations

import os
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Score-matrix elements per math-backend call. The math path keeps fp32
# scores, an fp32 softmax and a mask/cast temporary, so peak transient memory
# is roughly 10 bytes per element (~640 MB at the default 2**26).
_MAX_SCORE_ELEMENTS = int(os.environ.get("CORRIDORKEY_SDPA_MAX_SCORE_ELEMENTS", 2**26))

_fused_cache: dict[tuple[torch.device, torch.dtype], bool] = {}


@torch.compiler.disable
def _probe_fused_sdpa(device: torch.device, dtype: torch.dtype) -> Optional[bool]:
    try:
        q = torch.zeros(1, 1, 64, 64, device=device, dtype=dtype)
        # Unavailable backends each emit a UserWarning before failing.
        with warnings.catch_warnings(), \
                torch.autocast(device.type, enabled=False), \
                sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                             SDPBackend.EFFICIENT_ATTENTION]):
            warnings.simplefilter("ignore", UserWarning)
            F.scaled_dot_product_attention(q, q, q)
        return True
    except torch.OutOfMemoryError:
        return None  # inconclusive
    except RuntimeError:
        return False


def has_fused_sdpa(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether SDPA has a flash or memory-efficient kernel for this device/dtype.

    Probed once with a small unmasked call; the result is cached per device/dtype.
    """
    key = (device, dtype)
    if key not in _fused_cache:
        found = _probe_fused_sdpa(device, dtype)
        if found is None:
            return False  # bound this call, but probe again next time
        _fused_cache[key] = found
    return _fused_cache[key]


def _slice_dim0(t: Optional[torch.Tensor], q: torch.Tensor,
                start: int, end: int) -> Optional[torch.Tensor]:
    # SDPA broadcasts from the right: only a same-rank tensor shares q's dim 0.
    if t is None or t.dim() != q.dim() or t.shape[0] == 1:
        return t
    return t[start:end]


def _bounded_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], kwargs: dict) -> torch.Tensor:
    others = [k, v] + ([attn_mask] if attn_mask is not None else [])
    if any(t.dim() > q.dim() for t in others):
        # Broadcasting would add leading dims to the output; not split here.
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, **kwargs)
    # The probabilities are broadcast against v too before the final matmul.
    lead = torch.broadcast_shapes(
        q.shape[:-2], k.shape[:-2], v.shape[:-2],
        *([attn_mask.shape[:-2]] if attn_mask is not None and attn_mask.dim() > 2 else []))
    rows = q.shape[-2]
    per_row = lead.numel() * k.shape[-2]
    total = per_row * rows
    if total <= _MAX_SCORE_ELEMENTS:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, **kwargs)
    n = lead[0] if len(lead) else 1
    if q.dim() >= 3 and n > 1:
        # Prefer whole batch entries (e.g. attention windows); a single
        # oversized entry falls through to query splitting on recursion.
        step = max(1, n * _MAX_SCORE_ELEMENTS // total)
        return torch.cat([
            _bounded_sdpa(_slice_dim0(q, q, s, s + step),
                          _slice_dim0(k, q, s, s + step),
                          _slice_dim0(v, q, s, s + step),
                          _slice_dim0(attn_mask, q, s, s + step), kwargs)
            for s in range(0, n, step)], dim=0)
    # One query row (across all remaining batch and head dims) is the smallest
    # unit; it only exceeds the budget for heads * L_k above 2**26.
    step = max(1, _MAX_SCORE_ELEMENTS // per_row)
    mask_rows = (attn_mask is not None and attn_mask.dim() >= 2
                 and attn_mask.shape[-2] == rows)
    return torch.cat([
        F.scaled_dot_product_attention(
            q[..., s:s + step, :], k, v,
            attn_mask=attn_mask[..., s:s + step, :] if mask_rows else attn_mask,
            **kwargs)
        for s in range(0, rows, step)], dim=-2)


def _effective_dtype(q: torch.Tensor) -> torch.dtype:
    # Under autocast SDPA runs in the autocast dtype, not q's.
    if q.is_floating_point() and torch.is_autocast_enabled(q.device.type):
        return torch.get_autocast_dtype(q.device.type)
    return q.dtype


def _needs_bounding(q: torch.Tensor, dropout_p: float, is_causal: bool,
                    kwargs: dict) -> bool:
    # Only ROCm builds can lack fused kernels; CUDA builds always pass through.
    return not (is_causal or dropout_p or kwargs.get("enable_gqa")
                or not q.is_cuda or torch.version.hip is None
                or has_fused_sdpa(q.device, _effective_dtype(q)))


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None,
         dropout_p: float = 0.0, is_causal: bool = False, **kwargs) -> torch.Tensor:
    """Drop-in for F.scaled_dot_product_attention with bounded math-path memory."""
    if _needs_bounding(q, dropout_p, is_causal, kwargs):
        return _bounded_sdpa(q, k, v, attn_mask, kwargs)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, **kwargs)
