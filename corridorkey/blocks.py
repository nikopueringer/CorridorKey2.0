"""Building blocks for CorridorKey v2.

Transformer-native matting stack. Spatial pyramid is /16 ↔ /64 only,
stage 2 runs pure /16 Swin-2D → depatchify.

Publicly exported:
  • LayerNorm2d — channel-LN on (B, C, H, W) tensors.
  • SwiGLU — gated MLP, 2× expansion default (3 linears of d·2d each).
  • SwinBlock3D — windowed 3D (T, H, W) attention + SwiGLU.
  • FullSpatialBlock3D — per-frame full-spatial attention + SwiGLU.
  • SwinBlock2D — windowed 2D attention + SwiGLU.
  • FullSpatioTemporalBlock — unwindowed 3D attention (for /64 body).
  • PatchifyConv — Conv(kernel=r, stride=r), optional masked-channel
    zero-init (for stage-2 patchify_1 baton-overlap and patchify_2 skip).
  • Downsample4 — kernel-8 stride-4 conv (for /16 → /64).
  • Upsample4ICNR — Linear + PixelShuffle(4), ICNR-inited, optional
    post-smoothing Conv(kernel=3).
  • Depatchify — Linear(d → C·r²) + PixelShuffle(r) for /16 → /1 and /16 → /2.
  • icnr_init_ — Aitken 2017 ICNR init for conv feeding PixelShuffle.

Residual blocks use pre-norm plus zero-initialized output projections/gates so
every block is near-identity at t=0 and grows responsibility with training.
"""

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import sdpa
from .rope import RoPE2D, RoPE3D, apply_rotary


# ---------------------------------------------------------------------------
# Norms and utilities
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """RMSNorm over channel dim for (B, C, H, W) tensors.

    Kept the historical class name for backward compat of state-dict keys,
    but the implementation is now RMSNorm: mean subtraction dropped, no
    learnable bias, variance-only normalization. ~10-15% faster than
    LayerNorm, and matches the EUPE / DINOv3 convention used by our
    frozen foundations.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6,
                 affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
        else:
            self.register_parameter('weight', None)
        # RMSNorm has no bias; keep the attribute for any code that touches
        # it defensively, but it is always None.
        self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, H, W, C) → RMSNorm → back
        x = x.permute(0, 2, 3, 1)
        # Cast weight to input dtype so F.rms_norm's fused kernel dispatches
        # (autocast keeps params fp32; mismatch blocks the fast path).
        w = self.weight.to(x.dtype) if self.weight is not None else None
        x = F.rms_norm(x, (self.num_channels,), w, self.eps).to(x.dtype)
        return x.permute(0, 3, 1, 2).contiguous()


class RMSNorm(nn.Module):
    """Drop-in replacement for nn.RMSNorm that casts `weight` to input dtype
    inline so F.rms_norm's fused kernel dispatches under autocast.

    The stock nn.RMSNorm keeps `weight` in fp32; under autocast(bf16) the
    activation is bf16, and F.rms_norm refuses the fused path on dtype
    mismatch — this was the source of a perf-warning in training.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Newer torch autocast returns fp32 from rms_norm; keep the input dtype.
        return F.rms_norm(x, self.normalized_shape,
                           self.weight.to(x.dtype), self.eps).to(x.dtype)


def icnr_init_(conv: nn.Conv2d, upscale_factor: int):
    """ICNR init (Aitken 2017) for a Conv2d that feeds a PixelShuffle.

    Writes a "tile" where each r² group of output channels shares the
    same sub-kernel — the initial PixelShuffle becomes a nearest-neighbor
    upsample, so zero high-freq at t=0.
    """
    r2 = upscale_factor * upscale_factor
    W = conv.weight.data
    out_ch, in_ch, kh, kw = W.shape
    assert out_ch % r2 == 0, (
        f"icnr_init: out_channels ({out_ch}) must be divisible by r² ({r2}).")
    c_out = out_ch // r2
    sub = torch.empty(c_out, in_ch, kh, kw, dtype=W.dtype, device=W.device)
    nn.init.kaiming_uniform_(sub, a=5 ** 0.5)
    W.copy_(sub.repeat_interleave(r2, dim=0))
    if conv.bias is not None:
        conv.bias.data.zero_()


def icnr_init_linear_(linear: nn.Linear, upscale_factor: int,
                       out_channels_per_slot: int):
    """ICNR init for a Linear feeding PixelShuffle.

    `linear.weight` has shape `(out_dim, in_dim)` with
    `out_dim = out_channels_per_slot * upscale_factor²`. Groups of r² along
    out_dim share the same slot-weight so PixelShuffle at init is nearest-
    neighbor.
    """
    r2 = upscale_factor * upscale_factor
    out_dim, in_dim = linear.weight.data.shape
    assert out_dim == out_channels_per_slot * r2, (
        f"icnr_init_linear: out_dim={out_dim} != c·r² = "
        f"{out_channels_per_slot}·{r2}")
    sub = torch.empty(out_channels_per_slot, in_dim,
                      dtype=linear.weight.dtype, device=linear.weight.device)
    nn.init.kaiming_uniform_(sub, a=5 ** 0.5)
    linear.weight.data.copy_(sub.repeat_interleave(r2, dim=0))
    if linear.bias is not None:
        linear.bias.data.zero_()


def _random_orthogonal_matrix(dim: int, device: torch.device) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, device=device))
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


@torch.no_grad()
def init_orthogonal_rgb_passthrough_(
    patchify: 'PatchifyConv',
    depatchify: 'Depatchify',
    *,
    token_offset: int = 0,
    output_rgb_offset: int = 1,
    downsample: int = 1,
) -> bool:
    """Initialize patchify/depatchify as an exact RGB passthrough.

    A full random orthogonal basis maps each native RGB patch to tokens when
    the token width is 768. For lower-width stage-1 aux heads, the encoded
    subspace is built to contain the exact area-downsampled RGB rows, plus a
    random orthogonal complement. Returns False when the requested passthrough
    cannot be represented exactly.
    """
    if patchify.overlap_mult != 1 or depatchify.overlap_mult != 1:
        return False
    if depatchify.linear is None:
        return False

    patch = patchify.patch
    r = depatchify.r
    if patch != r * downsample:
        return False

    rgb_dim = 3 * patch * patch
    target_dim = 3 * r * r
    out_dim = patchify.conv.out_channels
    if out_dim > rgb_dim:
        return False
    if out_dim < target_dim:
        return False
    if patchify.conv.in_channels < 3:
        return False
    if token_offset < 0 or token_offset + out_dim > depatchify.in_dim:
        return False
    if output_rgb_offset < 0 or output_rgb_offset + 3 > depatchify.out_channels:
        return False

    basis_dtype = patchify.conv.weight.dtype
    device = patchify.conv.weight.device
    if out_dim == rgb_dim:
        basis = _random_orthogonal_matrix(rgb_dim, device)
    else:
        target_rows = torch.zeros(target_dim, rgb_dim, device=device)
        for c in range(3):
            for oy in range(r):
                for ox in range(r):
                    row = c * r * r + oy * r + ox
                    y0 = oy * downsample
                    x0 = ox * downsample
                    for dy in range(downsample):
                        for dx in range(downsample):
                            p_idx = c * patch * patch + (y0 + dy) * patch + (x0 + dx)
                            target_rows[row, p_idx] = 1.0 / downsample
        rot = _random_orthogonal_matrix(target_dim, device)
        fixed_rows = rot @ target_rows
        rand = torch.randn(rgb_dim, out_dim - target_dim, device=device)
        rand = rand - fixed_rows.T @ (fixed_rows @ rand)
        q, _ = torch.linalg.qr(rand, mode='reduced')
        basis = torch.cat([fixed_rows, q.T], dim=0)
    basis = basis.to(dtype=basis_dtype)

    patchify.conv.weight.zero_()
    patchify.conv.weight[:, :3, :patch, :patch].copy_(
        basis.view(out_dim, 3, patch, patch)
    )
    if patchify.conv.bias is not None:
        patchify.conv.bias.zero_()

    depatchify.linear.weight.zero_()
    if depatchify.linear.bias is not None:
        depatchify.linear.bias.zero_()
    if depatchify.fourier is not None:
        depatchify.fourier.reset_to_zero()
    w = depatchify.linear.weight
    for c in range(3):
        out_c = output_rgb_offset + c
        for oy in range(r):
            for ox in range(r):
                row = out_c * r * r + oy * r + ox
                cols = []
                y0 = oy * downsample
                x0 = ox * downsample
                for dy in range(downsample):
                    for dx in range(downsample):
                        p_idx = c * patch * patch + (y0 + dy) * patch + (x0 + dx)
                        cols.append(p_idx)
                target = torch.zeros(rgb_dim, device=device, dtype=basis.dtype)
                for p_idx in cols:
                    target[p_idx] = 1.0 / len(cols)
                vec = basis @ target
                w[row, token_offset:token_offset + out_dim].copy_(vec)
    return True


# ---------------------------------------------------------------------------
# MLP: SwiGLU 2× expansion
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """Gated MLP (SwiGLU) — equal-dim variant (in == out == dim).

    `h = silu(W_gate @ x) * (W_up @ x); out = W_down @ h`.
    Optional `dw_kernel` inserts a depthwise spatial conv on `h` before
    `W_down`, i.e. post-gating and pre-down-projection.
    At `mult=2.5` this is 7.5·dim² params (vs GELU-MLP 4× = 8·dim²) — smaller
    than the param-matched GELU-4× SwiGLU (mult ≈ 2.67), intentional
    VRAM/compute tradeoff.

    `zero_init_down=True` (default) zeros `w_down.weight` and `w_down.bias`
    so the residual branch contributes exactly 0 at init, letting us drop
    LayerScale gating. Set `zero_init_down=False` to keep Kaiming init
    (legacy behavior that relied on LayerScale for near-identity).
    """

    def __init__(
        self,
        dim: int,
        mult: float = 2.5,
        bias: bool = True,
        zero_init_down: bool = True,
        dw_kernel: int = 0,
        hidden: int | None = None,
    ):
        super().__init__()
        hidden = int(hidden) if hidden is not None else int(round(dim * mult))
        self.hidden = hidden
        self.dw_pad = int(dw_kernel) // 2 if dw_kernel and dw_kernel > 0 else 0
        self.w_gate = nn.Linear(dim, hidden, bias=bias)
        self.w_up   = nn.Linear(dim, hidden, bias=bias)
        self.dw = (
            nn.Conv2d(hidden, hidden, kernel_size=int(dw_kernel),
                      padding=0, groups=hidden, bias=True)
            if dw_kernel and dw_kernel > 0 else None
        )
        self.w_down = nn.Linear(hidden, dim, bias=bias)
        if zero_init_down:
            nn.init.zeros_(self.w_down.weight)
            if self.w_down.bias is not None:
                nn.init.zeros_(self.w_down.bias)

    def _dw_spatial(self, h: torch.Tensor) -> torch.Tensor:
        if self.dw is None:
            return h
        if h.dim() == 5:
            B, T, H, W, C = h.shape
            y = h.reshape(B * T, H, W, C)
            shape_out = (B, T, H, W, C)
        elif h.dim() == 4:
            B, H, W, C = h.shape
            y = h
            shape_out = (B, H, W, C)
        else:
            return h
        y = y.permute(0, 3, 1, 2)
        if self.dw_pad:
            mode = 'reflect' if H > self.dw_pad and W > self.dw_pad else 'replicate'
            y = F.pad(
                y,
                (self.dw_pad, self.dw_pad, self.dw_pad, self.dw_pad),
                mode=mode,
            )
        y = self.dw(y).permute(0, 2, 3, 1)
        return y.reshape(shape_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunk_tokens = int(getattr(self, 'inference_chunk_tokens', 0) or 0)
        if (chunk_tokens > 0 and not self.training and self.dw is None
                and x.numel() // x.shape[-1] > chunk_tokens):
            shape = x.shape
            c = shape[-1]
            y = x.reshape(-1, c)
            out = y.new_empty(y.shape)
            for start in range(0, y.shape[0], chunk_tokens):
                yi = y[start:start + chunk_tokens]
                h = F.silu(self.w_gate(yi)) * self.w_up(yi)
                out[start:start + yi.shape[0]] = self.w_down(h)
            return out.reshape(shape)
        h = F.silu(self.w_gate(x)) * self.w_up(x)
        h = self._dw_spatial(h)
        return self.w_down(h)


class DepthwiseSwiGLU2D(nn.Module):
    """Depthwise spatial SwiGLU branch for NHWC feature maps.

    The branch is gated by a per-channel gamma initialized to exact zero, so it
    is an exact residual identity at init.
    The value and gate kernels are both depthwise 7x7 filters. After gamma
    learns away from zero, they provide local translation-equivariant mixing
    without adding pointwise channel mixing that the MLP already supplies.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        init_std: float | None = None,
    ):
        super().__init__()
        assert kernel_size > 0 and kernel_size % 2 == 1, (
            f"kernel_size must be a positive odd integer; got {kernel_size}")
        self.dim = dim
        self.kernel_size = kernel_size
        self.norm = RMSNorm(dim)
        self.pad = kernel_size // 2
        self.value = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                               padding=0, groups=dim, bias=True)
        self.gate = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                              padding=0, groups=dim, bias=True)
        if init_std is None:
            nn.init.kaiming_uniform_(self.value.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        else:
            nn.init.normal_(self.value.weight, mean=0.0, std=init_std)
            nn.init.normal_(self.gate.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.gate.bias)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C). Returns a residual branch in the same layout."""
        u = self.norm(x)
        # NHWC-contiguous -> NCHW shape with channels-last strides. Avoid an
        # explicit contiguous copy; cuDNN handles this layout efficiently.
        u = u.permute(0, 3, 1, 2)
        if self.pad:
            mode = 'reflect' if x.shape[1] > self.pad and x.shape[2] > self.pad else 'replicate'
            u = F.pad(u, (self.pad, self.pad, self.pad, self.pad),
                      mode=mode)
        # One grouped conv with two filters per input channel. Keep value/gate
        # as separate parameters for optimizer state and checkpoint clarity,
        # but avoid launching two depthwise convs over the same activation.
        weight = torch.stack((self.value.weight, self.gate.weight), dim=1)
        weight = weight.reshape(2 * self.dim, 1, self.kernel_size, self.kernel_size)
        if self.value.bias is not None and self.gate.bias is not None:
            bias = torch.stack((self.value.bias, self.gate.bias), dim=1).reshape(2 * self.dim)
        else:
            bias = None
        out_elems_per_item = 2 * self.dim * x.shape[1] * x.shape[2]
        max_items = max(1, 1_000_000_000 // max(1, out_elems_per_item))
        if u.shape[0] > max_items:
            y = torch.cat([
                F.conv2d(u[i:i + max_items], weight, bias=bias, groups=self.dim)
                for i in range(0, u.shape[0], max_items)
            ], dim=0)
        else:
            y = F.conv2d(u, weight, bias=bias, groups=self.dim)
        y = y.view(y.shape[0], self.dim, 2, y.shape[2], y.shape[3])
        y = y[:, :, 0] * F.silu(y[:, :, 1])
        y = y * self.gamma.to(y.dtype).view(1, -1, 1, 1)
        return y.permute(0, 2, 3, 1)


class AdaLNZero(nn.Module):
    """Zero-init global conditioning for NHWC/NTHWC pre-norm activations."""

    def __init__(self, dim: int, cond_dim: int = 1, pairs: int = 2,
                 hidden_dim: int | None = None):
        super().__init__()
        self.dim = int(dim)
        self.pairs = int(pairs)
        hidden = int(hidden_dim or dim)
        self.in_proj = nn.Linear(int(cond_dim), hidden, bias=True)
        self.act = nn.SiLU()
        self.proj = nn.Linear(hidden, self.pairs * 3 * dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if cond.dim() == 2:
            cond = cond.unsqueeze(-1)
        if cond.dim() != 3:
            raise ValueError(
                f"AdaLNZero cond must be (B,T) or (B,T,C), got {tuple(cond.shape)}")
        cond = cond.to(device=x.device, dtype=self.in_proj.weight.dtype)
        params = self.proj(self.act(self.in_proj(cond)))
        params = params.to(dtype=x.dtype)
        if x.dim() == 5:
            view_shape = (x.shape[0], x.shape[1], 1, 1, self.dim)
        elif x.dim() == 4:
            view_shape = (x.shape[0], 1, 1, self.dim)
        else:
            raise ValueError(f"AdaLNZero x must be NHWC or NTHWC, got {tuple(x.shape)}")
        return tuple(p.reshape(view_shape) for p in params.chunk(self.pairs * 3, dim=-1))


class FusionLinSwiglu(nn.Module):
    """Linear projection with a SwiGLU wrapped AROUND it — for /16 ↔ VFM
    fusion at /64.

    Input: `(..., vfm_dim + aux_dim)` (the concat of VFM features and
    auxiliary downsampled stage-1 features).
    Output: `(..., out_dim)` (default: == vfm_dim).

    Two parallel paths, summed:

      • Straight linear shortcut:
          Linear(vfm_dim + aux_dim → out_dim), initialized as
          `[P | 0]`, where P is identity when dimensions match and otherwise
          a variance-preserving semi-orthogonal projection over every VFM
          channel. At t=0, the /64 stream is a stable VFM-derived latent and
          the aux half contributes zero.

      • Nonlinear SwiGLU refinement:
          LayerNorm → Linear(in → hidden) × 2 (gate + up) → silu(gate)·up
            → Linear(hidden → out_dim). `hidden = mult · in` (default 2×).
          The final Linear's weight/bias are zero-inited, so at t=0 this
          path contributes 0. Since hidden > in > out, the SwiGLU UP-
          projects the full input BEFORE any compression, giving
          genuine nonlinear mixing between the vfm and aux channels
          prior to collapse.

    Combined behavior: at t=0, `output == P(vfm)`. During training the
    shortcut learns source weighting and the SwiGLU learns nonlinear
    interactions between VFM and aux features.
    """

    def __init__(self, vfm_dim: int, aux_dim: int, out_dim: int = None,
                 mult: float = 2.5, bias: bool = True,
                 hidden: int = None):
        """If `hidden` is None, it defaults to `round((vfm_dim + aux_dim) · mult)`.
        Pass an explicit `hidden` to hold the SwiGLU hidden size constant
        when retrofitting new input channels (e.g. adding RVM features)
        without expanding the hidden layer — preserves the existing trained
        hidden units."""
        super().__init__()
        if out_dim is None:
            out_dim = vfm_dim
        in_dim = vfm_dim + aux_dim
        self.vfm_dim = vfm_dim
        self.aux_dim = aux_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        if hidden is None:
            hidden = int(round(in_dim * mult))
        self.hidden = hidden

        # Straight linear shortcut — semi-orthogonal VFM projection + zero aux.
        self.linear_shortcut = nn.Linear(in_dim, out_dim, bias=bias)
        with torch.no_grad():
            self.linear_shortcut.weight.zero_()
            if vfm_dim > 0:
                if out_dim == vfm_dim:
                    idx = torch.arange(out_dim)
                    self.linear_shortcut.weight[idx, idx] = 1.0
                elif out_dim < vfm_dim:
                    # Compress: rows are orthonormal, so whitened VFM features
                    # retain unit variance per output channel.
                    q, r = torch.linalg.qr(
                        torch.randn(vfm_dim, out_dim,
                                    device=self.linear_shortcut.weight.device,
                                    dtype=torch.float32),
                        mode='reduced',
                    )
                    signs = torch.sign(torch.diagonal(r))
                    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                    proj = (q * signs.view(1, -1)).T
                    self.linear_shortcut.weight[:, :vfm_dim].copy_(
                        proj.to(dtype=self.linear_shortcut.weight.dtype))
                else:
                    # Expand: columns are orthonormal, so the VFM vector norm is
                    # preserved while occupying the full output channel space.
                    q, r = torch.linalg.qr(
                        torch.randn(out_dim, vfm_dim,
                                    device=self.linear_shortcut.weight.device,
                                    dtype=torch.float32),
                        mode='reduced',
                    )
                    signs = torch.sign(torch.diagonal(r))
                    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                    proj = q * signs.view(1, -1)
                    self.linear_shortcut.weight[:, :vfm_dim].copy_(
                        proj.to(dtype=self.linear_shortcut.weight.dtype))
            if self.linear_shortcut.bias is not None:
                self.linear_shortcut.bias.zero_()

        # SwiGLU path — own pre-norm; zero-init on down so contributes 0 at t=0.
        self.norm = RMSNorm(in_dim)
        self.w_gate = nn.Linear(in_dim, hidden, bias=bias)
        self.w_up   = nn.Linear(in_dim, hidden, bias=bias)
        self.w_down = nn.Linear(hidden, out_dim, bias=bias)
        nn.init.zeros_(self.w_down.weight)
        if self.w_down.bias is not None:
            nn.init.zeros_(self.w_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_dim). Returns (..., out_dim). At t=0, output is a
        stable projection of the VFM portion of the input; aux/SwiGLU are zero."""
        chunk_tokens = int(getattr(self, 'inference_chunk_tokens', 0) or 0)
        if (chunk_tokens > 0 and not self.training
                and x.numel() // x.shape[-1] > chunk_tokens):
            shape = x.shape[:-1] + (self.out_dim,)
            y = x.reshape(-1, x.shape[-1])
            out = y.new_empty((y.shape[0], self.out_dim))
            for start in range(0, y.shape[0], chunk_tokens):
                yi = y[start:start + chunk_tokens]
                shortcut = self.linear_shortcut(yi)
                h = self.norm(yi)
                nonlinear = self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
                out[start:start + yi.shape[0]] = shortcut + nonlinear
            return out.reshape(shape)
        shortcut = self.linear_shortcut(x)
        h = self.norm(x)
        nonlinear = self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return shortcut + nonlinear


# ---------------------------------------------------------------------------
# Window partitioning (2D + 3D)
# ---------------------------------------------------------------------------


def window_partition_2d(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """(B, H, W, C) → (B·num_windows, window²·C flattened later) layout.

    Returns (windows, (H, W)) where windows is (B·nH·nW, window, window, C).
    Callers use `windows.flatten(1, 2)` to get (B·nH·nW, window², C).
    """
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, (
        f"window_partition_2d: H×W ({H}×{W}) must divide window={window_size}")
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C), (H, W)


def window_unpartition_2d(windows: torch.Tensor, window_size: int,
                          HW: Tuple[int, int], B: int) -> torch.Tensor:
    """Inverse of window_partition_2d. Returns (B, H, W, C)."""
    H, W = HW
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)


def window_partition_3d(x: torch.Tensor, T_w: int, H_w: int, W_w: int,
                        ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """(B, T, H, W, C) → (B·num_windows, T_w·H_w·W_w, C) later flattened.

    Returns (windows, (T, H, W)) where windows is
    (B·nT·nH·nW, T_w, H_w, W_w, C).
    """
    B, T, H, W, C = x.shape
    assert T % T_w == 0 and H % H_w == 0 and W % W_w == 0, (
        f"window_partition_3d: shape ({T},{H},{W}) vs window "
        f"({T_w},{H_w},{W_w}) — must divide evenly.")
    x = x.view(B, T // T_w, T_w, H // H_w, H_w, W // W_w, W_w, C)
    # Reorder to (B, n_t, n_h, n_w, T_w, H_w, W_w, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, T_w, H_w, W_w, C), (T, H, W)


def window_unpartition_3d(windows: torch.Tensor, T_w: int, H_w: int, W_w: int,
                           THW: Tuple[int, int, int], B: int) -> torch.Tensor:
    """Inverse of window_partition_3d. Returns (B, T, H, W, C)."""
    T, H, W = THW
    x = windows.view(B, T // T_w, H // H_w, W // W_w, T_w, H_w, W_w, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, T, H, W, -1)


def _shifted_region_ids_2d(
    H: int,
    W: int,
    Hp: int,
    Wp: int,
    window_size: int,
    shift_h: int,
    shift_w: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Per-window region ids for standard shifted-window masking.

    The ids are built on the padded pre-shift grid, then cyclically shifted
    exactly like the activations and partitioned into windows. Tokens with
    different non-negative ids must not attend to each other; id -1 denotes
    padding. This represents the standard Swin wraparound mask without
    materializing a full N×N mask, which is critical for full-temporal 3D
    windows.
    """

    sh, sw = int(shift_h), int(shift_w)
    if sh == 0 and sw == 0:
        return None
    w = int(window_size)
    region = torch.full((Hp, Wp), -1, dtype=torch.int16, device=device)
    real = torch.zeros((Hp, Wp), dtype=torch.bool, device=device)
    real[:H, :W] = True
    h_slices = (
        (slice(0, -w), slice(-w, -sh), slice(-sh, None))
        if sh > 0 else (slice(0, None),)
    )
    w_slices = (
        (slice(0, -w), slice(-w, -sw), slice(-sw, None))
        if sw > 0 else (slice(0, None),)
    )
    idx = 0
    for hs in h_slices:
        for ws in w_slices:
            sub = region[hs, ws]
            sub_real = real[hs, ws]
            sub[sub_real] = idx
            region[hs, ws] = sub
            idx += 1
    region = torch.roll(region, shifts=(-sh, -sw), dims=(0, 1))
    nH, nW = Hp // w, Wp // w
    region = region.view(nH, w, nW, w).permute(0, 2, 1, 3).contiguous()
    return region.view(nH * nW, w * w)


def _shifted_attention_mask_2d(
    H: int,
    W: int,
    Hp: int,
    Wp: int,
    window_size: int,
    shift_h: int,
    shift_w: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Standard Swin shifted-window bool mask for 2D SDPA.

    This is the high-throughput path used when the attention window is small
    enough to materialize the per-window N x N mask. It also handles padded
    tokens by masking padded keys from real queries; padded queries are cropped
    after reverse shift, so they are allowed to attend to real keys only to
    keep SDPA rows finite.
    """

    region = _shifted_region_ids_2d(
        H, W, Hp, Wp, window_size, shift_h, shift_w, device)
    if region is None:
        return None
    q_real = region.unsqueeze(2) >= 0
    k_real = region.unsqueeze(1) >= 0
    same_region = region.unsqueeze(1) == region.unsqueeze(2)
    allowed = same_region & q_real & k_real
    return torch.where(q_real, allowed, k_real.expand_as(allowed))


def _shifted_region_ids_3d(
    T: int,
    H: int,
    W: int,
    Hp: int,
    Wp: int,
    window_H: int,
    window_W: int,
    shift_h: int,
    shift_w: int,
    device: torch.device,
) -> torch.Tensor | None:
    # window_H == window_W for current Swin3D callers; keep the assertion
    # explicit so a future rectangular-window change updates this helper.
    if int(window_H) != int(window_W):
        raise ValueError("shifted 3D region ids currently require square spatial windows")
    spatial = _shifted_region_ids_2d(
        H, W, Hp, Wp, window_H, shift_h, shift_w, device)
    if spatial is None:
        return None
    return spatial.unsqueeze(1).expand(-1, T, -1).reshape(spatial.shape[0], T * spatial.shape[1])


def _sdpa_with_region_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    region_ids: torch.Tensor | None,
) -> torch.Tensor:
    """SDPA with standard shifted-window region isolation.

    `region_ids` has shape (num_windows, tokens). Tokens attend only to other
    tokens in the same non-negative region. Windows that are a single region
    are batched through normal SDPA; only wrapped/padded edge windows are split
    into smaller SDPA calls.
    """

    if region_ids is None:
        return sdpa(
            q, k, v, dropout_p=0.0, is_causal=False)
    if region_ids.shape != (q.shape[0], q.shape[2]):
        raise ValueError(
            f"region_ids shape {tuple(region_ids.shape)} does not match "
            f"attention windows/tokens {(q.shape[0], q.shape[2])}")

    out = torch.zeros_like(q)
    valid = region_ids >= 0
    first = region_ids[:, :1]
    full = valid.all(dim=1) & (region_ids == first).all(dim=1)
    full_idx = torch.nonzero(full, as_tuple=False).flatten()
    if full_idx.numel() > 0:
        out.index_copy_(
            0,
            full_idx,
            sdpa(
                q.index_select(0, full_idx),
                k.index_select(0, full_idx),
                v.index_select(0, full_idx),
                dropout_p=0.0,
                is_causal=False,
            ),
        )

    partial = torch.nonzero(~full, as_tuple=False).flatten()
    if partial.numel() == 0:
        return out

    # Shift masks have only a small number of repeated token-region layouts
    # across edge windows. Batch identical token slices together instead of
    # launching one SDPA call per window/region.
    groups: dict[tuple[int, ...], list[int]] = {}
    partial_cpu = region_ids.index_select(0, partial).detach().cpu()
    partial_list = partial.detach().cpu().tolist()
    for local_i, win_idx in enumerate(partial_list):
        row = partial_cpu[local_i]
        for label in torch.unique(row).tolist():
            if label < 0:
                continue
            token_tuple = tuple(torch.nonzero(row == label, as_tuple=False)
                                .flatten().tolist())
            if token_tuple:
                groups.setdefault(token_tuple, []).append(int(win_idx))

    out_nlhd = out.permute(0, 2, 1, 3)
    for token_tuple, win_list in groups.items():
        win_idx = torch.tensor(win_list, device=q.device, dtype=torch.long)
        token_idx = torch.tensor(token_tuple, device=q.device, dtype=torch.long)
        attn = sdpa(
            q.index_select(0, win_idx).index_select(2, token_idx),
            k.index_select(0, win_idx).index_select(2, token_idx),
            v.index_select(0, win_idx).index_select(2, token_idx),
            dropout_p=0.0,
            is_causal=False,
        )
        out_nlhd[win_idx[:, None], token_idx[None, :]] = attn.permute(0, 2, 1, 3)
    return out


@lru_cache(maxsize=512)
def _shifted_region_groups_2d_cached(
    H: int,
    W: int,
    Hp: int,
    Wp: int,
    window_size: int,
    shift_h: int,
    shift_w: int,
) -> tuple[int, tuple[int, ...], tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]]:
    region = _shifted_region_ids_2d(
        H, W, Hp, Wp, window_size, shift_h, shift_w, torch.device('cpu'))
    if region is None:
        return 0, (), ()
    num_windows = int(region.shape[0])
    valid = region >= 0
    first = region[:, :1]
    full = valid.all(dim=1) & (region == first).all(dim=1)
    full_idx = tuple(torch.nonzero(full, as_tuple=False).flatten().tolist())

    groups: dict[tuple[int, ...], list[int]] = {}
    partial = torch.nonzero(~full, as_tuple=False).flatten().tolist()
    for win_idx in partial:
        row = region[win_idx]
        for label in torch.unique(row).tolist():
            if label < 0:
                continue
            token_tuple = tuple(torch.nonzero(row == label, as_tuple=False)
                                .flatten().tolist())
            if token_tuple:
                groups.setdefault(token_tuple, []).append(int(win_idx))
    return num_windows, full_idx, tuple(
        (token_tuple, tuple(win_list))
        for token_tuple, win_list in groups.items()
    )


@lru_cache(maxsize=512)
def _shifted_region_groups_3d_cached(
    T: int,
    H: int,
    W: int,
    Hp: int,
    Wp: int,
    window_H: int,
    window_W: int,
    shift_h: int,
    shift_w: int,
) -> tuple[int, tuple[int, ...], tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]]:
    if int(window_H) != int(window_W):
        raise ValueError("shifted 3D region groups currently require square spatial windows")
    spatial = _shifted_region_ids_2d(
        H, W, Hp, Wp, window_H, shift_h, shift_w, torch.device('cpu'))
    if spatial is None:
        return 0, (), ()
    num_windows = int(spatial.shape[0])
    valid = spatial >= 0
    first = spatial[:, :1]
    full = valid.all(dim=1) & (spatial == first).all(dim=1)
    full_idx = tuple(torch.nonzero(full, as_tuple=False).flatten().tolist())

    spatial_tokens = window_H * window_W
    groups: dict[tuple[int, ...], list[int]] = {}
    partial = torch.nonzero(~full, as_tuple=False).flatten().tolist()
    for win_idx in partial:
        row = spatial[win_idx]
        for label in torch.unique(row).tolist():
            if label < 0:
                continue
            sp = torch.nonzero(row == label, as_tuple=False).flatten().tolist()
            if not sp:
                continue
            token_tuple = tuple(
                t * spatial_tokens + int(idx)
                for t in range(int(T))
                for idx in sp
            )
            groups.setdefault(token_tuple, []).append(int(win_idx))
    return num_windows, full_idx, tuple(
        (token_tuple, tuple(win_list))
        for token_tuple, win_list in groups.items()
    )


def _sdpa_with_region_groups(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_windows: int,
    full_idx: tuple[int, ...],
    groups: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
) -> torch.Tensor:
    if num_windows <= 0:
        return sdpa(
            q, k, v, dropout_p=0.0, is_causal=False)
    if q.shape[0] % num_windows != 0:
        raise ValueError(
            f"cached window count {num_windows} does not divide attention "
            f"window batch {q.shape[0]}")
    batch_windows = q.shape[0] // num_windows

    def _batched_windows(base: tuple[int, ...]) -> torch.Tensor:
        idx = torch.tensor(base, device=q.device, dtype=torch.long)
        if batch_windows == 1 or idx.numel() == 0:
            return idx
        offsets = torch.arange(
            batch_windows, device=q.device, dtype=torch.long) * num_windows
        return (idx.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)

    out = torch.zeros_like(q)
    if full_idx:
        win_idx = _batched_windows(full_idx)
        out.index_copy_(
            0,
            win_idx,
            sdpa(
                q.index_select(0, win_idx),
                k.index_select(0, win_idx),
                v.index_select(0, win_idx),
                dropout_p=0.0,
                is_causal=False,
            ),
        )
    if not groups:
        return out

    out_nlhd = out.permute(0, 2, 1, 3)
    for token_tuple, win_tuple in groups:
        win_idx = _batched_windows(win_tuple)
        token_idx = torch.tensor(token_tuple, device=q.device, dtype=torch.long)
        attn = sdpa(
            q.index_select(0, win_idx).index_select(2, token_idx),
            k.index_select(0, win_idx).index_select(2, token_idx),
            v.index_select(0, win_idx).index_select(2, token_idx),
            dropout_p=0.0,
            is_causal=False,
        )
        out_nlhd[win_idx[:, None], token_idx[None, :]] = attn.permute(0, 2, 1, 3)
    return out


# ---------------------------------------------------------------------------
# Attention cores
# ---------------------------------------------------------------------------


class _QKVProj(nn.Module):
    """Fused QKV projection."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (..., N, dim). Returns q, k, v each (..., num_heads, N, head_dim)."""
        *B_dims, N, _ = x.shape
        qkv = self.qkv(x).reshape(*B_dims, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(*range(len(B_dims)), -3, -2, -4, -1).contiguous()
        q, k, v = qkv.unbind(dim=-4)   # each (..., num_heads, N, head_dim)
        return q, k, v


class HeadwiseAttentionGate(nn.Module):
    """Query-conditioned scalar gate per attention head.

    The gate is applied to SDPA output while heads are still separated:
    (..., heads, N, head_dim). It uses a bias-free normal linear init,
    matching the paper's
    "extra q-projection logits" setup more closely than a hand-opened gate.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.proj = nn.Linear(self.dim, self.num_heads, bias=False)

    def forward(self, query_tokens: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        """query_tokens: (..., N, C), attn: (..., heads, N, head_dim)."""
        gate = torch.sigmoid(self.proj(query_tokens))
        gate = gate.transpose(-1, -2).unsqueeze(-1).to(dtype=attn.dtype)
        return attn * gate

    def grid(self, query_grid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """query_grid: (..., C). Returns (..., heads, 1)."""
        gate = torch.sigmoid(self.proj(query_grid)).to(dtype=dtype)
        return gate.unsqueeze(-1)


class QKVDepthwiseSpatial(nn.Module):
    """Depthwise spatial filter on fused QKV maps in NHWC/NTHWC layout.

    The kernel is initialized to exact identity, so enabling this module is
    nondisruptive for existing weights while still giving every off-center
    tap an immediate gradient path.
    """

    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        assert kernel_size > 0 and kernel_size % 2 == 1, (
            f"kernel_size must be a positive odd integer; got {kernel_size}")
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.pad = self.kernel_size // 2
        channels = 3 * self.dim
        self.dw = nn.Conv2d(channels, channels, kernel_size=self.kernel_size,
                            padding=0, groups=channels, bias=True)
        with torch.no_grad():
            self.dw.weight.zero_()
            self.dw.weight[:, 0, self.pad, self.pad] = 1.0
            if self.dw.bias is not None:
                self.dw.bias.zero_()

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        if qkv.dim() == 5:
            B, T, H, W, C3 = qkv.shape
            y = qkv.reshape(B * T, H, W, C3)
            out_shape = (B, T, H, W, C3)
        elif qkv.dim() == 4:
            B, H, W, C3 = qkv.shape
            y = qkv
            out_shape = (B, H, W, C3)
        else:
            raise ValueError(
                f"QKVDepthwiseSpatial expects NHWC or NTHWC, got {tuple(qkv.shape)}")
        if C3 != 3 * self.dim:
            raise ValueError(f"QKV channel count {C3} != 3*dim {3 * self.dim}")
        y = y.permute(0, 3, 1, 2)
        mode = 'reflect' if H > self.pad and W > self.pad else 'replicate'

        elems_per_item = C3 * (H + 2 * self.pad) * (W + 2 * self.pad)
        max_items = max(1, 1_000_000_000 // max(1, elems_per_item))

        def apply_dw(z: torch.Tensor) -> torch.Tensor:
            z = F.pad(z, (self.pad, self.pad, self.pad, self.pad), mode=mode)
            return self.dw(z)

        if y.shape[0] > max_items:
            y = torch.cat([
                apply_dw(y[i:i + max_items])
                for i in range(0, y.shape[0], max_items)
            ], dim=0)
        else:
            y = apply_dw(y)
        y = y.permute(0, 2, 3, 1)
        return y.reshape(out_shape)


class SwinAttention2D(nn.Module):
    """2D Swin window attention with 2D RoPE on Q, K.

    Windows are non-overlapping; shift between consecutive blocks is the
    caller's responsibility (via cyclic-shift before/after this module).
    """

    def __init__(self, dim: int, num_heads: int, window_size: int,
                 rope: Optional[RoPE2D] = None,
                 qkv_bias: bool = True,
                 qk_norm: bool = True,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.qkv = _QKVProj(dim, num_heads, qkv_bias=qkv_bias)
        self.qkv_dw = (
            QKVDepthwiseSpatial(dim, kernel_size=int(qkv_dw_kernel))
            if qkv_dw_kernel and qkv_dw_kernel > 0 else None
        )
        self.attn_gate = HeadwiseAttentionGate(dim, num_heads)
        # QK-norm: RMSNorm applied to Q and K AFTER the qkv linear, BEFORE
        # the dot product. Stabilizes attention at scale and in bf16 (where
        # a large QK dot-product can overflow). Default on.
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        # Zero-init output proj so residual branch contributes 0 at init.
        # Paired with removal of LayerScale in SwinBlock2D.
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.rope = rope or RoPE2D(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        region_groups: tuple[int, tuple[int, ...], tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]] | None = None,
        region_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, H, W, C). Returns (B, H, W, C).

        If (H, W) doesn't divide window_size, pads with zeros at bottom/right
        up to the next multiple and masks padded keys out of attention. The
        output is cropped back to (B, H, W, C).
        """
        B, H, W, C = x.shape
        w = self.window_size
        pad_h = (-H) % w
        pad_w = (-W) % w
        has_pad = pad_h != 0 or pad_w != 0
        x_orig = x
        if has_pad and self.qkv_dw is None:
            # F.pad arg order for (B, H, W, C): (C_l, C_r, W_l, W_r, H_l, H_r)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        if self.qkv_dw is not None:
            N = w * w
            nW_total = B * (Hp // w) * (Wp // w)
            y = self.qkv.qkv(x_orig)
            y = self.qkv_dw(y)
            if has_pad:
                y = F.pad(y, (0, 0, 0, pad_w, 0, pad_h))
            windows, _ = window_partition_2d(y, w)
            qkv = (windows.reshape(nW_total, N, 3, self.num_heads, self.head_dim)
                          .permute(2, 0, 3, 1, 4)
                          .contiguous())
            q, k, v = qkv.unbind(dim=0)
            gate_src = x_orig
            if has_pad:
                gate_src = F.pad(gate_src, (0, 0, 0, pad_w, 0, pad_h))
            gate_windows, _ = window_partition_2d(gate_src, w)
            gate_tokens = gate_windows.reshape(nW_total, N, C)
        else:
            windows, _ = window_partition_2d(x, w)
            nW_total = windows.shape[0]
            N = w * w
            tokens = windows.reshape(nW_total, N, C)
            q, k, v = self.qkv(tokens)
            gate_tokens = tokens
        # q, k, v: (nW_total, num_heads, N, head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope.get_cos_sin(w, w, device=x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if region_groups is not None:
            attn = _sdpa_with_region_groups(q, k, v, *region_groups)
        elif attn_mask is not None:
            nW = int(attn_mask.shape[0])
            if q.shape[0] % nW != 0:
                raise ValueError(
                    f"attention mask windows {nW} do not divide q batch {q.shape[0]}")
            B_eff = q.shape[0] // nW
            q_view = q.view(B_eff, nW, self.num_heads, N, self.head_dim)
            k_view = k.view(B_eff, nW, self.num_heads, N, self.head_dim)
            v_view = v.view(B_eff, nW, self.num_heads, N, self.head_dim)
            attn = sdpa(
                q_view, k_view, v_view,
                attn_mask=attn_mask.unsqueeze(0).unsqueeze(2),
                dropout_p=0.0, is_causal=False)
            attn = attn.reshape(nW_total, self.num_heads, N, self.head_dim)
        elif region_ids is not None:
            attn = _sdpa_with_region_ids(q, k, v, region_ids)
        elif has_pad:
            # Bool key-mask (True = real token). Broadcast over query dim L
            # and heads via (B·nWin, 1, 1, N). Every window has ≥1 real token
            # (pad_{h,w} < w), so softmax is always well-defined.
            real = torch.zeros(Hp, Wp, dtype=torch.bool, device=x.device)
            real[:H, :W] = True
            nH_w, nW_w = Hp // w, Wp // w
            real = real.view(nH_w, w, nW_w, w).permute(0, 2, 1, 3).contiguous()
            per_win_mask = real.view(nH_w * nW_w, N)               # (nWin, N)
            attn_mask = per_win_mask.unsqueeze(0).expand(B, -1, -1).reshape(nW_total, 1, 1, N)
            attn = sdpa(q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=0.0, is_causal=False)
        else:
            attn = sdpa(q, k, v, dropout_p=0.0, is_causal=False)

        attn = self.attn_gate(gate_tokens, attn)
        attn = attn.transpose(1, 2).reshape(nW_total, N, C)
        attn = self.proj(attn)

        out_w = attn.reshape(nW_total, w, w, C)
        out = window_unpartition_2d(out_w, w, (Hp, Wp), B)
        if has_pad:
            out = out[:, :H, :W, :].contiguous()
        return out


class HaloAttention2D(nn.Module):
    """Block-halo-overlapped 2D attention (HaloNet, Vaswani et al. 2021).

    Partitions input into non-overlapping `b×b` query blocks. Each block's
    queries attend to a `(b+2h)×(b+2h)` halo'd region of keys/values —
    the block itself plus `h` tokens of context on every side, reaching
    into adjacent blocks (zero-padded at image boundaries).

    Output shape is still per-block `b×b`, so block-level output stride
    is `b`, matching Swin semantics; but the attention context overlaps
    between adjacent blocks by `2h` tokens per axis, so context boundaries
    are soft rather than hard like in Swin.

    Combined with the caller's cyclic-shift alternation in the wrapping
    block (same pattern as Swin), this gives "halo + shift" — within-block
    halo context AND grid-averaging over a pair of blocks. See the model
    design notes.

    h=0 reduces to plain Swin windowed attention (without shift). Use
    from SwinBlock2D with `attn_class=HaloAttention2D` and non-zero halo.

    Masking:
      • Block-alignment padding (right/bottom of the grid) → masked out.
      • Halo tokens that fall outside the original H×W image → masked out
        (they would otherwise contribute the k-projection of zero-padded
        values, i.e. just the k-linear bias, to real queries' attention).
    """

    def __init__(self, dim: int, num_heads: int, block_size: int, halo: int,
                 rope: Optional[RoPE2D] = None,
                 qkv_bias: bool = True, qk_norm: bool = True):
        super().__init__()
        assert halo >= 0 and block_size > 0, f"block_size={block_size} halo={halo}"
        self.dim = dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.halo = halo
        self.head_dim = dim // num_heads
        self.qkv = _QKVProj(dim, num_heads, qkv_bias=qkv_bias)
        self.attn_gate = HeadwiseAttentionGate(dim, num_heads)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        # RoPE lives in the (b+2h)×(b+2h) halo'd coord system. Q uses the
        # central b×b subset of this coord system; K/V use the full set.
        self.rope = rope or RoPE2D(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C). Returns (B, H, W, C)."""
        B, H, W, C = x.shape
        b = self.block_size
        h = self.halo
        kh = b + 2 * h

        # 1. Pad to block alignment on bottom/right.
        pad_h = (-H) % b
        pad_w = (-W) % b
        has_block_pad = pad_h != 0 or pad_w != 0
        if has_block_pad:
            x_aligned = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        else:
            x_aligned = x
        Hp, Wp = H + pad_h, W + pad_w
        nH, nW = Hp // b, Wp // b
        nBlocks = nH * nW

        # 2. QKV projection on the block-aligned (no-halo) input.
        #    Output split into separate q_full, k_full, v_full because the
        #    block extraction differs between Q and K/V.
        heads = self.num_heads
        D = self.head_dim
        qkv = self.qkv.qkv(x_aligned)                     # (B, Hp, Wp, 3·heads·D)
        qkv = qkv.view(B, Hp, Wp, 3, heads, D)
        q_full = qkv[..., 0, :, :]                        # (B, Hp, Wp, heads, D)
        k_full = qkv[..., 1, :, :]
        v_full = qkv[..., 2, :, :]

        # 3. Q: block partition (no halo). Output (B·nBlocks, heads, b², D).
        Q = q_full.view(B, nH, b, nW, b, heads, D)
        Q = Q.permute(0, 1, 3, 5, 2, 4, 6).contiguous()   # (B, nH, nW, heads, b, b, D)
        Q = Q.view(B * nBlocks, heads, b * b, D)
        q_tokens = (x_aligned.view(B, nH, b, nW, b, C)
                             .permute(0, 1, 3, 2, 4, 5)
                             .reshape(B * nBlocks, b * b, C))

        # 4. K, V: halo-pad by `h` on spatial dims, then unfold (b+2h)×(b+2h)
        #    windows with stride b. Zero-pad at boundaries.
        if h > 0:
            k_pad = F.pad(k_full, (0, 0, 0, 0, h, h, h, h))
            v_pad = F.pad(v_full, (0, 0, 0, 0, h, h, h, h))
        else:
            k_pad = k_full
            v_pad = v_full
        # k_pad: (B, Hp+2h, Wp+2h, heads, D)
        # unfold(1, kh, b) → (B, nH, Wp+2h, heads, D, kh)
        # unfold(2, kh, b) → (B, nH, nW, heads, D, kh, kh)
        # permute to (B, nH, nW, heads, kh, kh, D)
        K = k_pad.unfold(1, kh, b).unfold(2, kh, b).permute(0, 1, 2, 3, 5, 6, 4).contiguous()
        V = v_pad.unfold(1, kh, b).unfold(2, kh, b).permute(0, 1, 2, 3, 5, 6, 4).contiguous()
        N_kv = kh * kh
        K = K.view(B * nBlocks, heads, N_kv, D)
        V = V.view(B * nBlocks, heads, N_kv, D)

        # 5. QK-norm + RoPE. K/V use full (kh × kh) coord tables; Q uses the
        #    central b×b subset of those tables.
        if self.qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        cos_kv, sin_kv = self.rope.get_cos_sin(kh, kh, device=x.device)
        # Extract central b×b rows of the flattened (kh·kh, D//2) table.
        device = cos_kv.device
        rows = torch.arange(h, h + b, device=device)
        cols = torch.arange(h, h + b, device=device)
        central_idx = (rows.unsqueeze(1) * kh + cols.unsqueeze(0)).flatten()
        cos_q = cos_kv.index_select(0, central_idx)
        sin_q = sin_kv.index_select(0, central_idx)
        Q = apply_rotary(Q, cos_q, sin_q)
        K = apply_rotary(K, cos_kv, sin_kv)

        # 6. Attention mask for padding + out-of-image halo.
        #    Build an (Hp+2h, Wp+2h) "valid" map (True = inside original
        #    H×W image), unfold into per-block (kh, kh) key masks.
        needs_mask = has_block_pad or h > 0
        if needs_mask:
            valid = torch.zeros(Hp + 2 * h, Wp + 2 * h, dtype=torch.bool, device=x.device)
            valid[h:h + H, h:h + W] = True
            valid_u = valid.unfold(0, kh, b).unfold(1, kh, b)  # (nH, nW, kh, kh)
            valid_u = valid_u.reshape(nBlocks, N_kv)
            attn_mask = valid_u.unsqueeze(0).expand(B, -1, -1).reshape(B * nBlocks, 1, 1, N_kv)
            attn = sdpa(Q, K, V, attn_mask=attn_mask,
                        dropout_p=0.0, is_causal=False)
        else:
            attn = sdpa(Q, K, V, dropout_p=0.0, is_causal=False)

        # 7. Unpack: (B·nBlocks, heads, b², D) → (B, Hp, Wp, C) → crop to (B, H, W, C)
        attn = self.attn_gate(q_tokens, attn)
        attn = attn.transpose(1, 2).reshape(B * nBlocks, b * b, heads * D)  # (BnBlocks, b², C)
        attn = self.proj(attn)
        out = attn.view(B, nH, nW, b, b, heads * D)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, Hp, Wp, heads * D)
        if has_block_pad:
            out = out[:, :H, :W, :].contiguous()
        return out


class SwinAttention3D(nn.Module):
    """3D Swin window attention with 3D RoPE on Q, K.

    Spatial windows are non-overlapping (H_w × W_w); the temporal axis is
    ALWAYS used in full (T_w = T). This means every window has shape
    (T, H_w, W_w) and attention is fully-temporal + windowed-spatial.

    Rationale: at our working resolutions (T in 3–48, spatial grid at /16),
    windowed temporal would either leave temporal context uncovered
    (single-frame windows) or add state-management complexity (temporal
    shifts interacting with variable T). Full-T within-window is cheap —
    attention scales as H_w²·W_w²·T², and at T≤48 with H_w=W_w=8 that's
    under 10M per window per head even at the worst T.
    """

    def __init__(self, dim: int, num_heads: int,
                 window_H: int, window_W: int,
                 rope: Optional[RoPE3D] = None,
                 qkv_bias: bool = True,
                 qk_norm: bool = True,
                 zero_init_proj: bool = True,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_H = window_H
        self.window_W = window_W
        self.qkv = _QKVProj(dim, num_heads, qkv_bias=qkv_bias)
        self.qkv_dw = (
            QKVDepthwiseSpatial(dim, kernel_size=int(qkv_dw_kernel))
            if qkv_dw_kernel and qkv_dw_kernel > 0 else None
        )
        self.attn_gate = HeadwiseAttentionGate(dim, num_heads)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        if zero_init_proj:
            # Zero-init output proj so residual branch contributes 0 at init.
            nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
        self.rope = rope or RoPE3D(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        t_stride: float = 1.0,
        region_groups: tuple[int, tuple[int, ...], tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]] | None = None,
        region_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, T, H, W, C). Returns (B, T, H, W, C).

        If (H, W) doesn't divide the spatial window, pads with zeros at
        bottom/right and masks padded keys out of attention. T is always
        used in full (T_w = T, no temporal padding).
        """
        B, T, H, W, C = x.shape
        T_w = T
        H_w = self.window_H
        W_w = self.window_W
        pad_h = (-H) % H_w
        pad_w = (-W) % W_w
        has_pad = pad_h != 0 or pad_w != 0
        x_orig = x
        if has_pad and self.qkv_dw is None:
            # F.pad order for (B, T, H, W, C): (C_l, C_r, W_l, W_r, H_l, H_r)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        N = T_w * H_w * W_w
        nH_w, nW_w = Hp // H_w, Wp // W_w
        nW_total = B * nH_w * nW_w
        window_batch = int(getattr(self, 'inference_window_batch', 0) or 0)
        if window_batch <= 0 and not self.training:
            elems_per_window = N * C
            max_windows = max(1, 200_000_000 // max(1, elems_per_window))
            if nW_total > max_windows:
                window_batch = max_windows

        def _slice_region_groups(start: int, end: int):
            if region_groups is None:
                return None
            if B != 1:
                return region_groups
            num_windows, full_idx, groups = region_groups
            if num_windows != nW_total:
                return region_groups
            full = tuple(idx - start for idx in full_idx
                         if start <= idx < end)
            sliced = []
            for token_tuple, win_tuple in groups:
                wins = tuple(idx - start for idx in win_tuple
                             if start <= idx < end)
                if wins:
                    sliced.append((token_tuple, wins))
            return (end - start, full, tuple(sliced))

        pad_attn_mask = None
        if has_pad and region_groups is None and region_ids is None:
            # Spatial validity (per window) tiled across T (T never padded).
            real = torch.zeros(Hp, Wp, dtype=torch.bool, device=x.device)
            real[:H, :W] = True
            real = real.view(nH_w, H_w, nW_w, W_w).permute(0, 2, 1, 3).contiguous()
            per_win_hw = real.view(nH_w * nW_w, H_w * W_w)
            per_win_mask = per_win_hw.unsqueeze(1).expand(-1, T_w, -1)
            per_win_mask = per_win_mask.reshape(nH_w * nW_w, N)
            pad_attn_mask = per_win_mask.unsqueeze(0).expand(
                B, -1, -1).reshape(nW_total, 1, 1, N)

        # Local positions per window: t=0..T-1 (full), h=0..H_w-1, w=0..W_w-1.
        # Temporal inter-window info isn't needed since T_w=T (no temporal
        # partitioning). Spatial inter-window info is handled by the
        # SwinBlock3D wrapper's cyclic shift.
        cos, sin = self.rope.get_cos_sin(T_w, H_w, W_w, device=x.device,
                                         t_stride=t_stride)

        def _take_windows(grid: torch.Tensor, start: int, end: int) -> torch.Tensor:
            # grid: (B, T, Hp, Wp, Cg). Return only requested spatial windows.
            Cg = grid.shape[-1]
            view = grid.view(B, T_w, nH_w, H_w, nW_w, W_w, Cg)
            out = []
            for idx in range(start, end):
                b = idx // (nH_w * nW_w)
                sp = idx - b * (nH_w * nW_w)
                ih = sp // nW_w
                iw = sp - ih * nW_w
                out.append(view[b, :, ih, :, iw, :, :])
            return torch.stack(out, dim=0).contiguous()

        def _attend_windows(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            gate_tokens: torch.Tensor,
                            start: int, end: int) -> torch.Tensor:
            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
            if region_groups is not None:
                groups = _slice_region_groups(start, end)
                attn = _sdpa_with_region_groups(q, k, v, *groups)
            elif region_ids is not None:
                attn = _sdpa_with_region_ids(q, k, v, region_ids[start:end])
            elif pad_attn_mask is not None:
                attn = sdpa(
                    q, k, v, attn_mask=pad_attn_mask[start:end],
                    dropout_p=0.0, is_causal=False)
            else:
                attn = sdpa(
                    q, k, v, dropout_p=0.0, is_causal=False)
            attn = self.attn_gate(gate_tokens, attn)
            attn = attn.transpose(1, 2).reshape(end - start, N, C)
            return self.proj(attn)

        use_lazy_window_path = (
            window_batch > 0 and not self.training and nW_total > window_batch
            and (region_groups is None or B == 1)
        )
        if use_lazy_window_path:
            gate_src = x_orig
            if has_pad:
                gate_src = F.pad(gate_src, (0, 0, 0, pad_w, 0, pad_h))
            qkv_grid = None
            if self.qkv_dw is not None:
                frame_chunk = int(
                    getattr(self, 'inference_qkv_frame_chunk', 0) or 0)
                frame_chunk = max(1, frame_chunk) if frame_chunk > 0 else T_w
                for t_start in range(0, T_w, frame_chunk):
                    t_end = min(t_start + frame_chunk, T_w)
                    y = self.qkv.qkv(x_orig[:, t_start:t_end])
                    y = self.qkv_dw(y)
                    if has_pad:
                        y = F.pad(y, (0, 0, 0, pad_w, 0, pad_h))
                    if qkv_grid is None:
                        qkv_grid = y.new_empty(B, T_w, Hp, Wp, 3 * C)
                    qkv_grid[:, t_start:t_end] = y
            else:
                qkv_grid = None

            attn = gate_src.new_empty((nW_total, N, C))
            for start in range(0, nW_total, window_batch):
                end = min(start + window_batch, nW_total)
                gate_tokens = _take_windows(
                    gate_src, start, end).reshape(end - start, N, C)
                if self.qkv_dw is not None:
                    qkv_win = _take_windows(qkv_grid, start, end)
                    qkv = (qkv_win.reshape(
                                end - start, N, 3, self.num_heads,
                                self.head_dim)
                           .permute(2, 0, 3, 1, 4)
                           .contiguous())
                    q, k, v = qkv.unbind(dim=0)
                else:
                    q, k, v = self.qkv(gate_tokens)
                attn[start:end] = _attend_windows(
                    q, k, v, gate_tokens, start, end)

            out_w = attn.reshape(nW_total, T_w, H_w, W_w, C)
            out = window_unpartition_3d(out_w, T_w, H_w, W_w, (T, Hp, Wp), B)
            if has_pad:
                out = out[:, :, :H, :W, :].contiguous()
            return out

        if self.qkv_dw is not None:
            y = self.qkv.qkv(x_orig)
            y = self.qkv_dw(y)
            if has_pad:
                y = F.pad(y, (0, 0, 0, pad_w, 0, pad_h))
            windows, _ = window_partition_3d(y, T_w, H_w, W_w)
            qkv_tokens = windows.reshape(
                nW_total, N, 3, self.num_heads, self.head_dim)
            gate_src = x_orig
            if has_pad:
                gate_src = F.pad(gate_src, (0, 0, 0, pad_w, 0, pad_h))
            gate_windows, _ = window_partition_3d(gate_src, T_w, H_w, W_w)
            gate_tokens = gate_windows.reshape(nW_total, N, C)
        else:
            windows, _ = window_partition_3d(x, T_w, H_w, W_w)
            nW_total = windows.shape[0]
            tokens = windows.reshape(nW_total, N, C)
            gate_tokens = tokens

        def _run_window_slice(start: int, end: int) -> torch.Tensor:
            if self.qkv_dw is not None:
                qkv = qkv_tokens[start:end].permute(
                    2, 0, 3, 1, 4).contiguous()
                q, k, v = qkv.unbind(dim=0)
            else:
                q, k, v = self.qkv(tokens[start:end])
            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
            if region_groups is not None:
                groups = _slice_region_groups(start, end)
                attn = _sdpa_with_region_groups(q, k, v, *groups)
            elif region_ids is not None:
                attn = _sdpa_with_region_ids(q, k, v, region_ids[start:end])
            elif pad_attn_mask is not None:
                attn = sdpa(
                    q, k, v, attn_mask=pad_attn_mask[start:end],
                    dropout_p=0.0, is_causal=False)
            else:
                attn = sdpa(
                    q, k, v, dropout_p=0.0, is_causal=False)
            attn = self.attn_gate(gate_tokens[start:end], attn)
            attn = attn.transpose(1, 2).reshape(end - start, N, C)
            return self.proj(attn)

        can_batch_windows = (
            window_batch > 0 and not self.training and nW_total > window_batch
            and (region_groups is None or B == 1)
        )
        if can_batch_windows:
            attn = gate_tokens.new_empty((nW_total, N, C))
            for start in range(0, nW_total, window_batch):
                end = min(start + window_batch, nW_total)
                attn[start:end] = _run_window_slice(start, end)
        else:
            attn = _run_window_slice(0, nW_total)
        out_w = attn.reshape(nW_total, T_w, H_w, W_w, C)
        out = window_unpartition_3d(out_w, T_w, H_w, W_w, (T, Hp, Wp), B)
        if has_pad:
            out = out[:, :, :H, :W, :].contiguous()
        return out


class HaloAttention3D(nn.Module):
    """Block-halo-overlapped 3D attention (spatial-halo, full-temporal).

    Spatial extension of HaloAttention2D: H, W are block-partitioned at
    stride `block_H`, `block_W` with halo `h` on each side; T is always
    used in full (matches Swin-3D's full-T convention). Each block's
    queries are `b_H · b_W · T` tokens attending to
    `(b_H+2h) · (b_W+2h) · T` tokens.

    h=0 reduces to plain Swin-3D windowed attention (without shift).
    """

    def __init__(self, dim: int, num_heads: int,
                 block_H: int, block_W: int, halo: int,
                 rope: Optional[RoPE3D] = None,
                 qkv_bias: bool = True, qk_norm: bool = True):
        super().__init__()
        assert halo >= 0, f"halo={halo} must be ≥ 0"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_H = block_H
        self.block_W = block_W
        self.halo = halo
        self.qkv = _QKVProj(dim, num_heads, qkv_bias=qkv_bias)
        self.attn_gate = HeadwiseAttentionGate(dim, num_heads)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.rope = rope or RoPE3D(self.head_dim)

    def forward(self, x: torch.Tensor, t_stride: float = 1.0) -> torch.Tensor:
        """x: (B, T, H, W, C). Returns (B, T, H, W, C)."""
        B, T, H, W, C = x.shape
        bH, bW = self.block_H, self.block_W
        h = self.halo
        khH, khW = bH + 2 * h, bW + 2 * h
        heads, D = self.num_heads, self.head_dim

        # 1. Block-align pad on H, W (T stays full).
        pad_h = (-H) % bH
        pad_w = (-W) % bW
        has_block_pad = pad_h != 0 or pad_w != 0
        if has_block_pad:
            x_a = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        else:
            x_a = x
        Hp, Wp = H + pad_h, W + pad_w
        nH, nW = Hp // bH, Wp // bW
        nBlocks = nH * nW

        # 2. QKV projection.
        qkv = self.qkv.qkv(x_a)                           # (B, T, Hp, Wp, 3·heads·D)
        qkv = qkv.view(B, T, Hp, Wp, 3, heads, D)
        q_full = qkv[..., 0, :, :]                        # (B, T, Hp, Wp, heads, D)
        k_full = qkv[..., 1, :, :]
        v_full = qkv[..., 2, :, :]

        # 3. Q: block partition (no halo). Per block: T·bH·bW tokens.
        # (B, T, Hp, Wp, heads, D) → (B, T, nH, bH, nW, bW, heads, D)
        # → permute to (B, nH, nW, heads, T, bH, bW, D)
        Q = q_full.view(B, T, nH, bH, nW, bW, heads, D)
        Q = Q.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        Q = Q.view(B * nBlocks, heads, T * bH * bW, D)
        q_tokens = (x_a.view(B, T, nH, bH, nW, bW, C)
                       .permute(0, 2, 4, 1, 3, 5, 6)
                       .reshape(B * nBlocks, T * bH * bW, C))

        # 4. K, V: halo pad spatially (T always full), then unfold.
        if h > 0:
            k_pad = F.pad(k_full, (0, 0, 0, 0, h, h, h, h))
            v_pad = F.pad(v_full, (0, 0, 0, 0, h, h, h, h))
        else:
            k_pad = k_full
            v_pad = v_full
        # Shape: (B, T, Hp+2h, Wp+2h, heads, D)
        # unfold(2, khH, bH) → (B, T, nH, Wp+2h, heads, D, khH)
        # unfold(3, khW, bW) → (B, T, nH, nW, heads, D, khH, khW)
        # permute to (B, nH, nW, heads, T, khH, khW, D)
        K = k_pad.unfold(2, khH, bH).unfold(3, khW, bW)
        V = v_pad.unfold(2, khH, bH).unfold(3, khW, bW)
        K = K.permute(0, 2, 3, 4, 1, 6, 7, 5).contiguous()
        V = V.permute(0, 2, 3, 4, 1, 6, 7, 5).contiguous()
        N_kv = T * khH * khW
        K = K.view(B * nBlocks, heads, N_kv, D)
        V = V.view(B * nBlocks, heads, N_kv, D)

        # 5. QK-norm + RoPE. K/V use full (T, khH, khW) coord tables.
        #    Q uses (T full, H in [h, h+bH), W in [h, h+bW)) subset.
        if self.qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        cos_kv, sin_kv = self.rope.get_cos_sin(T, khH, khW,
                                                device=x.device,
                                                t_stride=t_stride)
        dev = cos_kv.device
        t_idx = torch.arange(T, device=dev)
        r_idx = torch.arange(h, h + bH, device=dev)
        c_idx = torch.arange(h, h + bW, device=dev)
        # Flat row-major index into (T, khH, khW) grid:
        flat_q_idx = (t_idx[:, None, None] * (khH * khW)
                      + r_idx[None, :, None] * khW
                      + c_idx[None, None, :]).reshape(-1)
        cos_q = cos_kv.index_select(0, flat_q_idx)
        sin_q = sin_kv.index_select(0, flat_q_idx)
        Q = apply_rotary(Q, cos_q, sin_q)
        K = apply_rotary(K, cos_kv, sin_kv)

        # 6. Attention mask for block-pad + out-of-image halo.
        needs_mask = has_block_pad or h > 0
        if needs_mask:
            valid_hw = torch.zeros(Hp + 2 * h, Wp + 2 * h,
                                    dtype=torch.bool, device=x.device)
            valid_hw[h:h + H, h:h + W] = True
            valid_u = valid_hw.unfold(0, khH, bH).unfold(1, khW, bW)  # (nH, nW, khH, khW)
            valid_u = valid_u.reshape(nBlocks, khH * khW)
            # Tile over T (T always valid).
            valid_tiled = valid_u.unsqueeze(1).expand(-1, T, -1).reshape(nBlocks, N_kv)
            attn_mask = valid_tiled.unsqueeze(0).expand(B, -1, -1).reshape(B * nBlocks, 1, 1, N_kv)
            attn = sdpa(Q, K, V, attn_mask=attn_mask,
                        dropout_p=0.0, is_causal=False)
        else:
            attn = sdpa(Q, K, V,
                        dropout_p=0.0, is_causal=False)

        # 7. Unpack: (B·nBlocks, heads, T·bH·bW, D) → (B, T, H, W, C)
        attn = self.attn_gate(q_tokens, attn)
        attn = attn.transpose(1, 2).reshape(B * nBlocks, T * bH * bW, heads * D)
        attn = self.proj(attn)
        out = attn.view(B, nH, nW, T, bH, bW, heads * D)
        out = out.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
        out = out.view(B, T, Hp, Wp, heads * D)
        if has_block_pad:
            out = out[:, :, :H, :W, :].contiguous()
        return out


class AxialAttention2D(nn.Module):
    """Axial (pure row or pure column) 2D attention.

    For `axis='H'` (VERTICAL): each token attends to all tokens in its own
    column (same W, all H). Input reshape: (B, H, W, C) → (B·W, H, C).
    For `axis='W'` (HORIZONTAL): row-wise. (B, H, W, C) → (B·H, W, C).

    Uses shared RoPE2D with a trivial (size-1) axis on the un-attended
    direction — half the head_dim contributes positional modulation along
    the attended axis, half is pure dot-product. Augmentations (rescale,
    jitter) from `refresh_augs()` apply uniformly alongside the dense
    Swin blocks using the same RoPE instance.
    """

    def __init__(self, dim: int, num_heads: int, axis: str,
                 rope: Optional[RoPE2D] = None,
                 qkv_bias: bool = True, qk_norm: bool = True):
        super().__init__()
        assert axis in ('H', 'W'), f"axis must be 'H' or 'W', got {axis!r}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.axis = axis
        self.qkv = _QKVProj(dim, num_heads, qkv_bias=qkv_bias)
        self.attn_gate = HeadwiseAttentionGate(dim, num_heads)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.rope = rope or RoPE2D(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C). Returns (B, H, W, C)."""
        B, H, W, C = x.shape
        # Reshape so the attended axis is the sequence dim; the other spatial
        # axis joins the batch. Nothing else needs padding — any H or W works.
        if self.axis == 'H':
            # Columns: each of W columns has H tokens.
            # (B, H, W, C) → (B, W, H, C) → (B·W, H, C)
            x_strips = x.permute(0, 2, 1, 3).reshape(B * W, H, C)
            N = H
            num_h_rope, num_w_rope = H, 1
        else:  # 'W'
            # Rows: each of H rows has W tokens.
            # (B, H, W, C) → (B·H, W, C)
            x_strips = x.reshape(B * H, W, C)
            N = W
            num_h_rope, num_w_rope = 1, W

        q, k, v = self.qkv(x_strips)                 # (B·M, heads, N, head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope.get_cos_sin(num_h_rope, num_w_rope, device=x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        attn = sdpa(q, k, v, dropout_p=0.0, is_causal=False)
        attn = self.attn_gate(x_strips, attn)
        # (B·M, heads, N, head_dim) → (B·M, N, C)
        attn = attn.transpose(1, 2).reshape(-1, N, C)
        attn = self.proj(attn)

        # Un-reshape back to (B, H, W, C)
        if self.axis == 'H':
            out = attn.view(B, W, H, C).permute(0, 2, 1, 3).contiguous()
        else:
            out = attn.view(B, H, W, C)
        return out


class AxialAttention3D(nn.Module):
    """Axial 3D attention with full T.

    For `axis='H'` (VERTICAL): each token attends to all (t', h', w_fixed)
    with t' ∈ [0,T), h' ∈ [0,H) — i.e., T·H tokens per column-strip.
    For `axis='W'` (HORIZONTAL): per row-strip of T·W tokens.

    Uses shared RoPE3D with num=1 on the un-attended spatial axis. T is
    always full, scaled by `t_stride` just like Swin-3D and FullST.
    """

    def __init__(self, dim: int, num_heads: int, axis: str,
                 rope: Optional[RoPE3D] = None,
                 qkv_bias: bool = True, qk_norm: bool = True):
        super().__init__()
        assert axis in ('H', 'W'), f"axis must be 'H' or 'W', got {axis!r}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.axis = axis
        self.qkv = _QKVProj(dim, num_heads, qkv_bias=qkv_bias)
        self.attn_gate = HeadwiseAttentionGate(dim, num_heads)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.rope = rope or RoPE3D(self.head_dim)

    def forward(self, x: torch.Tensor, t_stride: float = 1.0) -> torch.Tensor:
        """x: (B, T, H, W, C). Returns (B, T, H, W, C)."""
        B, T, H, W, C = x.shape
        if self.axis == 'H':
            # Strip per W column: (T·H) tokens. Reshape:
            # (B, T, H, W, C) → (B, W, T, H, C) → (B·W, T·H, C)
            x_strips = x.permute(0, 3, 1, 2, 4).reshape(B * W, T * H, C)
            N = T * H
            num_t_rope, num_h_rope, num_w_rope = T, H, 1
        else:  # 'W'
            # (B, T, H, W, C) → (B, H, T, W, C) → (B·H, T·W, C)
            x_strips = x.permute(0, 2, 1, 3, 4).reshape(B * H, T * W, C)
            N = T * W
            num_t_rope, num_h_rope, num_w_rope = T, 1, W

        q, k, v = self.qkv(x_strips)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope.get_cos_sin(num_t_rope, num_h_rope, num_w_rope,
                                          device=x.device, t_stride=t_stride)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        attn = sdpa(q, k, v, dropout_p=0.0, is_causal=False)
        attn = self.attn_gate(x_strips, attn)
        attn = attn.transpose(1, 2).reshape(-1, N, C)
        attn = self.proj(attn)

        if self.axis == 'H':
            out = attn.view(B, W, T, H, C).permute(0, 2, 3, 1, 4).contiguous()
        else:
            out = attn.view(B, H, T, W, C).permute(0, 2, 1, 3, 4).contiguous()
        return out


class CSwinAttention2D(nn.Module):
    """Cross-shaped window attention (CSwin, Dong et al. 2022) — 2D.

    Heads split evenly: the first half attend along H (vertical stripes —
    each token sees its full column), the second half attend along W
    (horizontal stripes — each token sees its full row). Within one
    forward pass every output token has been mixed along BOTH axes
    (different heads carry the different mixings, then the output proj
    blends them together).

    State-dict layout intentionally identical to AxialAttention2D:
      qkv.qkv.{weight,bias}, q_norm.weight, k_norm.weight, proj.{weight,bias}
    This lets a weight-permutation surgery convert an axial checkpoint
    into a CSwin one without remapping keys.
    """

    def __init__(self, dim: int, num_heads: int,
                 rope: Optional[RoPE2D] = None,
                 qkv_bias: bool = True, qk_norm: bool = True,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        assert num_heads % 2 == 0, (
            f"CSwinAttention2D requires num_heads % 2 == 0, got {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.h_heads = num_heads // 2           # V-stripe heads (attend along H)
        self.w_heads = num_heads - self.h_heads  # H-stripe heads (attend along W)
        self.head_dim = dim // num_heads
        self.qkv = _QKVProj(dim, num_heads, qkv_bias=qkv_bias)
        self.qkv_dw = (
            QKVDepthwiseSpatial(dim, kernel_size=int(qkv_dw_kernel))
            if qkv_dw_kernel and qkv_dw_kernel > 0 else None
        )
        self.attn_gate = HeadwiseAttentionGate(dim, num_heads)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.rope = rope or RoPE2D(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C). Returns (B, H, W, C)."""
        B, H, W, C = x.shape
        h_heads, w_heads, D = self.h_heads, self.w_heads, self.head_dim

        # QKV is pointwise unless qkv_dw is enabled; the optional DW filter
        # runs on the full spatial map before strip reshape.
        if self.qkv_dw is not None:
            y = self.qkv_dw(self.qkv.qkv(x))
            qkv = (y.reshape(B, H * W, 3, self.num_heads, D)
                    .permute(2, 0, 3, 1, 4)
                    .contiguous())
            q, k, v = qkv.unbind(dim=0)
        else:
            x_flat = x.reshape(B, H * W, C)
            q, k, v = self.qkv(x_flat)                          # each (B, num_heads, H*W, D)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Split heads into V-stripe (attend H) and H-stripe (attend W) groups.
        q_h, k_h, v_h = q[:, :h_heads], k[:, :h_heads], v[:, :h_heads]
        q_w, k_w, v_w = q[:, h_heads:], k[:, h_heads:], v[:, h_heads:]

        # H-group: each of W columns is a strip of H tokens.
        # (B, h_heads, H*W, D) → (B, h_heads, H, W, D) → (B, W, h_heads, H, D) → (B·W, h_heads, H, D)
        def _to_hstrip(t):
            return (t.reshape(B, h_heads, H, W, D)
                     .permute(0, 3, 1, 2, 4)
                     .reshape(B * W, h_heads, H, D))
        q_h, k_h, v_h = _to_hstrip(q_h), _to_hstrip(k_h), _to_hstrip(v_h)

        # W-group: each of H rows is a strip of W tokens.
        def _to_wstrip(t):
            return (t.reshape(B, w_heads, H, W, D)
                     .permute(0, 2, 1, 3, 4)
                     .reshape(B * H, w_heads, W, D))
        q_w, k_w, v_w = _to_wstrip(q_w), _to_wstrip(k_w), _to_wstrip(v_w)

        # RoPE: H-group uses the H-axis coords (num_w=1 → identity on W-chunk);
        #       W-group uses the W-axis coords (num_h=1 → identity on H-chunk).
        cos_h, sin_h = self.rope.get_cos_sin(H, 1, device=x.device)
        cos_w, sin_w = self.rope.get_cos_sin(1, W, device=x.device)
        q_h = apply_rotary(q_h, cos_h, sin_h)
        k_h = apply_rotary(k_h, cos_h, sin_h)
        q_w = apply_rotary(q_w, cos_w, sin_w)
        k_w = apply_rotary(k_w, cos_w, sin_w)

        # Attention along each axis.
        attn_h = sdpa(q_h, k_h, v_h,
                      dropout_p=0.0, is_causal=False)
        attn_w = sdpa(q_w, k_w, v_w,
                      dropout_p=0.0, is_causal=False)

        # Un-strip back to (B, H, W, head_group, D) and concat heads.
        # H-group: (B·W, h_heads, H, D) → (B, W, h_heads, H, D) → (B, H, W, h_heads, D)
        attn_h = (attn_h.reshape(B, W, h_heads, H, D)
                        .permute(0, 3, 1, 2, 4)
                        .contiguous())
        # W-group: (B·H, w_heads, W, D) → (B, H, w_heads, W, D) → (B, H, W, w_heads, D)
        attn_w = (attn_w.reshape(B, H, w_heads, W, D)
                        .permute(0, 1, 3, 2, 4)
                        .contiguous())

        attn = torch.cat([attn_h, attn_w], dim=-2)
        attn = attn * self.attn_gate.grid(x, dtype=attn.dtype)
        attn = attn.reshape(B, H, W, C)
        return self.proj(attn)


class CSwinAttention3D(nn.Module):
    """Cross-shaped window attention — 3D (full temporal + split spatial).

    Heads split evenly: first half attend along H (V-stripes of T·H tokens
    per column), second half attend along W (H-stripes of T·W tokens per
    row). T is always full (matches Swin-3D and FullST conventions).

    State-dict layout intentionally identical to AxialAttention3D.
    """

    def __init__(self, dim: int, num_heads: int,
                 rope: Optional[RoPE3D] = None,
                 qkv_bias: bool = True, qk_norm: bool = True,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        assert num_heads % 2 == 0, (
            f"CSwinAttention3D requires num_heads % 2 == 0, got {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.h_heads = num_heads // 2
        self.w_heads = num_heads - self.h_heads
        self.head_dim = dim // num_heads
        self.qkv = _QKVProj(dim, num_heads, qkv_bias=qkv_bias)
        self.qkv_dw = (
            QKVDepthwiseSpatial(dim, kernel_size=int(qkv_dw_kernel))
            if qkv_dw_kernel and qkv_dw_kernel > 0 else None
        )
        self.attn_gate = HeadwiseAttentionGate(dim, num_heads)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.rope = rope or RoPE3D(self.head_dim)

    def forward(self, x: torch.Tensor, t_stride: float = 1.0) -> torch.Tensor:
        """x: (B, T, H, W, C). Returns (B, T, H, W, C)."""
        B, T, H, W, C = x.shape
        h_heads, w_heads, D = self.h_heads, self.w_heads, self.head_dim
        strip_batch = int(getattr(self, 'inference_strip_batch', 0) or 0)

        if (strip_batch > 0 and not self.training and self.qkv_dw is None
                and max(B * H, B * W) > strip_batch):
            cos_h, sin_h = self.rope.get_cos_sin(
                T, H, 1, device=x.device, t_stride=t_stride)
            cos_w, sin_w = self.rope.get_cos_sin(
                T, 1, W, device=x.device, t_stride=t_stride)

            x_h = x.permute(0, 3, 1, 2, 4).reshape(B * W, T * H, C)
            out_h = x.new_empty((B * W, h_heads, T * H, D))
            for start in range(0, B * W, strip_batch):
                end = min(start + strip_batch, B * W)
                q, k, v = self.qkv(x_h[start:end])
                q = q[:, :h_heads]
                k = k[:, :h_heads]
                v = v[:, :h_heads]
                if self.qk_norm:
                    q = self.q_norm(q)
                    k = self.k_norm(k)
                q = apply_rotary(q, cos_h, sin_h)
                k = apply_rotary(k, cos_h, sin_h)
                out_h[start:end] = sdpa(
                    q, k, v, dropout_p=0.0, is_causal=False)
            attn_h = (out_h.reshape(B, W, h_heads, T, H, D)
                           .permute(0, 3, 4, 1, 2, 5)
                           .contiguous())

            x_w = x.permute(0, 2, 1, 3, 4).reshape(B * H, T * W, C)
            out_w = x.new_empty((B * H, w_heads, T * W, D))
            for start in range(0, B * H, strip_batch):
                end = min(start + strip_batch, B * H)
                q, k, v = self.qkv(x_w[start:end])
                q = q[:, h_heads:]
                k = k[:, h_heads:]
                v = v[:, h_heads:]
                if self.qk_norm:
                    q = self.q_norm(q)
                    k = self.k_norm(k)
                q = apply_rotary(q, cos_w, sin_w)
                k = apply_rotary(k, cos_w, sin_w)
                out_w[start:end] = sdpa(
                    q, k, v, dropout_p=0.0, is_causal=False)
            attn_w = (out_w.reshape(B, H, w_heads, T, W, D)
                           .permute(0, 3, 1, 4, 2, 5)
                           .contiguous())

            attn = torch.cat([attn_h, attn_w], dim=-2)
            attn = attn * self.attn_gate.grid(x, dtype=attn.dtype)
            attn = attn.reshape(B, T, H, W, C)
            return self.proj(attn)

        if self.qkv_dw is not None:
            y = self.qkv_dw(self.qkv.qkv(x))
            qkv = (y.reshape(B, T * H * W, 3, self.num_heads, D)
                    .permute(2, 0, 3, 1, 4)
                    .contiguous())
            q, k, v = qkv.unbind(dim=0)
        else:
            x_flat = x.reshape(B, T * H * W, C)
            q, k, v = self.qkv(x_flat)                        # (B, num_heads, T*H*W, D)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q_h, k_h, v_h = q[:, :h_heads], k[:, :h_heads], v[:, :h_heads]
        q_w, k_w, v_w = q[:, h_heads:], k[:, h_heads:], v[:, h_heads:]

        # H-group (V-stripes, T·H tokens per column): (B, h_heads, T*H*W, D)
        # → (B, h_heads, T, H, W, D) → (B, W, h_heads, T, H, D) → (B·W, h_heads, T·H, D)
        def _to_hstrip(t):
            return (t.reshape(B, h_heads, T, H, W, D)
                     .permute(0, 4, 1, 2, 3, 5)
                     .reshape(B * W, h_heads, T * H, D))
        q_h, k_h, v_h = _to_hstrip(q_h), _to_hstrip(k_h), _to_hstrip(v_h)

        # W-group (H-stripes, T·W tokens per row): (B, w_heads, T*H*W, D)
        # → (B, w_heads, T, H, W, D) → (B, H, w_heads, T, W, D) → (B·H, w_heads, T·W, D)
        def _to_wstrip(t):
            return (t.reshape(B, w_heads, T, H, W, D)
                     .permute(0, 3, 1, 2, 4, 5)
                     .reshape(B * H, w_heads, T * W, D))
        q_w, k_w, v_w = _to_wstrip(q_w), _to_wstrip(k_w), _to_wstrip(v_w)

        cos_h, sin_h = self.rope.get_cos_sin(T, H, 1,
                                              device=x.device, t_stride=t_stride)
        cos_w, sin_w = self.rope.get_cos_sin(T, 1, W,
                                              device=x.device, t_stride=t_stride)
        q_h = apply_rotary(q_h, cos_h, sin_h)
        k_h = apply_rotary(k_h, cos_h, sin_h)
        q_w = apply_rotary(q_w, cos_w, sin_w)
        k_w = apply_rotary(k_w, cos_w, sin_w)

        attn_h = sdpa(q_h, k_h, v_h,
                      dropout_p=0.0, is_causal=False)
        attn_w = sdpa(q_w, k_w, v_w,
                      dropout_p=0.0, is_causal=False)

        # Un-strip: H-group → (B, T, H, W, h_heads, D)
        attn_h = (attn_h.reshape(B, W, h_heads, T, H, D)
                        .permute(0, 3, 4, 1, 2, 5)
                        .contiguous())
        # W-group → (B, T, H, W, w_heads, D)
        attn_w = (attn_w.reshape(B, H, w_heads, T, W, D)
                        .permute(0, 3, 1, 4, 2, 5)
                        .contiguous())

        attn = torch.cat([attn_h, attn_w], dim=-2)
        attn = attn * self.attn_gate.grid(x, dtype=attn.dtype)
        attn = attn.reshape(B, T, H, W, C)
        return self.proj(attn)


class FullAttention3D(nn.Module):
    """Global 3D attention (no windowing) with 3D RoPE.

    Used at /64 where the token count (~32² × T) is small enough for
    quadratic attention to be cheap.
    """

    def __init__(self, dim: int, num_heads: int,
                 rope: Optional[RoPE3D] = None,
                 qkv_bias: bool = True,
                 qk_norm: bool = True,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = _QKVProj(dim, num_heads, qkv_bias=qkv_bias)
        self.qkv_dw = (
            QKVDepthwiseSpatial(dim, kernel_size=int(qkv_dw_kernel))
            if qkv_dw_kernel and qkv_dw_kernel > 0 else None
        )
        self.attn_gate = HeadwiseAttentionGate(dim, num_heads)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        # Zero-init output proj so residual branch contributes 0 at init.
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.rope = rope or RoPE3D(self.head_dim)

    def forward(self, x: torch.Tensor, t_stride: float = 1.0) -> torch.Tensor:
        """x: (B, T, H, W, C). Returns (B, T, H, W, C)."""
        B, T, H, W, C = x.shape
        N = T * H * W
        query_chunk = int(getattr(self, 'inference_query_chunk_tokens', 0) or 0)
        if self.qkv_dw is not None:
            y = self.qkv_dw(self.qkv.qkv(x))
            qkv = (y.reshape(B, N, 3, self.num_heads, self.head_dim)
                    .permute(2, 0, 3, 1, 4)
                    .contiguous())
            q, k, v = qkv.unbind(dim=0)
        else:
            seq = x.reshape(B, N, C)
            q, k, v = self.qkv(seq)
        gate_tokens = x.reshape(B, N, C)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope.get_cos_sin(T, H, W, device=x.device,
                                         t_stride=t_stride)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if query_chunk > 0 and not self.training and N > query_chunk:
            attn = q.new_empty((B, self.num_heads, N, self.head_dim))
            for start in range(0, N, query_chunk):
                end = min(start + query_chunk, N)
                a = sdpa(
                    q[:, :, start:end], k, v,
                    dropout_p=0.0, is_causal=False)
                a = self.attn_gate(gate_tokens[:, start:end], a)
                attn[:, :, start:end] = a
        else:
            attn = sdpa(
                q, k, v, dropout_p=0.0, is_causal=False)
            attn = self.attn_gate(gate_tokens, attn)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        attn = self.proj(attn)
        return attn.reshape(B, T, H, W, C)


class FullResolutionRecurrentMemorySite(nn.Module):
    """Full-/64 recurrent memory site.

    The persistent state has the same H×W grid and channel width as the /64
    stream. Each frame first reads from the accumulated state via full
    cross-attention, then writes the current representation back with an RVM-
    style gated transformer candidate. Unlike RVM, the candidate updates only
    the recurrent memory; the frame stream remains the current-frame features
    plus the explicit read residual. The read output projection is zero-
    initialized, so adding the site to an existing checkpoint is behavior-
    neutral.
    """

    def __init__(self, dim: int, num_heads: int,
                 rope: Optional[RoPE2D] = None,
                 qkv_bias: bool = True,
                 qk_norm: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must divide num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads

        self.norm_x = RMSNorm(dim)
        self.norm_state = RMSNorm(dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_gate = HeadwiseAttentionGate(dim, num_heads)
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.rope = rope or RoPE2D(self.head_dim)

        self.write_norm_state = RMSNorm(dim)
        self.input_update = nn.Linear(dim, dim, bias=False)
        self.state_update = nn.Linear(dim, dim, bias=False)
        self.input_reset = nn.Linear(dim, dim, bias=False)
        self.state_reset = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.input_update.weight)
        nn.init.zeros_(self.state_update.weight)
        nn.init.zeros_(self.input_reset.weight)
        nn.init.zeros_(self.state_reset.weight)

        self.write_ca_norm = RMSNorm(dim)
        self.write_ca_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.write_ca_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.write_ca_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.write_ca_gate = HeadwiseAttentionGate(dim, num_heads)
        if self.qk_norm:
            self.write_ca_q_norm = RMSNorm(self.head_dim)
            self.write_ca_k_norm = RMSNorm(self.head_dim)
        self.write_ca_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.write_ca_proj.weight)
        if self.write_ca_proj.bias is not None:
            nn.init.zeros_(self.write_ca_proj.bias)

    def _split_heads(self, tokens: torch.Tensor,
                     proj: nn.Linear) -> torch.Tensor:
        B, N, C = tokens.shape
        return proj(tokens).reshape(
            B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _read(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x_t.shape
        if tuple(state.shape) != (B, H, W, C):
            raise ValueError(
                f"recurrent state shape {tuple(state.shape)} != "
                f"expected {(B, H, W, C)}")
        N = H * W
        x_norm = self.norm_x(x_t)
        s_norm = self.norm_state(state)
        q_tokens = x_norm.reshape(B, N, C)
        k_tokens = s_norm.reshape(B, N, C)
        q = self._split_heads(q_tokens, self.q)
        k = self._split_heads(k_tokens, self.k)
        v = self._split_heads(k_tokens, self.v)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope.get_cos_sin(H, W, device=x_t.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        attn = sdpa(
            q, k, v, dropout_p=0.0, is_causal=False)
        attn = self.attn_gate(q_tokens, attn)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        read = self.proj(attn).reshape(B, H, W, C)
        return x_t + read

    def _write_cross_attention(
        self,
        query_grid: torch.Tensor,
        kv_grid: torch.Tensor,
    ) -> torch.Tensor:
        B, H, W, C = query_grid.shape
        N = H * W
        q_tokens = self.write_ca_norm(query_grid).reshape(B, N, C)
        kv_tokens = kv_grid.reshape(B, N, C)
        q = self._split_heads(q_tokens, self.write_ca_q)
        k = self._split_heads(kv_tokens, self.write_ca_k)
        v = self._split_heads(kv_tokens, self.write_ca_v)
        if self.qk_norm:
            q = self.write_ca_q_norm(q)
            k = self.write_ca_k_norm(k)
        cos, sin = self.rope.get_cos_sin(H, W, device=query_grid.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        attn = sdpa(
            q, k, v, dropout_p=0.0, is_causal=False)
        attn = self.write_ca_gate(q_tokens, attn)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        attn = self.write_ca_proj(attn).reshape(B, H, W, C)
        return attn

    def _write_candidate(
        self,
        x_t: torch.Tensor,
        gated_state: torch.Tensor,
    ) -> torch.Tensor:
        return x_t + self._write_cross_attention(x_t, gated_state)

    def _write(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        update = torch.sigmoid(self.input_update(x_t) +
                               self.state_update(state))
        reset = torch.sigmoid(self.input_reset(x_t) +
                              self.state_reset(state))
        gated_state = reset * self.write_norm_state(state)
        candidate = self._write_candidate(x_t, gated_state)
        return state + update * (candidate - state)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, H, W, C); state: (B, H, W, C) or None."""
        B, T, H, W, C = x.shape
        if C != self.dim:
            raise ValueError(f"input channels {C} != recurrent dim {self.dim}")
        if state is None:
            state = x[:, 0].clone()
        elif tuple(state.shape) != (B, H, W, C):
            raise ValueError(
                f"recurrent state shape {tuple(state.shape)} != "
                f"expected {(B, H, W, C)}")
        outs = []
        for t in range(T):
            x_t = self._read(x[:, t], state)
            state = self._write(x_t, state)
            outs.append(x_t)
        return torch.stack(outs, dim=1), state


class FullSpatialAttention2D(nn.Module):
    """Per-frame full-spatial attention for NTHWC video tensors.

    Attention is global over H×W independently for each frame; there is no
    temporal mixing here. This is used before the /16 → /64 downsample so the
    high-resolution local/hint features can exchange spatial information while
    leaving temporal/global video reasoning to the /64 body.
    """

    def __init__(self, dim: int, num_heads: int,
                 rope: Optional[RoPE2D] = None,
                 qkv_bias: bool = True,
                 qk_norm: bool = True,
                 zero_init_proj: bool = True,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = _QKVProj(dim, num_heads, qkv_bias=qkv_bias)
        self.qkv_dw = (
            QKVDepthwiseSpatial(dim, kernel_size=int(qkv_dw_kernel))
            if qkv_dw_kernel and qkv_dw_kernel > 0 else None
        )
        self.attn_gate = HeadwiseAttentionGate(dim, num_heads)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        if zero_init_proj:
            nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
        self.rope = rope or RoPE2D(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, C). Returns same shape."""
        B, T, H, W, C = x.shape
        N = H * W
        BT = B * T
        query_chunk = int(getattr(self, 'inference_query_chunk_tokens', 0) or 0)
        if self.qkv_dw is not None:
            y = self.qkv_dw(self.qkv.qkv(x))
            qkv = (y.reshape(BT, N, 3, self.num_heads, self.head_dim)
                    .permute(2, 0, 3, 1, 4)
                    .contiguous())
            q, k, v = qkv.unbind(dim=0)
        else:
            tokens = x.reshape(BT, N, C)
            q, k, v = self.qkv(tokens)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope.get_cos_sin(H, W, device=x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        gate_tokens = x.reshape(BT, N, C)
        if query_chunk > 0 and not self.training and N > query_chunk:
            attn = q.new_empty((BT, self.num_heads, N, self.head_dim))
            for start in range(0, N, query_chunk):
                end = min(start + query_chunk, N)
                a = sdpa(
                    q[:, :, start:end], k, v,
                    dropout_p=0.0, is_causal=False)
                a = self.attn_gate(gate_tokens[:, start:end], a)
                attn[:, :, start:end] = a
        else:
            attn = sdpa(
                q, k, v, dropout_p=0.0, is_causal=False)
            attn = self.attn_gate(gate_tokens, attn)
        attn = attn.transpose(1, 2).reshape(BT, N, C)
        attn = self.proj(attn)
        return attn.reshape(B, T, H, W, C)


class FullSpatialBlock2D(nn.Module):
    """Pre-norm block with full H×W attention for BHWC tensors.

    Used sparingly in stage 2 to provide global spatial exchange while the
    surrounding Swin/CSwin blocks keep the cheaper local/axial inductive bias.
    """

    def __init__(self, dim: int, num_heads: int,
                 mlp_mult: float = 2.5,
                 rope: Optional[RoPE2D] = None,
                 local_mix_kernel: int = 0,
                 local_mix_init_std: float | None = None,
                 mlp_dw_kernel: int = 0,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        self.dim = dim
        self.norm1 = RMSNorm(dim)
        self.attn = FullSpatialAttention2D(
            dim, num_heads, rope=rope,
            qkv_dw_kernel=qkv_dw_kernel,
        )
        self.local_mix = (
            DepthwiseSwiGLU2D(dim, kernel_size=local_mix_kernel,
                              init_std=local_mix_init_std)
            if local_mix_kernel and local_mix_kernel > 0 else None
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mult=mlp_mult, dw_kernel=mlp_dw_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C). Returns same shape."""
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x[:, None]).squeeze(1)
        x = shortcut + x

        if self.local_mix is not None:
            x = x + self.local_mix(x)

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x


# ---------------------------------------------------------------------------
# LayerScale — grows residual contribution from ~0 at init
# ---------------------------------------------------------------------------


class LayerScale(nn.Module):
    """Per-channel scalar gamma multiplier, init=1e-6 (near-identity residual)."""

    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# ---------------------------------------------------------------------------
# Block wrappers (Swin-2D, Swin-3D, Full-3D)
# ---------------------------------------------------------------------------


def _cyclic_shift_2d(x: torch.Tensor, shift: Tuple[int, int]) -> torch.Tensor:
    """Roll a 2D spatial grid by (h, w). Caller provides the sign."""
    sh, sw = shift
    if sh == 0 and sw == 0:
        return x
    # x is (B, H, W, C)
    return torch.roll(x, shifts=(sh, sw), dims=(1, 2))


def _cyclic_shift_3d(x: torch.Tensor, shift: Tuple[int, int, int]) -> torch.Tensor:
    """Roll a 3D spatiotemporal grid by (t, h, w)."""
    st, sh, sw = shift
    if st == 0 and sh == 0 and sw == 0:
        return x
    # x is (B, T, H, W, C)
    return torch.roll(x, shifts=(st, sh, sw), dims=(1, 2, 3))


def _resolve_shift_2d(shift, window_size: int) -> Tuple[int, int]:
    """Accept shift as bool (legacy) or (sh, sw) tuple. Bool True → (w/2, w/2)."""
    if shift is True:
        return (window_size // 2, window_size // 2)
    if shift is False or shift is None:
        return (0, 0)
    assert isinstance(shift, (tuple, list)) and len(shift) == 2, (
        f"shift must be bool or (sh, sw); got {shift!r}")
    return (int(shift[0]), int(shift[1]))


def _resolve_shift_3d(shift, window_H: int, window_W: int) -> Tuple[int, int, int]:
    """Accept shift as bool (legacy) or (sh, sw) tuple for SPATIAL shift.
    Temporal shift is never used (T_w = T). Returns (st=0, sh, sw)."""
    if shift is True:
        return (0, window_H // 2, window_W // 2)
    if shift is False or shift is None:
        return (0, 0, 0)
    assert isinstance(shift, (tuple, list)) and len(shift) == 2, (
        f"shift must be bool or (sh, sw); got {shift!r}")
    return (0, int(shift[0]), int(shift[1]))


class SwinBlock2D(nn.Module):
    """Pre-norm Swin-2D block: RMSNorm → (shift?) → WindowAttn2D → residual
    → optional DW local mix → RMSNorm → SwiGLU → residual.

    `shift` accepts bool (legacy: True → (w/2, w/2)) OR a `(sh, sw)` tuple
    for the 4-phase pattern: (0, 0), (w/2, w/2), (w/2, 0), (0, w/2).

    Near-identity at init via zero-init output projections (attn.proj +
    SwiGLU.w_down). No LayerScale.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift=False, mlp_mult: float = 2.5,
                 rope: Optional[RoPE2D] = None,
                 local_mix_kernel: int = 0,
                 local_mix_init_std: float | None = None,
                 mlp_dw_kernel: int = 0,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_h, self.shift_w = _resolve_shift_2d(shift, window_size)
        self.shift = (self.shift_h, self.shift_w)

        self.norm1 = RMSNorm(dim)
        self.attn = SwinAttention2D(
            dim, num_heads, window_size, rope=rope,
            qkv_dw_kernel=qkv_dw_kernel,
        )

        self.local_mix = (
            DepthwiseSwiGLU2D(dim, kernel_size=local_mix_kernel,
                               init_std=local_mix_init_std)
            if local_mix_kernel and local_mix_kernel > 0 else None
        )

        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mult=mlp_mult, dw_kernel=mlp_dw_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C). Returns same shape."""
        sh, sw = self.shift_h, self.shift_w
        shortcut = x
        x = self.norm1(x)
        if sh or sw:
            H, W = x.shape[1], x.shape[2]
            w = self.window_size
            Hp = math.ceil(H / w) * w
            Wp = math.ceil(W / w) * w
            if Hp != H or Wp != W:
                x = F.pad(x, (0, 0, 0, Wp - W, 0, Hp - H))
            region_groups = _shifted_region_groups_2d_cached(
                H, W, Hp, Wp, w, sh, sw)
            x = _cyclic_shift_2d(x, (-sh, -sw))
            x = self.attn(x, region_groups=region_groups)
            x = _cyclic_shift_2d(x, (sh, sw))
            x = x[:, :H, :W, :].contiguous()
        else:
            x = self.attn(x)
        x = shortcut + x

        if self.local_mix is not None:
            x = x + self.local_mix(x)

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x


class AxialBlock2D(nn.Module):
    """Pre-norm axial 2D block: RMSNorm → AxialAttention2D(axis) → residual
    → RMSNorm → SwiGLU → residual. No padding, no windowing."""

    def __init__(self, dim: int, num_heads: int, axis: str,
                 mlp_mult: float = 2.5,
                 rope: Optional[RoPE2D] = None):
        super().__init__()
        self.dim = dim
        self.axis = axis
        self.norm1 = RMSNorm(dim)
        self.attn = AxialAttention2D(dim, num_heads, axis=axis, rope=rope)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mult=mlp_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x


class CSwinBlock2D(nn.Module):
    """Pre-norm CSwin 2D block: RMSNorm → CSwinAttention2D (half heads on
    H-axis, half on W-axis) → residual → optional DW local mix → RMSNorm
    → SwiGLU → residual.

    Each block yields a globally-mixed output on both axes in a single
    attention call.
    """

    def __init__(self, dim: int, num_heads: int,
                 mlp_mult: float = 2.5,
                 rope: Optional[RoPE2D] = None,
                 local_mix_kernel: int = 0,
                 local_mix_init_std: float | None = None,
                 mlp_dw_kernel: int = 0,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        self.dim = dim
        self.norm1 = RMSNorm(dim)
        self.attn = CSwinAttention2D(
            dim, num_heads, rope=rope,
            qkv_dw_kernel=qkv_dw_kernel,
        )
        self.local_mix = (
            DepthwiseSwiGLU2D(dim, kernel_size=local_mix_kernel,
                               init_std=local_mix_init_std)
            if local_mix_kernel and local_mix_kernel > 0 else None
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mult=mlp_mult, dw_kernel=mlp_dw_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        if self.local_mix is not None:
            x = x + self.local_mix(x)
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x


class FullSpatialBlock3D(nn.Module):
    """Pre-norm per-frame full-spatial block for stage-1 pre-downsample use.

    RMSNorm → full H×W attention per frame → residual → optional per-frame DW
    local mix → RMSNorm → SwiGLU → residual. Temporal mixing is deliberately
    absent; the /64 body handles full spatiotemporal reasoning after downsample.
    """

    def __init__(self, dim: int, num_heads: int,
                 mlp_mult: float = 2.5,
                 rope: Optional[RoPE2D] = None,
                 local_mix_kernel: int = 0,
                 local_mix_init_std: float | None = None,
                 adaln_cond_dim: int = 0,
                 mlp_dw_kernel: int = 0,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        self.dim = dim
        self.use_adaln_zero = bool(adaln_cond_dim and adaln_cond_dim > 0)
        self.norm1 = RMSNorm(dim)
        self.attn = FullSpatialAttention2D(
            dim, num_heads, rope=rope,
            zero_init_proj=not self.use_adaln_zero,
            qkv_dw_kernel=qkv_dw_kernel,
        )
        self.local_mix = (
            DepthwiseSwiGLU2D(dim, kernel_size=local_mix_kernel,
                              init_std=local_mix_init_std)
            if local_mix_kernel and local_mix_kernel > 0 else None
        )
        self.adaln = (
            AdaLNZero(dim, cond_dim=adaln_cond_dim, pairs=2)
            if self.use_adaln_zero else None
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mult=mlp_mult,
                          zero_init_down=not self.use_adaln_zero,
                          dw_kernel=mlp_dw_kernel)

    def _local_mix_frames(self, x: torch.Tensor) -> torch.Tensor:
        if self.local_mix is None:
            return x
        B, T, H, W, C = x.shape
        y = x.reshape(B * T, H, W, C)
        y = self.local_mix(y)
        return x + y.reshape(B, T, H, W, C)

    def forward(
        self,
        x: torch.Tensor,
        t_stride: float = 1.0,
        adaln_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del t_stride
        if self.adaln is not None and adaln_cond is None:
            adaln_cond = x.new_zeros(x.shape[0], x.shape[1])
        if self.adaln is not None:
            shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(adaln_cond, x)
        else:
            shift1 = scale1 = gate1 = shift2 = scale2 = gate2 = None

        shortcut = x
        x = self.norm1(x)
        if shift1 is not None:
            x = x * (1.0 + scale1) + shift1
        x = self.attn(x)
        x = shortcut + (x * gate1 if gate1 is not None else x)

        x = self._local_mix_frames(x)

        shortcut = x
        x = self.norm2(x)
        if shift2 is not None:
            x = x * (1.0 + scale2) + shift2
        x = self.mlp(x)
        x = shortcut + (x * gate2 if gate2 is not None else x)
        return x


class SwinBlock3D(nn.Module):
    """Pre-norm Swin-3D block. Full-temporal + windowed-spatial attention
    + optional per-frame DW local mix + SwiGLU. `shift` accepts bool
    (legacy: True → (w/2, w/2)) OR a `(sh, sw)` tuple for 4-phase patterns:
    (0,0), (w/2,w/2), (w/2,0), (0,w/2). Temporal shift is always 0
    (T_w = T).

    Near-identity at init via zero-init down projections (no LayerScale).
    """

    def __init__(self, dim: int, num_heads: int,
                 window_H: int = 8, window_W: int = 8,
                 shift=False, mlp_mult: float = 2.5,
                 rope: Optional[RoPE3D] = None,
                 local_mix_kernel: int = 0,
                 local_mix_init_std: float | None = None,
                 adaln_cond_dim: int = 0,
                 mlp_dw_kernel: int = 0,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        self.dim = dim
        self.window_H = window_H
        self.window_W = window_W
        _, self.shift_h, self.shift_w = _resolve_shift_3d(shift, window_H, window_W)
        self.shift = (self.shift_h, self.shift_w)
        self.use_adaln_zero = bool(adaln_cond_dim and adaln_cond_dim > 0)

        self.norm1 = RMSNorm(dim)
        self.attn = SwinAttention3D(
            dim, num_heads, window_H, window_W, rope=rope,
            zero_init_proj=not self.use_adaln_zero,
            qkv_dw_kernel=qkv_dw_kernel,
        )

        self.local_mix = (
            DepthwiseSwiGLU2D(dim, kernel_size=local_mix_kernel,
                              init_std=local_mix_init_std)
            if local_mix_kernel and local_mix_kernel > 0 else None
        )
        self.adaln = (
            AdaLNZero(dim, cond_dim=adaln_cond_dim, pairs=2)
            if self.use_adaln_zero else None
        )

        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mult=mlp_mult,
                          zero_init_down=not self.use_adaln_zero,
                          dw_kernel=mlp_dw_kernel)

    def _local_mix_frames(self, x: torch.Tensor) -> torch.Tensor:
        if self.local_mix is None:
            return x
        B, T, H, W, C = x.shape
        y = x.reshape(B * T, H, W, C)
        y = self.local_mix(y)
        return x + y.reshape(B, T, H, W, C)

    def forward(
        self,
        x: torch.Tensor,
        t_stride: float = 1.0,
        adaln_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B, T, H, W, C). Returns (B, T, H, W, C)."""
        if self.adaln is not None and adaln_cond is None:
            adaln_cond = x.new_zeros(x.shape[0], x.shape[1])
        if self.adaln is not None:
            shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(adaln_cond, x)
        else:
            shift1 = scale1 = gate1 = shift2 = scale2 = gate2 = None
        sh, sw = self.shift_h, self.shift_w
        shortcut = x
        x = self.norm1(x)
        if shift1 is not None:
            x = x * (1.0 + scale1) + shift1
        if sh or sw:
            T, H, W = x.shape[1], x.shape[2], x.shape[3]
            H_w, W_w = self.window_H, self.window_W
            Hp = math.ceil(H / H_w) * H_w
            Wp = math.ceil(W / W_w) * W_w
            if Hp != H or Wp != W:
                x = F.pad(x, (0, 0, 0, Wp - W, 0, Hp - H))
            region_groups = _shifted_region_groups_3d_cached(
                T, H, W, Hp, Wp, H_w, W_w, sh, sw)
            x = _cyclic_shift_3d(x, (0, -sh, -sw))
            x = self.attn(x, t_stride=t_stride, region_groups=region_groups)
            x = _cyclic_shift_3d(x, (0, sh, sw))
            x = x[:, :, :H, :W, :].contiguous()
        else:
            x = self.attn(x, t_stride=t_stride)
        x = shortcut + (x * gate1 if gate1 is not None else x)

        x = self._local_mix_frames(x)

        shortcut = x
        x = self.norm2(x)
        if shift2 is not None:
            x = x * (1.0 + scale2) + shift2
        x = self.mlp(x)
        x = shortcut + (x * gate2 if gate2 is not None else x)
        return x


class AxialBlock3D(nn.Module):
    """Pre-norm axial 3D block: RMSNorm → AxialAttention3D(axis) → residual
    → RMSNorm → SwiGLU → residual. T-full + one spatial axis."""

    def __init__(self, dim: int, num_heads: int, axis: str,
                 mlp_mult: float = 2.5,
                 rope: Optional[RoPE3D] = None):
        super().__init__()
        self.dim = dim
        self.axis = axis
        self.norm1 = RMSNorm(dim)
        self.attn = AxialAttention3D(dim, num_heads, axis=axis, rope=rope)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mult=mlp_mult)

    def forward(self, x: torch.Tensor, t_stride: float = 1.0) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, t_stride=t_stride)
        x = shortcut + x
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x


class CSwinBlock3D(nn.Module):
    """Pre-norm CSwin 3D block: RMSNorm → CSwinAttention3D (half heads on
    H-axis, half on W-axis; T always full) → residual → optional per-frame
    DW local mix → RMSNorm → SwiGLU → residual.
    """

    def __init__(self, dim: int, num_heads: int,
                 mlp_mult: float = 2.5,
                 rope: Optional[RoPE3D] = None,
                 local_mix_kernel: int = 0,
                 local_mix_init_std: float | None = None,
                 mlp_dw_kernel: int = 0,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        self.dim = dim
        self.norm1 = RMSNorm(dim)
        self.attn = CSwinAttention3D(
            dim, num_heads, rope=rope,
            qkv_dw_kernel=qkv_dw_kernel,
        )
        self.local_mix = (
            DepthwiseSwiGLU2D(dim, kernel_size=local_mix_kernel,
                              init_std=local_mix_init_std)
            if local_mix_kernel and local_mix_kernel > 0 else None
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mult=mlp_mult, dw_kernel=mlp_dw_kernel)

    def _local_mix_frames(self, x: torch.Tensor) -> torch.Tensor:
        if self.local_mix is None:
            return x
        B, T, H, W, C = x.shape
        y = x.reshape(B * T, H, W, C)
        y = self.local_mix(y)
        return x + y.reshape(B, T, H, W, C)

    def forward(self, x: torch.Tensor, t_stride: float = 1.0) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, t_stride=t_stride)
        x = shortcut + x
        x = self._local_mix_frames(x)
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x


class FullSpatioTemporalBlock(nn.Module):
    """Pre-norm block with full (unwindowed) 3D attention. For /64 body.
    Near-identity at init via zero-init down projections (no LayerScale).
    """

    def __init__(self, dim: int, num_heads: int,
                 mlp_mult: float = 2.5,
                 rope: Optional[RoPE3D] = None,
                 local_mix_kernel: int = 0,
                 local_mix_init_std: float | None = None,
                 mlp_dw_kernel: int = 0,
                 mlp_hidden: int | None = None,
                 qkv_dw_kernel: int = 0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = FullAttention3D(
            dim, num_heads, rope=rope,
            qkv_dw_kernel=qkv_dw_kernel,
        )
        self.local_mix = (
            DepthwiseSwiGLU2D(dim, kernel_size=local_mix_kernel,
                              init_std=local_mix_init_std)
            if local_mix_kernel and local_mix_kernel > 0 else None
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mult=mlp_mult, dw_kernel=mlp_dw_kernel,
                          hidden=mlp_hidden)

    def _local_mix_frames(self, x: torch.Tensor) -> torch.Tensor:
        if self.local_mix is None:
            return x
        B, T, H, W, C = x.shape
        y = x.reshape(B * T, H, W, C)
        y = self.local_mix(y)
        return x + y.reshape(B, T, H, W, C)

    def forward(
        self,
        x: torch.Tensor,
        t_stride: float = 1.0,
        recurrent_site: FullResolutionRecurrentMemorySite | None = None,
        recurrent_state: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, H, W, C). Returns (B, T, H, W, C)."""
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, t_stride=t_stride)
        x = shortcut + x
        x = self._local_mix_frames(x)
        if recurrent_site is not None:
            x, recurrent_state = recurrent_site(x, recurrent_state)

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        if recurrent_site is not None:
            return x, recurrent_state
        return x


# ---------------------------------------------------------------------------
# Patchify / downsample / upsample / depatchify
# ---------------------------------------------------------------------------


class PatchifyConv(nn.Module):
    """Conv2d(kernel=r, stride=r) that turns (B, C_in, H, W) into
    (B, C_out, H/r, W/r). Supports partial zero-init on output channels via
    `zero_init_first` — the first N output channels have their weights set
    to zero while the rest get Kaiming.

    `overlap_mult` controls the receptive field size:
      • 1 (default): kernel=patch, stride=patch — no overlap (pure ViT patch).
      • 2: kernel=2·patch, stride=patch, padding=patch/2 — each token's RF
        covers its own 16×16 + half-width strips of the 4 axis neighbors +
        quarter-size corners of the 4 diagonal neighbors. ("Half-overlap")
      • 3: kernel=3·patch, stride=patch, padding=patch — immediate + 8
        surrounding 16×16 blocks fully covered. ("Full-overlap")

    `overlap=True` is kept as an alias for overlap_mult=3.

    Init behavior for overlap_mult > 1: the CENTRAL patch×patch region of
    each kernel is Kaiming-inited (producing the "own block" features
    identically to the non-overlap baseline at step 0); the surrounding
    spill regions that reach into neighboring 16×16 blocks are
    ZERO-inited. This way at init a token is a pure function of its own
    block (same as non-overlap), and the overlap contribution from
    neighbors emerges only as those spill weights train away from zero.
    """

    def __init__(self, in_channels: int, out_channels: int, patch: int,
                 zero_init_first: int = 0,
                 zero_init_all: bool = False,
                 bias: bool = True,
                 overlap: bool = False,
                 overlap_mult: int = 1):
        super().__init__()
        self.patch = patch
        if overlap and overlap_mult == 1:
            overlap_mult = 3  # legacy alias
        assert overlap_mult >= 1, f"overlap_mult must be >= 1, got {overlap_mult}"
        self.overlap_mult = overlap_mult
        kernel = overlap_mult * patch
        padding = (kernel - patch) // 2
        assert (kernel - patch) % 2 == 0, (
            f"overlap_mult·patch - patch must be even; got kernel={kernel}, patch={patch}")
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel, stride=patch,
                              padding=padding, bias=bias)
        if zero_init_all:
            nn.init.zeros_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)
        else:
            # Default nn.Conv2d __init__ already Kaiming-inited the full
            # kernel + uniform-inited the bias. For overlap_mult > 1 we
            # want the Kaiming signal only in the central patch×patch
            # region; zero the spill. For `zero_init_first` we additionally
            # zero the first N output channels entirely (kernel *and* bias
            # for those channels).
            with torch.no_grad():
                if overlap_mult > 1:
                    # Zero the spill: build a (kernel, kernel) mask with a
                    # patch×patch block of 1s in the center.
                    c_lo = padding
                    c_hi = padding + patch
                    mask = torch.zeros(kernel, kernel,
                                       dtype=self.conv.weight.dtype,
                                       device=self.conv.weight.device)
                    mask[c_lo:c_hi, c_lo:c_hi] = 1.0
                    self.conv.weight.data.mul_(mask)
                if zero_init_first > 0:
                    self.conv.weight.data[:zero_init_first].zero_()
                    if self.conv.bias is not None:
                        self.conv.bias.data[:zero_init_first].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Downsample4(nn.Module):
    """Conv(k=8, s=4) downsample from /16 → /64. Kernel>stride gives overlapping
    receptive fields (an implicit low-pass before sub-sample), an improvement
    over the minimum-cost k=4 s=4 form. Output layer-norm by default."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel: int = 8, stride: int = 4, norm: bool = True):
        super().__init__()
        assert kernel >= stride, "kernel smaller than stride would drop pixels"
        # Padding chosen so the output size is exactly in//stride (for in % stride == 0).
        # pad = (kernel - stride) // 2. At k=8 s=4 → pad=2 (centered).
        pad = (kernel - stride) // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel, stride=stride, padding=pad)
        self.norm = LayerNorm2d(out_channels) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class Upsample4ICNR(nn.Module):
    """ICNR-inited /64 → /16 upsample: 1×1 projection to c_out·r² channels,
    PixelShuffle(r=4), optional post Conv(k=3) smoothing.

    Post-conv is symmetric in spirit with the kernel-8 overlap used on
    the downsample side — it mixes across the 4×4 pixel-shuffle output
    at cost `c_out² · 9`. Disable via `post_smooth=False` if the next
    transformer blocks are considered sufficient to smooth.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 r: int = 4, post_smooth: bool = True, norm: bool = True):
        super().__init__()
        self.r = r
        self.proj = nn.Conv2d(in_channels, out_channels * r * r, kernel_size=1)
        icnr_init_(self.proj, upscale_factor=r)
        self.post_smooth = post_smooth
        if post_smooth:
            self.smooth = nn.Conv2d(out_channels, out_channels,
                                    kernel_size=3, padding=1)
            # Zero-init so the upsample path is pure nearest-neighbor at t=0.
            # Combined with a residual around smooth in forward, this
            # preserves the ICNR pixel-shuffle output unchanged at init and
            # lets the 3×3 smoother grow in as a learned correction.
            nn.init.zeros_(self.smooth.weight)
            if self.smooth.bias is not None:
                nn.init.zeros_(self.smooth.bias)
        else:
            self.smooth = nn.Identity()
        self.norm = LayerNorm2d(out_channels) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.pixel_shuffle(x, self.r)
        if self.post_smooth:
            # Residual around the zero-inited smoother: at init this is the
            # identity; during training the 3×3 learns a correction.
            x = x + self.smooth(x)
        return self.norm(x)


def _build_depatch_fourier_basis(r: int, num_freqs: int) -> torch.Tensor:
    coords = (torch.arange(r, dtype=torch.float32) + 0.5) / float(r) - 0.5
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    dirs = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0))
    freq_vecs = []
    f = 1
    while len(freq_vecs) < num_freqs:
        for dx, dy in dirs:
            freq_vecs.append((f * dx, f * dy))
            if len(freq_vecs) >= num_freqs:
                break
        f += 1

    terms = []
    for fx, fy in freq_vecs:
        phase = 2.0 * math.pi * (fx * xx + fy * yy)
        terms.append(torch.sin(phase))
        terms.append(torch.cos(phase))
    basis = torch.stack(terms, dim=0)
    rms = basis.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
    basis = basis / rms
    return basis / math.sqrt(float(basis.shape[0]))


class FeatureFourierDepatchResidual(nn.Module):
    """Feature-conditioned subpixel Fourier residual for depatch heads."""

    def __init__(self, in_dim: int, out_channels: int, r: int,
                 num_freqs: int = 16, kernel_size: int = 3,
                 bias: bool = True):
        super().__init__()
        num_freqs = int(num_freqs)
        kernel_size = int(kernel_size)
        if num_freqs <= 0:
            raise ValueError("num_freqs must be positive")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError(
                f"kernel_size must be a positive odd integer; got {kernel_size}")
        self.r = int(r)
        self.out_channels = int(out_channels)
        self.num_terms = 2 * num_freqs
        self.pad = kernel_size // 2
        self.coeff = nn.Conv2d(
            in_dim, out_channels * self.num_terms,
            kernel_size=kernel_size, padding=0, bias=bias,
        )
        self.reset_to_zero()
        self.register_buffer(
            'basis',
            _build_depatch_fourier_basis(self.r, num_freqs),
            persistent=False,
        )

    def reset_to_zero(self) -> None:
        nn.init.zeros_(self.coeff.weight)
        if self.coeff.bias is not None:
            nn.init.zeros_(self.coeff.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        if self.pad:
            mode = 'reflect' if H > self.pad and W > self.pad else 'replicate'
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode=mode)
        coeff = self.coeff(x)
        coeff = coeff.view(B, self.out_channels, self.num_terms, H, W)
        basis = self.basis.to(device=coeff.device, dtype=coeff.dtype)
        y = torch.einsum('bonhw,nuv->bohuwv', coeff, basis)
        return y.contiguous().view(B, self.out_channels, H * self.r, W * self.r)


class Depatchify(nn.Module):
    """Depatchify from /r grid to /1.

    Parameterized by `overlap_mult` (receptive-field multiplier):
      • 1 (default): Linear(d → C·r²) + PixelShuffle(r). Each output pixel
        is a function of ONE input token.
      • 2: ConvTranspose2d(d, C, kernel=2·r, stride=r, padding=r/2). Each
        input token paints a 2r×2r patch; output pixel is a sum of 2×2
        neighboring tokens' contributions.
      • 3: ConvTranspose2d(d, C, kernel=3·r, stride=r, padding=r). Each
        output pixel sums contributions from the 3×3 token neighborhood.

    `overlap=True` is kept as an alias for overlap_mult=3.

    Used for:
      • stage-1 aux head: d_baton=512 → 4 channels at /2 (r=8).
      • stage-2 final out: d=1280 → 4 channels at /1 (r=16).

    `zero_init_all` zeros the weight + bias — paired with an identity RGB
    skip at the call site so at t=0 the module outputs 0.
    """

    def __init__(self, in_dim: int, out_channels: int, r: int,
                 zero_init_all: bool = True, bias: bool = True,
                 overlap: bool = False,
                 overlap_mult: int = 1,
                 fourier_features: int = 0,
                 fourier_kernel: int = 3):
        super().__init__()
        self.r = r
        self.out_channels = out_channels
        self.in_dim = in_dim
        if overlap and overlap_mult == 1:
            overlap_mult = 3  # legacy alias
        assert overlap_mult >= 1, f"overlap_mult must be >= 1, got {overlap_mult}"
        self.overlap_mult = overlap_mult

        if overlap_mult == 1:
            self.linear = nn.Linear(in_dim, out_channels * r * r, bias=bias)
            if zero_init_all:
                nn.init.zeros_(self.linear.weight)
                if self.linear.bias is not None:
                    nn.init.zeros_(self.linear.bias)
            self.deconv = None
        else:
            kernel = overlap_mult * r
            padding = (kernel - r) // 2
            assert (kernel - r) % 2 == 0, (
                f"overlap_mult·r - r must be even; kernel={kernel}, r={r}")
            self.deconv = nn.ConvTranspose2d(in_dim, out_channels,
                                             kernel_size=kernel, stride=r,
                                             padding=padding, bias=bias)
            if zero_init_all:
                nn.init.zeros_(self.deconv.weight)
                if self.deconv.bias is not None:
                    nn.init.zeros_(self.deconv.bias)
            else:
                # Central-kaiming / spill-zero: same init pattern as
                # PatchifyConv's overlap case. The r×r central region of
                # each kernel maps to the input token's own output block;
                # the rest spills into neighboring blocks. Zero the spill
                # at init so at step 0 each token only contributes to its
                # own block (matches non-overlap baseline), and the
                # cross-block blending learns in.
                with torch.no_grad():
                    c_lo = padding
                    c_hi = padding + r
                    mask = torch.zeros(kernel, kernel,
                                       dtype=self.deconv.weight.dtype,
                                       device=self.deconv.weight.device)
                    mask[c_lo:c_hi, c_lo:c_hi] = 1.0
                    self.deconv.weight.data.mul_(mask)
            self.linear = None
        self.fourier = (
            FeatureFourierDepatchResidual(
                in_dim, out_channels, r,
                num_freqs=fourier_features,
                kernel_size=fourier_kernel,
                bias=bias,
            )
            if int(fourier_features) > 0 else None
        )
    def forward(self, x_tokens_2d: torch.Tensor) -> torch.Tensor:
        """x_tokens_2d: (B, C=in_dim, H, W). Returns (B, out_channels, H·r, W·r)."""
        if self.overlap_mult > 1:
            out = self.deconv(x_tokens_2d)
        else:
            B, C, H, W = x_tokens_2d.shape
            # (B, C, H, W) → (B, H, W, C) → Linear → (B, H, W, C_out·r²)
            y = x_tokens_2d.permute(0, 2, 3, 1)
            y = self.linear(y)
            y = y.permute(0, 3, 1, 2).contiguous()     # (B, C_out·r², H, W)
            out = F.pixel_shuffle(y, self.r)           # (B, C_out, H·r, W·r)
        if self.fourier is not None:
            out = out + self.fourier(x_tokens_2d)
        return out
