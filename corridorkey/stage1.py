"""Stage 1 — baton producer for CorridorKey v2.

Pure-transformer spatiotemporal encoder/decoder on the /16 ↔ /64 pyramid:

    /1 RGB + hint
      └ Patchify(k=16) → d=512 at /16
      └ 2 × FullSpatialBlock3D (encoder side; per-frame full H×W)
      └ Downsample(k=8, s=4) → d=512 aux at /64
      └ cat frozen foundations + aux → d=1536 at /64
      └ first half of FullSpatioTemporalBlock stack
      └ zero-init upsample to /16 + learned skip from encoder /16
      └ 4 × SwinBlock3D local /16 refinement
      └ zero-init downsample add back to /64
      └ remaining FullSpatioTemporalBlock stack
      └ Upsample4ICNR → d=512 at /16
      └ learned skip from latest /16 refinement
      └ 8 × mixed Swin/CSwinBlock3D (decoder side)
           with zero-init patch-stem skip before the final 2 blocks
    → baton (d=512 at /16)
    → aux head: Depatchify r=8 → 4-channel /2 output
         initialized as half-res RGB passthrough when the 512d projected
         RGB patch basis is available (pred_alpha = 0 init)

No recurrence. Temporal mixing lives entirely in attention: global at /64
(full spatiotemporal), windowed at /16 (Swin-3D with full-T windows and
balanced half-window spatial shift phases).

Windows cleanly divide the /16 grid when `shape_quantum=128` (the
dataset contract for this model), so no intra-window padding is needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    PatchifyConv, Downsample4, Upsample4ICNR, Depatchify,
    FullSpatialBlock3D, SwinBlock3D, CSwinBlock3D,
    FullSpatioTemporalBlock, FullResolutionRecurrentMemorySite,
    FusionLinSwiglu,
    init_orthogonal_rgb_passthrough_,
)
from .pinned import HostBuffer
from .rope import RoPE2D, RoPE3D


def _skip_add_in_place(module, target: torch.Tensor, *others: torch.Tensor) -> bool:
    """Opt-in (`inference_inplace_skips`, inference only): whether a skip
    connection `a + scale * target` may be computed in place on `target`, a
    tensor the caller drops right after. Exact when `target` already has the
    dtype the out-of-place result would have: the same multiply and the same
    (commutative) add, without two more full-size tensors."""
    if not getattr(module, "inference_inplace_skips", False) or module.training:
        return False
    dtype = target.dtype
    for other in others:
        dtype = torch.promote_types(dtype, other.dtype)
    return dtype == target.dtype


@torch.no_grad()
def _init_center_area_downsample_(conv: nn.Conv2d) -> None:
    """Initialize k=8/s=4 downsample as exact per-cell area pooling.

    The central stride×stride region averages the corresponding input cell;
    all spill taps start at zero. When channel counts differ, a semi-orthogonal
    channel projection is used before the same spatial average.
    """
    out_ch, in_ch, kh, kw = conv.weight.shape
    sy, sx = conv.stride
    if kh < sy or kw < sx:
        raise ValueError(
            f"downsample kernel {(kh, kw)} must cover stride {(sy, sx)}")
    y0 = (kh - sy) // 2
    x0 = (kw - sx) // 2
    if (kh - sy) % 2 != 0 or (kw - sx) % 2 != 0:
        raise ValueError(
            f"downsample kernel-stride delta must be even; "
            f"kernel={(kh, kw)} stride={(sy, sx)}")

    conv.weight.zero_()
    if conv.bias is not None:
        conv.bias.zero_()

    if out_ch == in_ch:
        proj = torch.eye(out_ch, device=conv.weight.device, dtype=conv.weight.dtype)
    elif out_ch < in_ch:
        q, r = torch.linalg.qr(
            torch.randn(in_ch, out_ch, device=conv.weight.device,
                        dtype=torch.float32),
            mode='reduced',
        )
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        proj = (q * signs.view(1, -1)).T.to(dtype=conv.weight.dtype)
    else:
        q, r = torch.linalg.qr(
            torch.randn(out_ch, in_ch, device=conv.weight.device,
                        dtype=torch.float32),
            mode='reduced',
        )
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        proj = (q * signs.view(1, -1)).to(dtype=conv.weight.dtype)

    conv.weight[:, :, y0:y0 + sy, x0:x0 + sx].copy_(
        proj[:, :, None, None] / float(sy * sx))


def _frontloaded_qkv_kernels(total: int, kernel: int,
                            fraction: float = 1.0) -> list[int]:
    kernel = int(kernel or 0)
    total = int(total)
    if total <= 0 or kernel <= 0:
        return [0 for _ in range(max(total, 0))]
    frac = max(0.0, min(float(fraction), 1.0))
    active = total if frac >= 1.0 else int(round(total * frac))
    active = max(0, min(total, active))
    return [kernel if i < active else 0 for i in range(total)]


def _stage1_qkv_kernels(n_enc: int, n_body: int, n_mid: int, n_dec: int,
                        kernel: int, fraction: float = 1.0) -> list[int]:
    plan = _frontloaded_qkv_kernels(
        n_enc + n_body + n_mid + n_dec,
        kernel,
        fraction,
    )
    if not plan or int(kernel or 0) <= 0 or float(fraction) >= 1.0:
        return plan

    # The front-loaded plan would spend the first /16 slots on enc_16, which
    # is now full-spatial attention. Move those /16 QKV-DW slots onto the
    # local mid_16 Swin detour instead; keep the /64 body allocation unchanged.
    enc_active = [i for i in range(n_enc) if plan[i] > 0]
    if not enc_active or n_mid <= 0:
        return plan

    for i in enc_active:
        plan[i] = 0

    mid_start = n_enc + n_body
    mid_slots = [
        mid_start + i for i in range(n_mid)
        if plan[mid_start + i] == 0
    ]
    placed = 0
    for i in mid_slots[:len(enc_active)]:
        plan[i] = int(kernel)
        placed += 1

    # Preserve the requested active count if a nonstandard layout has fewer
    # empty mid slots than relocated enc slots.
    for i in enc_active[placed:]:
        plan[i] = int(kernel)
    return plan


def _qkv_kernel_at(kernels, i: int) -> int:
    if isinstance(kernels, (list, tuple)):
        return int(kernels[i]) if i < len(kernels) else 0
    return int(kernels or 0)


def _stage1_dec_blocks(d_16, num_heads_16, window_HW, mlp_mult, rope_16,
                       n_blocks=8, local_mix_kernel=0,
                       local_mix_init_std=None, mlp_dw_kernel=0,
                       qkv_dw_kernel=0):
    """Build the decoder pattern.

    An earlier layout used 6 blocks: 2 dense Swin → 2 CSwin → 2 dense Swin.
    The current layout uses 8 blocks: 2 dense Swin → CSwin → 2 inserted dense Swin
    → CSwin → 2 dense Swin. The CSwin pair still gives two
    full-global sweeps (each block mixes both H and W axes in parallel
    heads), with a small local refinement stack between the two sweeps.

    Dense Swin shifts expose all four half-window phases. The final pair is
    the orthogonal H-only/W-only pair because the late patch-stem skip enters
    immediately before those blocks.
    """
    w = window_HW
    hw = w // 2
    if n_blocks == 6:
        specs = [
            ("swin", (0, 0)),
            ("swin", (hw, hw)),
            ("cswin", None),
            ("cswin", None),
            ("swin", (hw, 0)),
            ("swin", (0, hw)),
        ]
    elif n_blocks == 8:
        specs = [
            ("swin", (0, 0)),
            ("swin", (hw, hw)),
            ("cswin", None),
            ("swin", (hw, 0)),
            ("swin", (0, hw)),
            ("cswin", None),
            ("swin", (hw, 0)),
            ("swin", (0, hw)),
        ]
    else:
        raise ValueError(
            f"n_dec_16 must be 6 or 8 for the fixed CSwin pattern; got {n_blocks}")
    blocks = []
    for i, (kind, shift) in enumerate(specs):
        k = _qkv_kernel_at(qkv_dw_kernel, i)
        if kind == "swin":
            blocks.append(
                SwinBlock3D(d_16, num_heads_16, w, w, shift=shift,
                            mlp_mult=mlp_mult, rope=rope_16,
                            local_mix_kernel=local_mix_kernel,
                            local_mix_init_std=local_mix_init_std,
                            mlp_dw_kernel=mlp_dw_kernel,
                            qkv_dw_kernel=k)
            )
        else:
            blocks.append(
                CSwinBlock3D(d_16, num_heads_16,
                             mlp_mult=mlp_mult, rope=rope_16,
                             local_mix_kernel=local_mix_kernel,
                             local_mix_init_std=local_mix_init_std,
                             mlp_dw_kernel=mlp_dw_kernel,
                             qkv_dw_kernel=k)
            )
    return blocks


def _stage1_mid16_blocks(d_16, num_heads_16, window_HW, mlp_mult, rope_16,
                         n_blocks=2, local_mix_kernel=0,
                         local_mix_init_std=None, mlp_dw_kernel=0,
                         qkv_dw_kernel=0):
    w = window_HW
    hw = w // 2
    phases = [(0, 0), (hw, hw), (hw, 0), (0, hw)]
    return [
        SwinBlock3D(d_16, num_heads_16, w, w,
                    shift=phases[i % len(phases)],
                    mlp_mult=mlp_mult, rope=rope_16,
                    local_mix_kernel=local_mix_kernel,
                    local_mix_init_std=local_mix_init_std,
                    mlp_dw_kernel=mlp_dw_kernel,
                    qkv_dw_kernel=_qkv_kernel_at(qkv_dw_kernel, i))
        for i in range(int(n_blocks))
    ]


class Stage1(nn.Module):
    def __init__(
        self,
        # Channel widths
        d_16: int = 512,
        d_64: int = 1536,
        cradio_dim: int = 0,
        moge_dim: int = 768,
        rvm_dim: int = 0,   # 0 disables RVM feature fusion
        # Depth
        n_enc_16: int = 2,
        n_body_64: int = 10,
        n_dec_16: int = 6,
        # Heads — d_16=512 with 8 heads gives head_dim=64; RoPE3D splits
        # pairs nearly evenly across T/H/W.
        num_heads_16: int = 8,    # 512 / 8 = 64
        num_heads_64: int = 16,   # 1536 / 16 = 96
        # Windowing (spatial only — temporal is always full). Window=16 at
        # /16 means each Swin-3D block attends across 16×16 spatial tokens
        # (256 pixel receptive field per block). The V+H axial pair in the
        # middle of dec_16 gives full global mixing, so the windowed blocks
        # only need local refinement at a smaller window. SwinAttention3D
        # pads internally for /16 grids not divisible by 16 (shape_quantum=128
        # guarantees /8 divisibility at /16 — w=16 may still need padding
        # for oddly-sized crops).
        window_HW: int = 16,
        local_mix_kernel: int = 7,
        local_mix_init_std: float | None = None,
        body64_local_mix_kernel: int = 7,
        body64_local_mix_init_std: float | None = None,
        body64_mlp_hidden: int | None = None,
        body64_direct_concat: bool = False,
        recurrent_64_sites: tuple[int, ...] | None = None,
        hint_adaln_blocks: int = 2,
        mid_16_blocks: int = 4,
        mlp_dw_kernel: int = 0,
        qkv_dw_kernel: int = 0,
        qkv_dw_fraction: float = 1.0,
        # MLP
        mlp_mult: float = 2.5,
        # Per-block gradient checkpointing during training. Each Swin-3D /
        # Full-ST block is individually checkpointed → peak activation memory
        # during backward is ~one block's activations, not the whole stage.
        gradient_checkpoint: bool = False,
        # Overlap patchify/depatchify (stem + aux_depatchify).
        # overlap_patchify is a legacy bool alias (True → mult=3).
        # overlap_mult picks the receptive-field multiplier:
        #   1 = no overlap; 2 = half-overlap (kernel=2·patch);
        #   3 = full-overlap (kernel=3·patch, immediate + 8 neighbors).
        overlap_patchify: bool = False,
        overlap_mult: int = 2,
        depatch_fourier_features: int = 0,
        depatch_fourier_kernel: int = 3,
        # MAE-style pretraining hooks. Disabled in the normal inference path.
        enable_mae_mask_tokens: bool = False,
        bottleneck_head_stride: int | None = 8,
        # RoPE training-time augmentations (DINOv3-style, per-forward coherent).
        # None / 0 / 1.0 respectively disables each.
        rope_shift_coords: float = None,
        rope_jitter_coords: float = None,
        rope_rescale_coords: float = None,
        rope_base: float = 100.0,
    ):
        super().__init__()
        self.d_16 = d_16
        self.d_64 = d_64
        self.window_HW = window_HW
        self.gradient_checkpoint = gradient_checkpoint
        self.enable_mae_mask_tokens = bool(enable_mae_mask_tokens)
        self.bottleneck_head_stride = bottleneck_head_stride
        self.hint_adaln_blocks = max(0, int(hint_adaln_blocks))
        self.mid_16_blocks_count = max(0, int(mid_16_blocks))
        self.recurrent_64_sites = tuple(
            int(i) for i in (recurrent_64_sites or ())
        )
        if len(set(self.recurrent_64_sites)) != len(self.recurrent_64_sites):
            raise ValueError(
                f"recurrent_64_sites contains duplicates: "
                f"{self.recurrent_64_sites}")
        bad_sites = [
            i for i in self.recurrent_64_sites
            if i < 0 or i >= int(n_body_64)
        ]
        if bad_sites:
            raise ValueError(
                f"recurrent_64_sites {bad_sites} outside body block range "
                f"0..{int(n_body_64) - 1}")
        qkv_plan = _stage1_qkv_kernels(
            n_enc_16,
            n_body_64,
            self.mid_16_blocks_count,
            n_dec_16,
            qkv_dw_kernel,
            qkv_dw_fraction,
        )
        qkv_pos = 0
        qkv_enc = qkv_plan[qkv_pos:qkv_pos + n_enc_16]
        qkv_pos += n_enc_16
        qkv_body = qkv_plan[qkv_pos:qkv_pos + n_body_64]
        qkv_pos += n_body_64
        qkv_mid = qkv_plan[qkv_pos:qkv_pos + self.mid_16_blocks_count]
        qkv_pos += self.mid_16_blocks_count
        qkv_dec = qkv_plan[qkv_pos:qkv_pos + n_dec_16]
        self.qkv_dw_active_blocks = sum(1 for k in qkv_plan if k > 0)
        self.qkv_dw_total_blocks = len(qkv_plan)

        # Stem: 4 input channels (RGB + hint) → d_16 at /16 via k=16 s=16.
        # Normal Kaiming init (no zero-init here — we want the stem to
        # actually produce features from t=0 since stage 1 has nothing
        # else at /16 to take up the slack).
        self.stem = PatchifyConv(4, d_16, patch=16, zero_init_first=0,
                                 overlap=overlap_patchify,
                                 overlap_mult=overlap_mult)

        # Shared RoPE instances per attention scale — saves frequency tables
        # and (negligibly) memory. The pre-down /16 blocks use spatial-only
        # RoPE; later /16 and /64 blocks use spatiotemporal RoPE.
        head_dim_16 = d_16 // num_heads_16
        head_dim_64 = d_64 // num_heads_64
        _rope_kw = dict(base=rope_base,
                        shift_coords=rope_shift_coords,
                        jitter_coords=rope_jitter_coords,
                        rescale_coords=rope_rescale_coords)
        self.rope_16_spatial = RoPE2D(head_dim_16, **_rope_kw)
        self.rope_16 = RoPE3D(head_dim_16, **_rope_kw)
        self.rope_64 = RoPE3D(head_dim_64, **_rope_kw)

        # /16 encoder side: per-frame full-spatial attention. This gives the
        # high-res hint/RGB features global spatial context before downsample,
        # while temporal mixing remains concentrated in the /64 body.
        assert n_enc_16 >= 1
        self.enc_16 = nn.ModuleList([
            FullSpatialBlock3D(
                d_16, num_heads_16,
                mlp_mult=mlp_mult, rope=self.rope_16_spatial,
                local_mix_kernel=local_mix_kernel,
                local_mix_init_std=local_mix_init_std,
                adaln_cond_dim=(1 if i < self.hint_adaln_blocks else 0),
                mlp_dw_kernel=mlp_dw_kernel,
                qkv_dw_kernel=qkv_enc[i],
            )
            for i in range(n_enc_16)
        ])

        # /16 → /64 downsample — kernel-8 stride-4. Produces d_down_aux=d_16
        # channels that get concat'd with VFM → FusionLinSwiglu
        # → d_64. Init as exact center-cell area pooling so the aux stream has
        # useful /16-derived content immediately; fusion still zeros aux
        # contribution at t=0, so the model function remains VFM-only initially.
        d_down_aux = d_16                                # 512 by default
        self.d_down_aux = d_down_aux
        self.cradio_dim = cradio_dim
        self.moge_dim = moge_dim
        self.rvm_dim = rvm_dim
        self.vfm_channels = cradio_dim + moge_dim + rvm_dim
        self.down = Downsample4(d_16, d_down_aux, kernel=8, stride=4, norm=False)
        _init_center_area_downsample_(self.down.conv)

        # Fusion block: linear projection with SwiGLU wrapped AROUND it.
        #   in_dim  = vfm + d_down_aux
        #   out_dim = d_64             (1536)
        #   hidden  = in_dim · mlp_mult
        # Linear shortcut: semi-orthogonal VFM projection, zero-for-aux.
        # SwiGLU: up-projects from full in_dim before down-project, so the
        # nonlinear gate sees the full input before compression. w_down is
        # zero-inited → SwiGLU contributes 0 at t=0.
        # Hidden size is pinned before RVM so retrofitting the recurrent
        # stream does not expand an already-trained fusion MLP. C-RADIO is
        # not a retrofit path for existing checkpoints; when explicitly enabled,
        # it is treated as part of the base frozen-foundation budget.
        pre_rvm_in = (
            self.cradio_dim + self.moge_dim + d_down_aux
        )
        hidden_pinned = int(round(pre_rvm_in * mlp_mult))
        self.body64_direct_concat = bool(body64_direct_concat)
        fuse_in_dim = self.vfm_channels + d_down_aux
        if self.body64_direct_concat:
            if d_64 != fuse_in_dim:
                raise ValueError(
                    f"body64_direct_concat requires d_64 == VFM+aux "
                    f"({fuse_in_dim}); got d_64={d_64}")
            self.fuse = None
        else:
            self.fuse = FusionLinSwiglu(
                vfm_dim=self.vfm_channels, aux_dim=d_down_aux,
                out_dim=d_64, mult=mlp_mult, bias=True,
                hidden=hidden_pinned,
            )

        # /64 body. Full spatiotemporal attention — no windowing. Token
        # count (H/64 · W/64 · T) is small enough that full quadratic
        # attention is cheap here.
        self.body_64 = nn.ModuleList([
            FullSpatioTemporalBlock(d_64, num_heads_64, mlp_mult=mlp_mult,
                                     rope=self.rope_64,
                                     local_mix_kernel=body64_local_mix_kernel,
                                     local_mix_init_std=body64_local_mix_init_std,
                                     mlp_dw_kernel=mlp_dw_kernel,
                                     mlp_hidden=body64_mlp_hidden,
                                     qkv_dw_kernel=qkv_body[i])
            for i in range(n_body_64)
        ])
        self.body_64_split = n_body_64 // 2
        self.recurrent_64 = nn.ModuleList([
            FullResolutionRecurrentMemorySite(
                d_64, num_heads_64, rope=RoPE2D(head_dim_64, **_rope_kw))
            for _ in self.recurrent_64_sites
        ])
        self._recurrent_64_index = {
            site: i for i, site in enumerate(self.recurrent_64_sites)
        }

        # Mid-stage /64 → /16 → /64 detour. The up/down projections are
        # zero-inited so the loop is exact-identity at init while still
        # allowing a local /16 refinement route to grow during training.
        self.mid_up = Upsample4ICNR(d_64, d_16, r=4, post_smooth=True, norm=False)
        self.mid_skip_scale = nn.Parameter(torch.ones(d_16))
        self.mid_16 = nn.ModuleList(
            _stage1_mid16_blocks(
                d_16, num_heads_16, window_HW, mlp_mult, self.rope_16,
                n_blocks=self.mid_16_blocks_count,
                local_mix_kernel=local_mix_kernel,
                local_mix_init_std=local_mix_init_std,
                mlp_dw_kernel=mlp_dw_kernel,
                qkv_dw_kernel=qkv_mid,
            )
        )
        self.mid_down = Downsample4(d_16, d_64, kernel=8, stride=4, norm=False)
        with torch.no_grad():
            nn.init.zeros_(self.mid_up.proj.weight)
            if self.mid_up.proj.bias is not None:
                nn.init.zeros_(self.mid_up.proj.bias)
            nn.init.zeros_(self.mid_down.conv.weight)
            if self.mid_down.conv.bias is not None:
                nn.init.zeros_(self.mid_down.conv.bias)

        # /64 → /16 upsample. ICNR-inited + post-conv smoothing to mimic
        # the overlap the kernel-8 downsample provided on the encoder side.
        self.up = Upsample4ICNR(d_64, d_16, r=4, post_smooth=True, norm=False)

        # Learned per-channel U-Net skip from /16 encoder → /16 decoder.
        # Shape (d_16,). Initialized to 1.0 — at t=0 the decoder sees the
        # encoder /16 features at full strength. The model can learn to
        # scale this down if the /64 body's output is preferred. Alternative
        # was init=0 (start with no skip, grow), but encoder features contain
        # early-pipeline RGB-derived detail that's useful to have available
        # from step 1.
        self.uskip_scale = nn.Parameter(torch.ones(d_16))
        self.patch_late_skip_scale = nn.Parameter(torch.zeros(d_16))

        # /16 decoder side: mixed Swin/CSwin pattern. The 6-block default is
        # the compact 2 Swin → 2 CSwin → 2 Swin layout; the optional 8-block
        # layout inserts a local Swin pair between the two CSwin sweeps. The
        # late patch-stem skip is injected before the final Swin pair.
        assert n_dec_16 in (6, 8), (
            f"n_dec_16 must be 6 or 8 for the CSwin-in-the-middle pattern; got {n_dec_16}.")
        self.dec_16 = nn.ModuleList(
            _stage1_dec_blocks(d_16, num_heads_16, window_HW, mlp_mult,
                                self.rope_16, n_blocks=n_dec_16,
                                local_mix_kernel=local_mix_kernel,
                                local_mix_init_std=local_mix_init_std,
                                mlp_dw_kernel=mlp_dw_kernel,
                                qkv_dw_kernel=qkv_dec)
        )

        # Aux head: baton at /16 (d=512) → 4-channel prediction at /2 via
        # 8× depatchify. With production dims and non-overlap patching, the
        # FG rows are initialized as area-downsampled inverse RGB patch basis
        # and alpha rows remain zero.
        self.aux_depatchify = Depatchify(d_16, out_channels=4, r=8,
                                          zero_init_all=True, bias=True,
                                          overlap=overlap_patchify,
                                          overlap_mult=overlap_mult,
                                          fourier_features=depatch_fourier_features,
                                          fourier_kernel=depatch_fourier_kernel)
        self.rgb_passthrough_init = init_orthogonal_rgb_passthrough_(
            self.stem, self.aux_depatchify,
            token_offset=0,
            output_rgb_offset=1,
            downsample=2,
        )
        if self.rgb_passthrough_init:
            nn.init.zeros_(self.up.proj.weight)
            if self.up.proj.bias is not None:
                nn.init.zeros_(self.up.proj.bias)
        if self.enable_mae_mask_tokens:
            self.mae_token_16 = nn.Parameter(torch.zeros(d_16))
            self.mae_token_64_aux = nn.Parameter(torch.zeros(d_down_aux))
            if self.vfm_channels > 0:
                self.mae_token_vfm = nn.Parameter(torch.zeros(self.vfm_channels))
        if bottleneck_head_stride is not None:
            if bottleneck_head_stride not in (4, 8):
                raise ValueError(
                    "bottleneck_head_stride must be 4 or 8 when enabled")
            r = 64 // int(bottleneck_head_stride)
            self.bottleneck_depatchify = Depatchify(
                d_64, out_channels=4, r=r,
                zero_init_all=True, bias=True,
                overlap=overlap_patchify,
                overlap_mult=overlap_mult,
                fourier_features=depatch_fourier_features,
                fourier_kernel=depatch_fourier_kernel,
            )

    # -----------------------------------------------------------------

    def _ckpt_block(
        self,
        blk,
        x: torch.Tensor,
        t_stride: float,
        adaln_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Call `blk(x, t_stride=t_stride)` — wrapped in `torch.utils.checkpoint`
        when training with gradient_checkpoint enabled so only one block's
        activations are resident at a time during backward."""
        if adaln_cond is not None:
            if self.gradient_checkpoint and self.training and torch.is_grad_enabled():
                return torch.utils.checkpoint.checkpoint(
                    blk, x, t_stride, adaln_cond, use_reentrant=False,
                )
            return blk(x, t_stride, adaln_cond)
        if self.gradient_checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                blk, x, t_stride, use_reentrant=False,
            )
        return blk(x, t_stride=t_stride)

    def _swin3d_run(
        self,
        blocks,
        x_5d: torch.Tensor,
        t_stride: float,
        hint_quality: torch.Tensor | None = None,
        late_skip_5d: torch.Tensor | None = None,
        late_skip_scale: torch.Tensor | None = None,
        late_skip_index: int | None = None,
    ) -> torch.Tensor:
        """Run a list of Swin3D blocks on a 5D tensor (B, T, C, H, W).

        Swin blocks accept (B, T, H, W, C), so permute in & out here.
        """
        frame_chunk = int(getattr(self, "low_vram_enc_frame_chunk", 0) or 0)
        if (frame_chunk > 0 and not self.training and late_skip_5d is None
                and len(blocks) > 0
                and all(isinstance(blk, FullSpatialBlock3D) for blk in blocks)
                and x_5d.shape[1] > frame_chunk):
            outs = []
            for t0 in range(0, x_5d.shape[1], frame_chunk):
                t1 = min(t0 + frame_chunk, x_5d.shape[1])
                x = x_5d[:, t0:t1].permute(0, 1, 3, 4, 2).contiguous()
                hq = (
                    hint_quality[:, t0:t1]
                    if hint_quality is not None else None
                )
                for blk in blocks:
                    cond = hq if getattr(blk, "adaln", None) is not None else None
                    x = self._ckpt_block(blk, x, t_stride, adaln_cond=cond)
                outs.append(x.permute(0, 1, 4, 2, 3).contiguous())
            return torch.cat(outs, dim=1)

        # (B, T, C, H, W) → (B, T, H, W, C)
        x = x_5d.permute(0, 1, 3, 4, 2).contiguous()
        skip = None
        if late_skip_5d is not None:
            skip = late_skip_5d.permute(0, 1, 3, 4, 2).contiguous()
            if late_skip_scale is None:
                raise ValueError("late_skip_scale is required with late_skip_5d")
            if late_skip_index is None:
                late_skip_index = max(len(blocks) - 2, 0)
            scale = late_skip_scale.to(device=x.device, dtype=x.dtype).view(
                1, 1, 1, 1, -1)
        else:
            scale = None
        for i, blk in enumerate(blocks):
            if skip is not None and i == late_skip_index:
                x = x + skip * scale
            cond = hint_quality if getattr(blk, "adaln", None) is not None else None
            x = self._ckpt_block(blk, x, t_stride, adaln_cond=cond)
        return x.permute(0, 1, 4, 2, 3).contiguous()

    def _swin3d_run_lean(
        self,
        blocks,
        holder: list,
        t_stride: float,
        skip_holder: list | None = None,
        late_skip_scale: torch.Tensor | None = None,
        late_skip_index: int | None = None,
    ) -> torch.Tensor:
        """`_swin3d_run` for the mid/decoder stacks with tensor ownership
        handed over (`inference_lean`): the (B, T, C, H, W) input arrives in a
        one-element list so this method holds the only reference and it is
        freed once permuted, and the late skip is permuted only right before
        the block that adds it. Same operations in the same order."""
        src = holder.pop()
        x = src.permute(0, 1, 3, 4, 2).contiguous()
        if x.data_ptr() == src.data_ptr():
            # Already NTHWC: own a copy, since blocks may add into x in place.
            x = x.clone()
        del src
        scale = None
        if skip_holder is not None:
            if late_skip_scale is None:
                raise ValueError("late_skip_scale is required with a late skip")
            if late_skip_index is None:
                late_skip_index = max(len(blocks) - 2, 0)
            scale = late_skip_scale.to(device=x.device, dtype=x.dtype).view(
                1, 1, 1, 1, -1)
        for i, blk in enumerate(blocks):
            if skip_holder is not None and i == late_skip_index:
                skip = skip_holder.pop().to(x.device, non_blocking=True).permute(
                    0, 1, 3, 4, 2).contiguous()
                if _skip_add_in_place(self, x, skip, scale):
                    x.add_(skip * scale)
                else:
                    x = x + skip * scale
                del skip
            x = self._ckpt_block(blk, x, t_stride)
        return x.permute(0, 1, 4, 2, 3).contiguous()

    def _full_run(self, blocks, x_5d: torch.Tensor, t_stride: float) -> torch.Tensor:
        """Run full-spatiotemporal blocks on (B, T, C, H, W)."""
        x = x_5d.permute(0, 1, 3, 4, 2).contiguous()
        for blk in blocks:
            x = self._ckpt_block(blk, x, t_stride)
        return x.permute(0, 1, 4, 2, 3).contiguous()

    def _full_run_nthwc(
        self,
        blocks,
        x: torch.Tensor,
        t_stride: float,
    ) -> torch.Tensor:
        """Run full-spatiotemporal blocks on native (B, T, H, W, C) layout."""
        for blk in blocks:
            x = self._ckpt_block(blk, x, t_stride)
        return x

    def _ckpt_body64_with_recurrent(
        self,
        blk,
        site,
        x: torch.Tensor,
        state: torch.Tensor | None,
        t_stride: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gradient_checkpoint and self.training and torch.is_grad_enabled():
            if state is None:
                def fn(inp):
                    return blk(
                        inp, t_stride,
                        recurrent_site=site,
                        recurrent_state=None,
                    )
                return torch.utils.checkpoint.checkpoint(
                    fn, x, use_reentrant=False,
                )

            def fn(inp, st):
                return blk(
                    inp, t_stride,
                    recurrent_site=site,
                    recurrent_state=st,
                )
            return torch.utils.checkpoint.checkpoint(
                fn, x, state, use_reentrant=False,
            )
        return blk(
            x, t_stride=t_stride,
            recurrent_site=site,
            recurrent_state=state,
        )

    def _run_body64_range(
        self,
        start: int,
        end: int,
        x: torch.Tensor,
        t_stride: float,
        recurrent_states: list[torch.Tensor | None] | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None] | None]:
        for block_idx in range(start, end):
            site_idx = self._recurrent_64_index.get(block_idx)
            if site_idx is not None:
                if recurrent_states is None:
                    raise RuntimeError(
                        "internal error: recurrent site active without states")
                x, recurrent_states[site_idx] = self._ckpt_body64_with_recurrent(
                    self.body_64[block_idx],
                    self.recurrent_64[site_idx],
                    x,
                    recurrent_states[site_idx],
                    t_stride,
                )
            else:
                x = self._ckpt_block(self.body_64[block_idx], x, t_stride)
        return x, recurrent_states

    def _apply_mae_token(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        token: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if mask is None:
            return x
        if not self.enable_mae_mask_tokens:
            raise RuntimeError(f"{name} received an MAE mask but MAE tokens are disabled")
        expected = (x.shape[0], x.shape[1], 1, x.shape[-2], x.shape[-1])
        if tuple(mask.shape) != expected:
            raise ValueError(
                f"{name} mask shape {tuple(mask.shape)} != expected {expected}")
        mask = mask.to(device=x.device, dtype=torch.bool)
        tok = token.to(device=x.device, dtype=x.dtype).view(1, 1, -1, 1, 1)
        return torch.where(mask, tok, x)

    @staticmethod
    def _resolve_hint_quality(
        hint: torch.Tensor,
        hint_quality: torch.Tensor | None,
    ) -> torch.Tensor:
        if hint_quality is None:
            return (hint.detach().amax(dim=(-1, -2, -3)) > 1e-5).to(dtype=hint.dtype)
        q = hint_quality.to(device=hint.device, dtype=hint.dtype)
        while q.dim() > 2 and q.shape[-1] == 1:
            q = q.squeeze(-1)
        if q.dim() != 2:
            raise ValueError(
                f"hint_quality must be (B,T) or trailing-singleton equivalent; got {tuple(q.shape)}")
        if tuple(q.shape) != tuple(hint.shape[:2]):
            raise ValueError(
                f"hint_quality shape {tuple(q.shape)} != hint B,T {tuple(hint.shape[:2])}")
        return q.clamp_(-1.0, 1.0)

    def forward(
        self,
        rgb: 'torch.Tensor | None' = None,  # (B, T, 3, H, W) signed asinh encoded scene-linear RGB
        hint: 'torch.Tensor | None' = None,  # (B, T, 1, H, W) in [0, 1]
        moge_feat: 'torch.Tensor | None' = None,  # (B, T, moge_dim, H/64, W/64)
        cradio_feat: 'torch.Tensor | None' = None,  # (B, T, cradio_dim, H/64, W/64)
        rvm_feat: 'torch.Tensor | None' = None,  # (B, T, rvm_dim, H/64, W/64) if enabled
        t_stride: float = 1.0,
        mae_mask_16: 'torch.Tensor | None' = None,  # (B,T,1,H/16,W/16)
        mae_mask_64: 'torch.Tensor | None' = None,  # (B,T,1,H/64,W/64)
        mae_vfm_mask_64: 'torch.Tensor | None' = None,  # (B,T,1,H/64,W/64)
        hint_quality: 'torch.Tensor | None' = None,  # (B,T), -1=decide subject, 0=no frame hint, 1=perfect
        recurrent_states: 'tuple[torch.Tensor | None, ...] | None' = None,
        inputs: 'dict | None' = None,
    ) -> dict:
        """Returns dict with:
            baton: (B, T, d_16, H/16, W/16)  — feeds stage 2
            aux:   (B, T, 4, H/2, W/2)       — stage-1 supervised output

        `inputs` (inference): rgb, hint, moge_feat, cradio_feat and rvm_feat
        handed over in a dict that this method empties, so when the caller
        keeps no other reference the full-resolution input is freed after the
        stem and the /64 features after the fusion. Same operations.
        """
        if inputs is not None:
            rgb, hint = inputs.pop("rgb"), inputs.pop("hint")
            moge_feat = inputs.pop("moge_feat")
            cradio_feat = inputs.pop("cradio_feat", None)
            rvm_feat = inputs.pop("rvm_feat", None)
        B, T, _, H, W = rgb.shape
        assert H % 64 == 0 and W % 64 == 0, (
            f"Stage1: H,W must be multiples of 64 (shape_quantum); got {(H, W)}")

        # Merge B·T for conv ops that don't care about T
        BT = B * T
        hint_quality = self._resolve_hint_quality(hint, hint_quality)
        stem_chunk = int(getattr(self, "inference_stem_frame_chunk", 0) or 0)
        if stem_chunk > 0 and not self.training:
            # Opt-in: concat and patchify a few frames at a time (the stem is
            # per frame), so the full-resolution concat and cuDNN's layout
            # buffers exist for one chunk instead of the whole window.
            # Bit-identical: cuDNN keeps its algorithm for the smaller batch.
            f_16 = None
            for t0 in range(0, T, stem_chunk):
                t1 = min(t0 + stem_chunk, T)
                x_t = torch.cat([rgb[:, t0:t1], hint[:, t0:t1]], dim=2)
                y = self.stem(x_t.reshape(B * (t1 - t0), 4, H, W))
                del x_t
                if f_16 is None:
                    f_16 = y.new_empty((B, T) + tuple(y.shape[1:]))
                f_16[:, t0:t1] = y.view(B, t1 - t0, *y.shape[1:])
                del y
            del rgb, hint
        else:
            x_in = torch.cat([rgb, hint], dim=2).reshape(BT, 4, H, W)
            del rgb, hint

            # /1 → /16 patchify
            f_16 = self.stem(x_in)                                   # (BT, d_16, H/16, W/16)
            del x_in
            f_16 = f_16.reshape(B, T, self.d_16, H // 16, W // 16)
        if mae_mask_16 is not None:
            f_16 = self._apply_mae_token(f_16, mae_mask_16,
                                         self.mae_token_16, "stage1 /16")
        f_16_patch = f_16

        # /16 encoder side (per-frame full-spatial). Keep a handle for the
        # U-skip before the downsample eats the spatial detail.
        f_16_enc = self._swin3d_run(
            self.enc_16, f_16, t_stride, hint_quality=hint_quality)
        lean = bool(getattr(self, "inference_lean", False)) and not self.training
        if lean:
            del f_16  # f_16_patch is the only handle the late skip needs
            if getattr(self, "inference_offload_late_skip", False):
                # Parked in host memory until the decoder's late skip needs it.
                if getattr(self, "_late_skip_host", None) is None:
                    self._late_skip_host = HostBuffer()
                parked = self._late_skip_host.view(f_16_patch.shape, f_16_patch.dtype)
                parked.copy_(f_16_patch, non_blocking=True)
                f_16_patch = parked

        # /16 → /64 downsample (center-cell area init; produces d_down_aux channels)
        f_16_flat = f_16_enc.reshape(BT, self.d_16, H // 16, W // 16)
        f_64_down = self.down(f_16_flat)                          # (BT, d_down_aux, H/64, W/64)
        del f_16_flat
        h_64, w_64 = H // 64, W // 64
        f_64_down = f_64_down.reshape(B, T, self.d_down_aux, h_64, w_64)
        if mae_mask_64 is not None:
            f_64_down = self._apply_mae_token(
                f_64_down, mae_mask_64, self.mae_token_64_aux,
                "stage1 /64 aux")

        # VFM features at /64 (already channel-normalized by FrozenFoundations).
        # Concat order is fixed at [cradio | moge | rvm] — the fuse layer's
        # linear_shortcut init relies on this ordering.
        assert moge_feat.shape[-2:] == (h_64, w_64), (
            f"moge_feat spatial {moge_feat.shape[-2:]} != /64 ({(h_64, w_64)})")
        vfm_parts = []
        if self.cradio_dim > 0:
            assert cradio_feat is not None, (
                "cradio_dim > 0 requires cradio_feat to be passed in")
            assert cradio_feat.shape[-2:] == (h_64, w_64), (
                f"cradio_feat spatial {cradio_feat.shape[-2:]} != /64 ({(h_64, w_64)})")
            vfm_parts.append(cradio_feat)
        vfm_parts.append(moge_feat)
        if self.rvm_dim > 0:
            assert rvm_feat is not None, (
                "rvm_dim > 0 requires rvm_feat to be passed in")
            # RVM shares C-RADIO's input (H/4) and patch size (16), so features
            # land on the /64 grid directly — no resize needed.
            assert rvm_feat.shape[-2:] == (h_64, w_64), (
                f"rvm_feat spatial {rvm_feat.shape[-2:]} != /64 ({(h_64, w_64)}); "
                f"RVM should be fed at C-RADIO's H/4 input size")
            vfm_parts.append(rvm_feat)
        vfm_cat = torch.cat(vfm_parts, dim=2)                  # (B, T, vfm_C, h, w)

        # Pad VFM to match self.vfm_channels if something got disabled
        # (e.g. one foundation off in smoke-test) so the SwiGLU input shape
        # stays constant.
        pad_c = self.vfm_channels - vfm_cat.shape[2]
        if pad_c > 0:
            vfm_cat = F.pad(vfm_cat, (0, 0, 0, 0, 0, pad_c))
        if mae_vfm_mask_64 is not None:
            if not hasattr(self, "mae_token_vfm"):
                raise RuntimeError(
                    "received an MAE VFM mask but VFM MAE token is disabled")
            vfm_cat = self._apply_mae_token(
                vfm_cat, mae_vfm_mask_64, self.mae_token_vfm,
                "stage1 /64 VFM")

        # Full concat at /64 (vfm + downsampled aux) → either direct /64 body
        # input or fusion (linear+SwiGLU).
        combined = torch.cat([vfm_cat, f_64_down], dim=2)      # (B, T, vfm+aux, h, w)
        del vfm_cat, f_64_down, vfm_parts, moge_feat, cradio_feat, rvm_feat
        tokens = combined.permute(0, 1, 3, 4, 2).contiguous()  # (B, T, h, w, fuse_in_dim)
        del combined
        f_64 = tokens if self.fuse is None else self.fuse(tokens)  # (B, T, h, w, d_64)
        del tokens

        # /64 body, with a mid-stage detour back through /16 for local
        # refinement before returning to /64 for the remaining global blocks.
        recurrent_states_out = None
        if len(self.recurrent_64) > 0:
            if recurrent_states is None:
                recurrent_states_out = [None for _ in self.recurrent_64]
            else:
                if len(recurrent_states) != len(self.recurrent_64):
                    raise ValueError(
                        f"got {len(recurrent_states)} recurrent states for "
                        f"{len(self.recurrent_64)} recurrent sites")
                recurrent_states_out = list(recurrent_states)

        split = self.body_64_split
        if recurrent_states_out is None:
            f_64 = self._full_run_nthwc(self.body_64[:split], f_64, t_stride)
        else:
            f_64, recurrent_states_out = self._run_body64_range(
                0, split, f_64, t_stride, recurrent_states_out)

        f_64_chw = f_64.permute(0, 1, 4, 2, 3).contiguous()
        f_64_mid_flat = f_64_chw.reshape(BT, self.d_64, H // 64, W // 64)
        del f_64_chw
        f_16_mid_up = self.mid_up(f_64_mid_flat).reshape(
            B, T, self.d_16, H // 16, W // 16)
        del f_64_mid_flat
        mid_scale = self.mid_skip_scale.view(1, 1, self.d_16, 1, 1)
        if _skip_add_in_place(self, f_16_enc, mid_scale, f_16_mid_up):
            f_16_mid = f_16_enc.mul_(mid_scale).add_(f_16_mid_up)
        else:
            f_16_mid = f_16_mid_up + mid_scale * f_16_enc
        del f_16_mid_up, f_16_enc
        if len(self.mid_16) > 0 and lean:
            holder = [f_16_mid]
            del f_16_mid
            f_16_mid = self._swin3d_run_lean(self.mid_16, holder, t_stride)
        elif len(self.mid_16) > 0:
            f_16_mid = self._swin3d_run(self.mid_16, f_16_mid, t_stride)
        f_16_mid_flat = f_16_mid.reshape(BT, self.d_16, H // 16, W // 16)
        f_64_delta = self.mid_down(f_16_mid_flat).reshape(
            B, T, self.d_64, H // 64, W // 64)
        del f_16_mid_flat
        f_64 = f_64 + f_64_delta.permute(0, 1, 3, 4, 2).contiguous()
        del f_64_delta

        if recurrent_states_out is None:
            f_64 = self._full_run_nthwc(self.body_64[split:], f_64, t_stride)
        else:
            f_64, recurrent_states_out = self._run_body64_range(
                split, len(self.body_64), f_64, t_stride,
                recurrent_states_out)
        f_64_chw = f_64.permute(0, 1, 4, 2, 3).contiguous()
        del f_64

        bottleneck_out = None
        if self.bottleneck_head_stride is not None:
            f_64_head = f_64_chw.reshape(BT, self.d_64, H // 64, W // 64)
            raw = self.bottleneck_depatchify(f_64_head)
            del f_64_head
            s = int(self.bottleneck_head_stride)
            bottleneck_out = raw.reshape(B, T, 4, H // s, W // s)
            del raw

        # /64 → /16 upsample
        f_64_flat = f_64_chw.reshape(BT, self.d_64, H // 64, W // 64)
        del f_64_chw
        f_16_up = self.up(f_64_flat)                               # (BT, d_16, H/16, W/16)
        del f_64_flat
        f_16_up = f_16_up.reshape(B, T, self.d_16, H // 16, W // 16)

        # Learned-weight U-Net skip: add encoder /16 features scaled by a
        # per-channel parameter. Broadcasts (d_16,) over (B, T, C, H, W).
        scale = self.uskip_scale.view(1, 1, self.d_16, 1, 1)
        if _skip_add_in_place(self, f_16_mid, scale, f_16_up):
            f_16_dec = f_16_mid.mul_(scale).add_(f_16_up)
        else:
            f_16_dec = f_16_up + scale * f_16_mid
        del f_16_up, f_16_mid

        # /16 decoder side (Swin-3D)
        if lean:
            holder, patch_holder = [f_16_dec], [f_16_patch]
            del f_16_dec, f_16_patch
            baton = self._swin3d_run_lean(
                self.dec_16, holder, t_stride,
                skip_holder=patch_holder,
                late_skip_scale=self.patch_late_skip_scale,
                late_skip_index=max(len(self.dec_16) - 2, 0),
            )
        else:
            baton = self._swin3d_run(
                self.dec_16, f_16_dec, t_stride,
                late_skip_5d=f_16_patch,
                late_skip_scale=self.patch_late_skip_scale,
                late_skip_index=max(len(self.dec_16) - 2, 0),
            )   # (B, T, d_16, H/16, W/16)
            del f_16_dec, f_16_patch

        # Aux head: /16 baton → /2 pred (α, FG). With production dims and
        # non-overlap patching, FG is the exact area-downsampled inverse of
        # the stem RGB patch basis at init; alpha remains zero.
        BT = B * T
        baton_flat = baton.reshape(BT, self.d_16, H // 16, W // 16)
        aux_raw = self.aux_depatchify(baton_flat)                  # (BT, 4, H/2, W/2)
        del baton_flat
        aux_out = aux_raw.reshape(B, T, 4, H // 2, W // 2)
        del aux_raw

        out = {'baton': baton, 'aux': aux_out, 'bottleneck': bottleneck_out}
        if recurrent_states_out is not None:
            out['recurrent_states'] = tuple(recurrent_states_out)
        return out
