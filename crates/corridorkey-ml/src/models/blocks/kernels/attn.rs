//! CubeCL kernels for 2D/3D attention blocks: fused QKV head-split + QK-RMSNorm + RoPE,
//! headwise sigmoid gating + head merge, CSwin strip packing/unpacking, and Swin window
//! partition/unpartition with cyclic shift.

use cubecl::prelude::*;

use corridorkey_base::core::CkFloat;
use corridorkey_core::compute::row_position;

/// `sigmoid(x) = 1 / (1 + exp(-x))`.
#[cube]
pub fn sigmoid<F: Float>(x: F) -> F {
    F::new(1.0f32) / (F::new(1.0f32) + F::exp(-x))
}

/// Fuses QKV bias addition (when `apply_bias != 0`), head unpacking (`[B, N, 3, H, D]` ->
/// `[B, H, N, D]`), per-head `q_norm` / `k_norm` (`RMSNorm(head_dim)`), and `apply_rotary`
/// (`cos`, `sin` of shape `[N, D/2]`) into a single kernel launch.
///
/// One cube per `(batch, head, token)` vector; unit `p` owns the pair `(2p, 2p + 1)` of q, k and
/// v, so neighbouring units read and write neighbouring addresses. The q / k sums of squares are
/// reduced in f32 through shared memory. `block_size` is a power of two `>= head_dim / 2`.
#[cube(launch)]
pub fn prep_qkv_norm_rope_kernel<F: CkFloat>(
    qkv: &[F],
    qkv_bias: &[F],
    q_norm_w: &[F],
    k_norm_w: &[F],
    cos: &[F],
    sin: &[F],
    q_out: &mut [F],
    k_out: &mut [F],
    v_out: &mut [F],
    total_bhn: u32,
    num_heads: u32,
    n_tokens: u32,
    head_dim: u32,
    apply_bias: u32,
    eps: f32,
    #[comptime] block_size: u32,
) {
    // Declared outside any branch (CUDA LLVM lowering).
    let mut s_q2 = Shared::<[f32]>::new_slice(block_size as usize);
    let mut s_k2 = Shared::<[f32]>::new_slice(block_size as usize);
    let idx = row_position();
    if idx < total_bhn {
        let p = UNIT_POS_X;
        let n = idx % n_tokens;
        let bh = idx / n_tokens;
        let h = bh % num_heads;
        let b = bh / num_heads;

        let dim = num_heads * head_dim;
        let tok_base = (b * n_tokens + n) * 3u32 * dim;
        let q_off = h * head_dim;
        let k_off = dim + q_off;
        let v_off = 2u32 * dim + q_off;
        let half_d = head_dim / 2u32;
        let active = p < half_d;
        let d0 = 2u32 * p;
        let d1 = d0 + 1u32;

        let mut q0 = 0.0f32;
        let mut q1 = 0.0f32;
        let mut k0 = 0.0f32;
        let mut k1 = 0.0f32;
        let mut v0 = 0.0f32;
        let mut v1 = 0.0f32;
        if active {
            q0 = f32::cast_from(qkv[(tok_base + q_off + d0) as usize]);
            q1 = f32::cast_from(qkv[(tok_base + q_off + d1) as usize]);
            k0 = f32::cast_from(qkv[(tok_base + k_off + d0) as usize]);
            k1 = f32::cast_from(qkv[(tok_base + k_off + d1) as usize]);
            v0 = f32::cast_from(qkv[(tok_base + v_off + d0) as usize]);
            v1 = f32::cast_from(qkv[(tok_base + v_off + d1) as usize]);
            if apply_bias != 0u32 {
                q0 += f32::cast_from(qkv_bias[(q_off + d0) as usize]);
                q1 += f32::cast_from(qkv_bias[(q_off + d1) as usize]);
                k0 += f32::cast_from(qkv_bias[(k_off + d0) as usize]);
                k1 += f32::cast_from(qkv_bias[(k_off + d1) as usize]);
                v0 += f32::cast_from(qkv_bias[(v_off + d0) as usize]);
                v1 += f32::cast_from(qkv_bias[(v_off + d1) as usize]);
            }
        }
        s_q2[p as usize] = q0 * q0 + q1 * q1;
        s_k2[p as usize] = k0 * k0 + k1 * k1;
        sync_cube();
        // From the runtime `CUBE_DIM_X`, not the comptime `block_size` (dsp-kitchen §6b).
        let mut stride = CUBE_DIM_X / 2u32;
        while stride > 0u32 {
            if p < stride {
                let sq = s_q2[p as usize] + s_q2[(p + stride) as usize];
                let sk = s_k2[p as usize] + s_k2[(p + stride) as usize];
                s_q2[p as usize] = sq;
                s_k2[p as usize] = sk;
            }
            sync_cube();
            stride /= 2u32;
        }

        if active {
            let inv_d = 1.0f32 / f32::cast_from(head_dim);
            let scale_q = 1.0f32 / f32::sqrt(s_q2[0] * inv_d + eps);
            let scale_k = 1.0f32 / f32::sqrt(s_k2[0] * inv_d + eps);
            q0 = q0 * scale_q * f32::cast_from(q_norm_w[d0 as usize]);
            q1 = q1 * scale_q * f32::cast_from(q_norm_w[d1 as usize]);
            k0 = k0 * scale_k * f32::cast_from(k_norm_w[d0 as usize]);
            k1 = k1 * scale_k * f32::cast_from(k_norm_w[d1 as usize]);
            let rope = n * half_d + p;
            let c = f32::cast_from(cos[rope as usize]);
            let s = f32::cast_from(sin[rope as usize]);
            let out_base = ((b * num_heads + h) * n_tokens + n) * head_dim;
            q_out[(out_base + d0) as usize] = F::cast_from(q0 * c - q1 * s);
            q_out[(out_base + d1) as usize] = F::cast_from(q1 * c + q0 * s);
            k_out[(out_base + d0) as usize] = F::cast_from(k0 * c - k1 * s);
            k_out[(out_base + d1) as usize] = F::cast_from(k1 * c + k0 * s);
            v_out[(out_base + d0) as usize] = F::cast_from(v0);
            v_out[(out_base + d1) as usize] = F::cast_from(v1);
        }
    }
}

/// Fuses headwise sigmoid gating (`HeadwiseAttentionGate`) with head transpose + flatten:
/// reads `attn` (`[B, num_heads, N, head_dim]`) and `gate_logits` (`[B, N, num_heads]`),
/// and writes `out` (`[B, N, num_heads * head_dim]`).
#[cube(launch)]
pub fn merge_heads_gate_kernel<F: CkFloat>(
    attn: &[F],
    gate_logits: &[F],
    output: &mut [F],
    total_elems: u32,
    num_heads: u32,
    n_tokens: u32,
    head_dim: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let d = idx % head_dim;
        let bnh = idx / head_dim;
        let h = bnh % num_heads;
        let bn = bnh / num_heads;
        let n = bn % n_tokens;
        let b = bn / n_tokens;

        let attn_idx = (((b * num_heads + h) * n_tokens + n) * head_dim + d) as usize;
        let gate_idx = (bn * num_heads + h) as usize;
        let g = sigmoid::<F>(gate_logits[gate_idx]);
        output[idx as usize] = attn[attn_idx] * g;
    }
}

/// Fuses `CSwinAttention2D` QKV preparation: splits heads into H-stripes (`0..h_heads`,
/// `[B * W, h_heads, H, D]`) and W-stripes (`h_heads..num_heads`, `[B * H, w_heads, W, D]`),
/// applies `q_norm` and `k_norm`, and applies axial 2D RoPE (`cos_h`/`sin_h` on H-stripes,
/// `cos_w`/`sin_w` on W-stripes). Rows are `(b, head, token)`, token fastest.
///
/// One cube per `(token, head)` vector; unit `p` owns the pair `(2p, 2p + 1)` of q, k and v
/// (coalesced reads and writes), q / k sums of squares reduced in f32 through shared memory.
/// `block_size` is a power of two `>= head_dim / 2`.
#[cube(launch)]
pub fn prep_cswin2d_qkv_kernel<F: CkFloat>(
    qkv: &[F],
    qkv_bias: &[F],
    q_norm_w: &[F],
    k_norm_w: &[F],
    cos_h: &[F],
    sin_h: &[F],
    cos_w: &[F],
    sin_w: &[F],
    q_h: &mut [F],
    k_h: &mut [F],
    v_h: &mut [F],
    q_w: &mut [F],
    k_w: &mut [F],
    v_w: &mut [F],
    total_rows: u32,
    h_spatial: u32,
    w_spatial: u32,
    num_heads: u32,
    head_dim: u32,
    apply_bias: u32,
    eps: f32,
    #[comptime] block_size: u32,
) {
    // Declared outside any branch (CUDA LLVM lowering).
    let mut s_q2 = Shared::<[f32]>::new_slice(block_size as usize);
    let mut s_k2 = Shared::<[f32]>::new_slice(block_size as usize);
    let idx = row_position();
    if idx < total_rows {
        let p = UNIT_POS_X;
        let n_tokens = h_spatial * w_spatial;
        let tok = idx % n_tokens;
        let bh = idx / n_tokens;
        let h = bh % num_heads;
        let b = bh / num_heads;
        let y = tok / w_spatial;
        let x = tok % w_spatial;
        let tok_base = (b * n_tokens + tok) * 3u32 * num_heads * head_dim;
        let dim = num_heads * head_dim;
        let q_off = h * head_dim;
        let k_off = dim + q_off;
        let v_off = 2u32 * dim + q_off;
        let half_d = head_dim / 2u32;
        let active = p < half_d;
        let d0 = 2u32 * p;
        let d1 = d0 + 1u32;

        let mut q0 = 0.0f32;
        let mut q1 = 0.0f32;
        let mut k0 = 0.0f32;
        let mut k1 = 0.0f32;
        let mut v0 = 0.0f32;
        let mut v1 = 0.0f32;
        if active {
            q0 = f32::cast_from(qkv[(tok_base + q_off + d0) as usize]);
            q1 = f32::cast_from(qkv[(tok_base + q_off + d1) as usize]);
            k0 = f32::cast_from(qkv[(tok_base + k_off + d0) as usize]);
            k1 = f32::cast_from(qkv[(tok_base + k_off + d1) as usize]);
            v0 = f32::cast_from(qkv[(tok_base + v_off + d0) as usize]);
            v1 = f32::cast_from(qkv[(tok_base + v_off + d1) as usize]);
            if apply_bias != 0u32 {
                q0 += f32::cast_from(qkv_bias[(q_off + d0) as usize]);
                q1 += f32::cast_from(qkv_bias[(q_off + d1) as usize]);
                k0 += f32::cast_from(qkv_bias[(k_off + d0) as usize]);
                k1 += f32::cast_from(qkv_bias[(k_off + d1) as usize]);
                v0 += f32::cast_from(qkv_bias[(v_off + d0) as usize]);
                v1 += f32::cast_from(qkv_bias[(v_off + d1) as usize]);
            }
        }
        s_q2[p as usize] = q0 * q0 + q1 * q1;
        s_k2[p as usize] = k0 * k0 + k1 * k1;
        sync_cube();
        // From the runtime `CUBE_DIM_X`, not the comptime `block_size` (dsp-kitchen §6b).
        let mut stride = CUBE_DIM_X / 2u32;
        while stride > 0u32 {
            if p < stride {
                let sq = s_q2[p as usize] + s_q2[(p + stride) as usize];
                let sk = s_k2[p as usize] + s_k2[(p + stride) as usize];
                s_q2[p as usize] = sq;
                s_k2[p as usize] = sk;
            }
            sync_cube();
            stride /= 2u32;
        }

        if active {
            let inv_d = 1.0f32 / f32::cast_from(head_dim);
            let scale_q = 1.0f32 / f32::sqrt(s_q2[0] * inv_d + eps);
            let scale_k = 1.0f32 / f32::sqrt(s_k2[0] * inv_d + eps);
            q0 = q0 * scale_q * f32::cast_from(q_norm_w[d0 as usize]);
            q1 = q1 * scale_q * f32::cast_from(q_norm_w[d1 as usize]);
            k0 = k0 * scale_k * f32::cast_from(k_norm_w[d0 as usize]);
            k1 = k1 * scale_k * f32::cast_from(k_norm_w[d1 as usize]);
            let h_heads = num_heads / 2u32;
            if h < h_heads {
                let out_base = (((b * w_spatial + x) * h_heads + h) * h_spatial + y) * head_dim;
                let rope = y * half_d + p;
                let c = f32::cast_from(cos_h[rope as usize]);
                let s = f32::cast_from(sin_h[rope as usize]);
                q_h[(out_base + d0) as usize] = F::cast_from(q0 * c - q1 * s);
                q_h[(out_base + d1) as usize] = F::cast_from(q1 * c + q0 * s);
                k_h[(out_base + d0) as usize] = F::cast_from(k0 * c - k1 * s);
                k_h[(out_base + d1) as usize] = F::cast_from(k1 * c + k0 * s);
                v_h[(out_base + d0) as usize] = F::cast_from(v0);
                v_h[(out_base + d1) as usize] = F::cast_from(v1);
            } else {
                let w_heads = num_heads - h_heads;
                let wh = h - h_heads;
                let out_base = (((b * h_spatial + y) * w_heads + wh) * w_spatial + x) * head_dim;
                let rope = x * half_d + p;
                let c = f32::cast_from(cos_w[rope as usize]);
                let s = f32::cast_from(sin_w[rope as usize]);
                q_w[(out_base + d0) as usize] = F::cast_from(q0 * c - q1 * s);
                q_w[(out_base + d1) as usize] = F::cast_from(q1 * c + q0 * s);
                k_w[(out_base + d0) as usize] = F::cast_from(k0 * c - k1 * s);
                k_w[(out_base + d1) as usize] = F::cast_from(k1 * c + k0 * s);
                v_w[(out_base + d0) as usize] = F::cast_from(v0);
                v_w[(out_base + d1) as usize] = F::cast_from(v1);
            }
        }
    }
}

/// Fuses `CSwinAttention2D` strip un-permutation (`attn_h: [B*W, h_heads, H, D]` and
/// `attn_w: [B*H, w_heads, W, D]`) and `HeadwiseAttentionGate` (`gate_logits: [B, H, W, num_heads]`)
/// into `[B, H, W, num_heads * D]`.
#[cube(launch)]
pub fn merge_cswin2d_gate_kernel<F: CkFloat>(
    attn_h: &[F],
    attn_w: &[F],
    gate_logits: &[F],
    output: &mut [F],
    total_elems: u32,
    h_spatial: u32,
    w_spatial: u32,
    num_heads: u32,
    head_dim: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let d = idx % head_dim;
        let bnh = idx / head_dim;
        let h = bnh % num_heads;
        let btok = bnh / num_heads;
        let n_tokens = h_spatial * w_spatial;
        let tok = btok % n_tokens;
        let b = btok / n_tokens;
        let y = tok / w_spatial;
        let x = tok % w_spatial;

        let h_heads = num_heads / 2u32;
        let val = if h < h_heads {
            let src = (((b * w_spatial + x) * h_heads + h) * h_spatial + y) * head_dim + d;
            attn_h[src as usize]
        } else {
            let w_heads = num_heads - h_heads;
            let hw = h - h_heads;
            let src = (((b * h_spatial + y) * w_heads + hw) * w_spatial + x) * head_dim + d;
            attn_w[src as usize]
        };

        let gate_idx = (btok * num_heads + h) as usize;
        output[idx as usize] = val * sigmoid::<F>(gate_logits[gate_idx]);
    }
}

/// Pads `[B, H, W, C]` with zeros to `[B, Hp, Wp, C]` and rolls by `(-shift_h, -shift_w)` in one pass.
#[cube(launch)]
pub fn pad_roll_2d_kernel<F: CkFloat>(
    input: &[F],
    output: &mut [F],
    total_elems: u32,
    h: u32,
    w: u32,
    hp: u32,
    wp: u32,
    c: u32,
    shift_h: u32,
    shift_w: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let ch = idx % c;
        let pix = idx / c;
        let px = pix % wp;
        let py = (pix / wp) % hp;
        let b = pix / (wp * hp);

        let src_y = (py + shift_h) % hp;
        let src_x = (px + shift_w) % wp;
        if src_y < h && src_x < w {
            let src_idx = (((b * h + src_y) * w + src_x) * c + ch) as usize;
            output[idx as usize] = input[src_idx];
        } else {
            output[idx as usize] = F::new(0.0f32);
        }
    }
}

/// Partitions `[B, Hp, Wp, C]` into `[B * (Hp/w) * (Wp/w), w * w, C]`.
#[cube(launch)]
pub fn window_partition_2d_kernel<F: CkFloat>(
    input: &[F],
    output: &mut [F],
    total_elems: u32,
    hp: u32,
    wp: u32,
    win_size: u32,
    c: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let ch = idx % c;
        let rem = idx / c;
        let lx = rem % win_size;
        let rem2 = rem / win_size;
        let ly = rem2 % win_size;
        let win_idx = rem2 / win_size;

        let nw = wp / win_size;
        let nh = hp / win_size;
        let wx = win_idx % nw;
        let wy = (win_idx / nw) % nh;
        let b = win_idx / (nw * nh);

        let py = wy * win_size + ly;
        let px = wx * win_size + lx;
        let src_idx = (((b * hp + py) * wp + px) * c + ch) as usize;
        output[idx as usize] = input[src_idx];
    }
}

/// Unpartitions `[B * (Hp/w) * (Wp/w), w * w, C]`, rolls by `(+shift_h, +shift_w)`, crops to
/// `[B, H, W, C]`, and adds `shortcut` (`[B, H, W, C]`) in a single pass!
#[cube(launch)]
pub fn window_unpartition_unroll_residual_2d_kernel<F: CkFloat>(
    windows: &[F],
    shortcut: &[F],
    output: &mut [F],
    total_elems: u32,
    h: u32,
    w: u32,
    hp: u32,
    wp: u32,
    win_size: u32,
    c: u32,
    shift_h: u32,
    shift_w: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let ch = idx % c;
        let pix = idx / c;
        let x = pix % w;
        let y = (pix / w) % h;
        let b = pix / (w * h);

        let py = (y + hp - shift_h) % hp;
        let px = (x + wp - shift_w) % wp;
        let wy = py / win_size;
        let ly = py % win_size;
        let wx = px / win_size;
        let lx = px % win_size;

        let nw = wp / win_size;
        let nh = hp / win_size;
        let win_idx = (b * nh + wy) * nw + wx;
        let tok_idx = ly * win_size + lx;
        let src_idx = ((win_idx * (win_size * win_size) + tok_idx) * c + ch) as usize;
        output[idx as usize] = shortcut[idx as usize] + windows[src_idx];
    }
}

/// Fills `output` with zeros (`0.0`).
#[cube(launch)]
pub fn zero_buffer_kernel<F: CkFloat>(output: &mut [F], total_elems: u32) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        output[idx as usize] = F::new(0.0f32);
    }
}

/// Gathers `src[win_indices[w], h, token_indices[t], d]` from `src` (`[total_wins, num_heads, n_tokens, head_dim]`)
/// into `dst` (`[n_sel_wins, num_heads, n_sel_tokens, head_dim]`).
#[cube(launch)]
pub fn gather_window_tokens_kernel<F: CkFloat>(
    src: &[F],
    win_indices: &[u32],
    token_indices: &[u32],
    dst: &mut [F],
    total_elems: u32,
    num_heads: u32,
    n_tokens_full: u32,
    n_sel_tokens: u32,
    head_dim: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let d = idx % head_dim;
        let rem = idx / head_dim;
        let t_sel = rem % n_sel_tokens;
        let rem2 = rem / n_sel_tokens;
        let h = rem2 % num_heads;
        let w_sel = rem2 / num_heads;

        let win_full = win_indices[w_sel as usize];
        let tok_full = token_indices[t_sel as usize];
        let src_idx =
            (((win_full * num_heads + h) * n_tokens_full + tok_full) * head_dim + d) as usize;
        dst[idx as usize] = src[src_idx];
    }
}

/// Scatters `src` (`[n_sel_wins, num_heads, n_sel_tokens, head_dim]`) back into
/// `dst[win_indices[w], h, token_indices[t], d]` (`[total_wins, num_heads, n_tokens, head_dim]`).
#[cube(launch)]
pub fn scatter_window_tokens_kernel<F: CkFloat>(
    src: &[F],
    win_indices: &[u32],
    token_indices: &[u32],
    dst: &mut [F],
    total_elems: u32,
    num_heads: u32,
    n_tokens_full: u32,
    n_sel_tokens: u32,
    head_dim: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let d = idx % head_dim;
        let rem = idx / head_dim;
        let t_sel = rem % n_sel_tokens;
        let rem2 = rem / n_sel_tokens;
        let h = rem2 % num_heads;
        let w_sel = rem2 / num_heads;

        let win_full = win_indices[w_sel as usize];
        let tok_full = token_indices[t_sel as usize];
        let dst_idx =
            (((win_full * num_heads + h) * n_tokens_full + tok_full) * head_dim + d) as usize;
        dst[dst_idx] = src[idx as usize];
    }
}
