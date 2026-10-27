//! CubeCL kernels for 3D spatiotemporal attention blocks (`SwinAttention3D`, `CSwinAttention3D`)
//! and per-channel scaled skip connections (`corridorkey/blocks.py`, `corridorkey/stage1.py`).

use cubecl::prelude::*;

use corridorkey_base::core::CkFloat;
use corridorkey_core::compute::row_position;

use super::attn::sigmoid;

/// Fused QKV bias + head-group split + QK-RMSNorm + 3D axial RoPE for `CSwinAttention3D`.
///
/// Splits `num_heads` evenly on `[B, T, H, W, 3 * dim]`:
/// - `0..h_heads` (V-stripes along `T * H`): written to `q_h, k_h, v_h` of shape
///   `[B * W, h_heads, T * H, head_dim]` with `(cos_h, sin_h)` (`[T * H, head_dim / 2]`).
/// - `h_heads..num_heads` (H-stripes along `T * W`): written to `q_w, k_w, v_w` of shape
///   `[B * H, w_heads, T * W, head_dim]` with `(cos_w, sin_w)` (`[T * W, head_dim / 2]`).
/// Rows are `(b, t, y, x, head)`, head fastest.
///
/// One cube per `(token, head)` vector; unit `p` owns the pair `(2p, 2p + 1)` of q, k and v
/// (coalesced reads and writes), q / k sums of squares reduced in f32 through shared memory.
/// `block_size` is a power of two `>= head_dim / 2`.
#[cube(launch)]
pub fn prep_cswin3d_qkv_kernel<F: CkFloat>(
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
    t_grid: u32,
    h_grid: u32,
    w_grid: u32,
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
        let h = idx % num_heads;
        let bthw = idx / num_heads;
        let x = bthw % w_grid;
        let bth = bthw / w_grid;
        let y = bth % h_grid;
        let bt = bth / h_grid;
        let t = bt % t_grid;
        let b = bt / t_grid;
        let tok_base = bthw * 3u32 * num_heads * head_dim;
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
                let out_base =
                    ((((b * w_grid + x) * h_heads + h) * t_grid + t) * h_grid + y) * head_dim;
                let rope = (t * h_grid + y) * half_d + p;
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
                let out_base =
                    ((((b * h_grid + y) * w_heads + wh) * t_grid + t) * w_grid + x) * head_dim;
                let rope = (t * w_grid + x) * half_d + p;
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

/// Un-strips `attn_h` (`[B * W, h_heads, T * H, head_dim]`) and `attn_w`
/// (`[B * H, w_heads, T * W, head_dim]`), multiplies by `sigmoid(gate_logits)`
/// (`[B, T, H, W, num_heads]`), and writes `[B, T, H, W, dim]`.
#[cube(launch)]
pub fn merge_cswin3d_gate_kernel<F: CkFloat>(
    attn_h: &[F],
    attn_w: &[F],
    gate_logits: &[F],
    output: &mut [F],
    total_elems: u32,
    t_grid: u32,
    h_grid: u32,
    w_grid: u32,
    num_heads: u32,
    head_dim: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let dim = num_heads * head_dim;
        let c = idx % dim;
        let bthw = idx / dim;
        let x = bthw % w_grid;
        let bth = bthw / w_grid;
        let y = bth % h_grid;
        let bt = bth / h_grid;
        let t = bt % t_grid;
        let b = bt / t_grid;

        let h = c / head_dim;
        let d = c % head_dim;
        let h_heads = num_heads / 2u32;
        let w_heads = num_heads - h_heads;
        let gate = sigmoid::<F>(gate_logits[(bthw * num_heads + h) as usize]);

        let val = if h < h_heads {
            let src = ((((b * w_grid + x) * h_heads + h) * t_grid + t) * h_grid + y) * head_dim + d;
            attn_h[src as usize]
        } else {
            let wh = h - h_heads;
            let src =
                ((((b * h_grid + y) * w_heads + wh) * t_grid + t) * w_grid + x) * head_dim + d;
            attn_w[src as usize]
        };
        output[idx as usize] = val * gate;
    }
}

/// Partitions `[B, T, H, W, Cg]` into `[B * (Hp / win) * (Wp / win), T * win * win, Cg]`,
/// zero-padding spatial coordinates outside `(H, W)`.
#[cube(launch)]
pub fn window_partition_3d_kernel<F: CkFloat>(
    input: &[F],
    windows: &mut [F],
    total_elems: u32,
    t_grid: u32,
    h: u32,
    w: u32,
    hp: u32,
    wp: u32,
    win_size: u32,
    channels: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let c = idx % channels;
        let rem0 = idx / channels;
        let lx = rem0 % win_size;
        let rem1 = rem0 / win_size;
        let ly = rem1 % win_size;
        let rem2 = rem1 / win_size;
        let t = rem2 % t_grid;
        let win_idx = rem2 / t_grid;

        let nw_w = wp / win_size;
        let nh_w = hp / win_size;
        let wx = win_idx % nw_w;
        let rem3 = win_idx / nw_w;
        let wy = rem3 % nh_w;
        let b = rem3 / nh_w;

        let y = wy * win_size + ly;
        let x = wx * win_size + lx;
        let v = if y < h && x < w {
            let src = (((b * t_grid + t) * h + y) * w + x) * channels + c;
            input[src as usize]
        } else {
            F::new(0.0f32)
        };
        windows[idx as usize] = v;
    }
}

/// Unpartitions `[B * (Hp / win) * (Wp / win), T * win * win, C]`, reverses spatial cyclic shift
/// `(shift_h, shift_w)`, crops to `[B, T, H, W, C]`, and (when `add_shortcut != 0`) adds
/// `shortcut`.
#[cube(launch)]
pub fn window_unpartition_3d_kernel<F: CkFloat>(
    windows: &[F],
    shortcut: &[F],
    output: &mut [F],
    total_elems: u32,
    t_grid: u32,
    h: u32,
    w: u32,
    hp: u32,
    wp: u32,
    win_size: u32,
    channels: u32,
    shift_h: u32,
    shift_w: u32,
    add_shortcut: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let c = idx % channels;
        let rem0 = idx / channels;
        let x = rem0 % w;
        let rem1 = rem0 / w;
        let y = rem1 % h;
        let rem2 = rem1 / h;
        let t = rem2 % t_grid;
        let b = rem2 / t_grid;

        let py = (y + hp - shift_h) % hp;
        let px = (x + wp - shift_w) % wp;
        let wy = py / win_size;
        let ly = py % win_size;
        let wx = px / win_size;
        let lx = px % win_size;

        let nw_w = wp / win_size;
        let nh_w = hp / win_size;
        let win_idx = (b * nh_w + wy) * nw_w + wx;
        let win_tok = (t * win_size + ly) * win_size + lx;
        let n_win_toks = t_grid * win_size * win_size;
        let src = (win_idx * n_win_toks + win_tok) * channels + c;
        let i = idx as usize;
        let v = windows[src as usize];
        output[i] = if add_shortcut != 0u32 {
            shortcut[i] + v
        } else {
            v
        };
    }
}

/// Computes `out[i] = a[i] + b[i] * scale[i % dim]` for per-channel U-Net / stem skip connections
/// in `Stage1`.
#[cube(launch)]
pub fn add_scaled_skip_kernel<F: CkFloat>(
    a: &[F],
    b: &[F],
    scale: &[F],
    output: &mut [F],
    total_elems: u32,
    dim: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let c = (idx % dim) as usize;
        let i = idx as usize;
        output[i] = a[i] + b[i] * scale[c];
    }
}
