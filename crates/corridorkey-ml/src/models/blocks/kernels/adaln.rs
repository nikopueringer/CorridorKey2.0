//! CubeCL kernels for `AdaLNZero` conditioning (`corridorkey/blocks.py`).
//!
//! Fuses `SiLU(in_proj(cond) + bias)` and per-frame `(shift, scale, gate)` modulation/residual
//! directly from the packed `[BT, 6 * dim]` parameter buffer without intermediate tensor splits
//! or broadcast copies.

use cubecl::prelude::*;

use corridorkey_base::core::CkFloat;

/// Applies `out[i] = silu(input[i] + bias[i % dim])` where `silu(v) = v / (1 + exp(-v))`.
#[cube(launch)]
pub fn silu_bias_kernel<F: CkFloat>(
    input: &[F],
    bias: &[F],
    output: &mut [F],
    total_elems: u32,
    dim: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let c = (idx % dim) as usize;
        let i = idx as usize;
        let v = input[i] + bias[c];
        let sig = F::new(1.0f32) / (F::new(1.0f32) + F::exp(-v));
        output[i] = v * sig;
    }
}

/// Applies AdaLN modulation on `[BT, H * W, dim]`:
/// `out[bt, hw, c] = x[bt, hw, c] * (1 + scale[bt, c]) + shift[bt, c]`,
/// reading `shift` at `params[bt * params_stride + pair_offset + c]` and `scale` at
/// `params[bt * params_stride + pair_offset + dim + c]` (plus `proj_bias`).
#[cube(launch)]
pub fn adaln_modulate_kernel<F: CkFloat>(
    x: &[F],
    params: &[F],
    proj_bias: &[F],
    output: &mut [F],
    total_elems: u32,
    hw: u32,
    dim: u32,
    params_stride: u32,
    pair_offset: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let c = idx % dim;
        let bt = idx / (hw * dim);
        let p_base = bt * params_stride + pair_offset;
        let shift = params[(p_base + c) as usize] + proj_bias[(pair_offset + c) as usize];
        let scale =
            params[(p_base + dim + c) as usize] + proj_bias[(pair_offset + dim + c) as usize];
        let i = idx as usize;
        output[i] = x[i] * (F::new(1.0f32) + scale) + shift;
    }
}

/// Applies AdaLN gated residual connection on `[BT, H * W, dim]`:
/// `out[bt, hw, c] = shortcut[bt, hw, c] + branch[bt, hw, c] * gate[bt, c]`,
/// reading `gate` at `params[bt * params_stride + pair_offset + 2 * dim + c]` (plus `proj_bias`).
#[cube(launch)]
pub fn adaln_gate_residual_kernel<F: CkFloat>(
    shortcut: &[F],
    branch: &[F],
    params: &[F],
    proj_bias: &[F],
    output: &mut [F],
    total_elems: u32,
    hw: u32,
    dim: u32,
    params_stride: u32,
    pair_offset: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let c = idx % dim;
        let bt = idx / (hw * dim);
        let gate_off = pair_offset + 2u32 * dim + c;
        let gate = params[(bt * params_stride + gate_off) as usize] + proj_bias[gate_off as usize];
        let i = idx as usize;
        output[i] = shortcut[i] + branch[i] * gate;
    }
}
