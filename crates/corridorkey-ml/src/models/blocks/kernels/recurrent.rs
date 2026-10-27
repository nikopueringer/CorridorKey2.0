//! CubeCL kernels for RVM's gated recurrent core (`_GatedTransformerCore` in `corridorkey/rvm.py`).

use cubecl::prelude::*;

use corridorkey_base::core::CkFloat;

#[cube]
fn sigmoid_f32(x: f32) -> f32 {
    1.0f32 / (1.0f32 + f32::exp(-x))
}

/// GRU gates of one frame: with `gx = x · [W_iu | W_ir]` and `gs = s · [W_su | W_sr]` (`[rows, 2·dim]`)
/// and `ln_s = state_layer_norm(s)`: `update = σ(gx_u + gs_u)`,
/// `gated = σ(gx_r + gs_r) · ln_s` (the key / value sequence of the cross-attention).
#[cube(launch)]
pub fn gru_gates_kernel<F: CkFloat>(
    gx: &[F],
    gs: &[F],
    ln_s: &[F],
    update: &mut [F],
    gated: &mut [F],
    total: u32,
    dim: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let r = i / dim;
        let c = i % dim;
        let u_idx = (r * 2u32 * dim + c) as usize;
        let r_idx = (r * 2u32 * dim + dim + c) as usize;
        let u = sigmoid_f32(f32::cast_from(gx[u_idx]) + f32::cast_from(gs[u_idx]));
        let reset = sigmoid_f32(f32::cast_from(gx[r_idx]) + f32::cast_from(gs[r_idx]));
        update[i as usize] = F::cast_from(u);
        gated[i as usize] = F::cast_from(reset * f32::cast_from(ln_s[i as usize]));
    }
}

/// `s' = (1 - u) · s + u · h` over `[tokens, dim]` (token 0 = CLS), written as the new state and,
/// for the patch tokens, as the normalised feature of the frame:
/// `features[feature_offset + (t - 1) · dim + c] = (s' - mean[c]) · inv_std[c]`.
#[cube(launch)]
pub fn state_update_kernel<F: CkFloat>(
    state: &[F],
    update: &[F],
    h: &[F],
    mean: &[F],
    inv_std: &[F],
    new_state: &mut [F],
    features: &mut [F],
    total: u32,
    dim: u32,
    feature_offset: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let t = i / dim;
        let c = i % dim;
        let u = f32::cast_from(update[i as usize]);
        let s = (1.0f32 - u) * f32::cast_from(state[i as usize]) + u * f32::cast_from(h[i as usize]);
        new_state[i as usize] = F::cast_from(s);
        if t > 0u32 {
            let f = (s - f32::cast_from(mean[c as usize])) * f32::cast_from(inv_std[c as usize]);
            features[(feature_offset + (t - 1u32) * dim + c) as usize] = F::cast_from(f);
        }
    }
}

/// `dst[i] = src[offset + i]` for `i < len` (one frame out of a batch).
#[cube(launch)]
pub fn copy_range_kernel<F: CkFloat>(src: &[F], dst: &mut [F], len: u32, offset: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < len {
        dst[i as usize] = src[(offset + i) as usize];
    }
}
