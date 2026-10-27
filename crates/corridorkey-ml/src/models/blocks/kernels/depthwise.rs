//! Coalesced NHWC depthwise spatial convolution kernels (`QKVDepthwiseSpatial` and
//! `DepthwiseSwiGLU2D` in `corridorkey/blocks.py`).
//!
//! Because tokens sit in contiguous `[B, H, W, C]` order, storing depthwise filter weights in
//! tap-major `[K * K, C]` order makes both the neighbour pixel reads `input[b, ry, rx, c]` and the
//! filter reads `weight[tap, c]` contiguous across adjacent threads `c` in a warp — with inline
//! reflect/replicate boundary indexing and zero padding copies or `NHWC ↔ NCHW` transposes.

use cubecl::prelude::*;

use corridorkey_base::core::CkFloat;

use super::swiglu::silu;

/// Boundary coordinate for `coord = pos + tap - pad` along an axis of length `len`:
/// - `reflect == 1` (`len > pad`): mirrors once across `0` or `len - 1`.
/// - `reflect == 0` (`len <= pad`): clamps to `0..len - 1` (`replicate`).
#[cube]
pub fn boundary_coord(pos: u32, tap: u32, pad: u32, len: u32, #[comptime] reflect: u32) -> u32 {
    let raw = (pos + tap) as i32 - pad as i32;
    let max_idx = (len - 1) as i32;
    let mapped = if reflect == 1u32 {
        let low = select(raw < 0i32, -raw, raw);
        select(low > max_idx, 2i32 * max_idx - low, low)
    } else {
        let low = select(raw < 0i32, 0i32, raw);
        select(low > max_idx, max_idx, low)
    };
    mapped as u32
}

/// NHWC depthwise 2D convolution (`QKVDepthwiseSpatial`) with optional fused input bias:
/// `output[b, y, x, c] = bias[c] + sum_{ky, kx} (input[b, ry, rx, c] + in_bias[c]) * weight[(ky * K + kx) * C + c]`.
#[cube(launch)]
pub fn depthwise_conv2d_nhwc_kernel<F: CkFloat>(
    input: &[F],
    in_bias: &[F],
    weight_tap_major: &[F],
    bias: &[F],
    output: &mut [F],
    total: u32,
    height: u32,
    width: u32,
    channels: u32,
    apply_in_bias: u32,
    #[comptime] kernel_size: u32,
    #[comptime] reflect: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total {
        let c = idx % channels;
        let pixel = idx / channels;
        let x = pixel % width;
        let bh = pixel / width;
        let y = bh % height;
        let b = bh / height;

        let pad = kernel_size / 2u32;
        let in_b = if apply_in_bias != 0u32 {
            in_bias[c as usize]
        } else {
            F::new(0.0f32)
        };
        let mut acc = bias[c as usize];
        let mut ky = 0u32;
        while ky < kernel_size {
            let ry = boundary_coord(y, ky, pad, height, reflect);
            let row_base = (b * height + ry) * width;
            let mut kx = 0u32;
            while kx < kernel_size {
                let rx = boundary_coord(x, kx, pad, width, reflect);
                let tap = ky * kernel_size + kx;
                let in_idx = ((row_base + rx) * channels + c) as usize;
                let w_idx = (tap * channels + c) as usize;
                acc += (input[in_idx] + in_b) * weight_tap_major[w_idx];
                kx += 1u32;
            }
            ky += 1u32;
        }
        output[idx as usize] = acc;
    }
}

/// Fused NHWC depthwise SwiGLU residual (`DepthwiseSwiGLU2D`):
/// evaluates both the `value` and `gate` depthwise `K × K` filters from the same reflected
/// coordinates of `normed`, multiplies `val * silu(gate) * gamma[c]`, and adds into `residual`:
/// `output[i] = residual[i] + (val * silu(gate)) * gamma[c]`.
#[cube(launch)]
pub fn depthwise_swiglu_2d_kernel<F: CkFloat>(
    normed: &[F],
    residual: &[F],
    value_w_tap_major: &[F],
    value_b: &[F],
    gate_w_tap_major: &[F],
    gate_b: &[F],
    gamma: &[F],
    output: &mut [F],
    total: u32,
    height: u32,
    width: u32,
    channels: u32,
    #[comptime] kernel_size: u32,
    #[comptime] reflect: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total {
        let c = idx % channels;
        let pixel = idx / channels;
        let x = pixel % width;
        let bh = pixel / width;
        let y = bh % height;
        let b = bh / height;

        let pad = kernel_size / 2u32;
        let mut val_acc = value_b[c as usize];
        let mut gate_acc = gate_b[c as usize];
        let mut ky = 0u32;
        while ky < kernel_size {
            let ry = boundary_coord(y, ky, pad, height, reflect);
            let row_base = (b * height + ry) * width;
            let mut kx = 0u32;
            while kx < kernel_size {
                let rx = boundary_coord(x, kx, pad, width, reflect);
                let tap = ky * kernel_size + kx;
                let in_idx = ((row_base + rx) * channels + c) as usize;
                let w_idx = (tap * channels + c) as usize;
                let sample = normed[in_idx];
                val_acc += sample * value_w_tap_major[w_idx];
                gate_acc += sample * gate_w_tap_major[w_idx];
                kx += 1u32;
            }
            ky += 1u32;
        }
        let mixed = val_acc * silu::<F>(gate_acc) * gamma[c as usize];
        output[idx as usize] = residual[idx as usize] + mixed;
    }
}
