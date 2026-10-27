//! CubeCL kernels for NHWC patchify/depatchify (`im2col_nhwc_kernel`, `col2im_nhwc_kernel`) and
//! Stage 2's gated RGB pixel-skip residual (`add_gated_rgb_skip_kernel`).

use cubecl::prelude::*;

use super::swiglu::silu;

/// Gathers `K × K × C_in` input patches from `input` (`[BT, H_in, W_in, C_in]`) with `stride`
/// and zero-padding `pad` into `cols` (`[BT * H_out * W_out, K * K * C_in]`).
#[cube(launch)]
pub fn im2col_nhwc_kernel<F: Float>(
    input: &[F],
    cols: &mut [F],
    total_elems: u32,
    h_in: u32,
    w_in: u32,
    c_in: u32,
    h_out: u32,
    w_out: u32,
    k: u32,
    stride: u32,
    pad: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < total_elems {
        let ci = idx % c_in;
        let rem0 = idx / c_in;
        let kx = rem0 % k;
        let rem1 = rem0 / k;
        let ky = rem1 % k;
        let tok = rem1 / k;
        let ox = tok % w_out;
        let rem2 = tok / w_out;
        let oy = rem2 % h_out;
        let batch = rem2 / h_out;

        let py = oy * stride + ky;
        let px = ox * stride + kx;
        let val = if py >= pad && px >= pad {
            let iy = py - pad;
            let ix = px - pad;
            if iy < h_in && ix < w_in {
                let src = ((batch * h_in + iy) * w_in + ix) * c_in + ci;
                input[src as usize]
            } else {
                F::new(0.0f32)
            }
        } else {
            F::new(0.0f32)
        };
        cols[idx as usize] = val;
    }
}

/// Accumulates `overlap_mult × overlap_mult` transposed-convolution (or pixel-shuffle when
/// `overlap_mult == 1`) patches from `cols` (`[BT, H_in, W_in, K, K, C_out]`, where
/// `K = overlap_mult * stride`) into `output` (`[BT, H_in * stride, W_in * stride, C_out]`)
/// and adds `bias` (`[C_out]`).
///
/// Because `K = overlap_mult * stride`, for any padded output coordinate `py = oy + pad`,
/// `base_iy = py / stride` and `rem_y = py % stride` uniquely determine the at most
/// `overlap_mult` contributing input tokens `iy = base_iy - m_y` with tap `ky = m_y * stride + rem_y`.
#[cube(launch)]
pub fn col2im_nhwc_kernel<F: Float>(
    cols: &[F],
    bias: &[F],
    output: &mut [F],
    total_elems: u32,
    h_in: u32,
    w_in: u32,
    c_out: u32,
    stride: u32,
    overlap_mult: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < total_elems as usize {
        let idx_u = idx as u32;
        let ch = idx_u % c_out;
        let pixel = idx_u / c_out;
        let w_out = w_in * stride;
        let h_out = h_in * stride;
        let ox = pixel % w_out;
        let oy = (pixel / w_out) % h_out;
        let batch = pixel / (w_out * h_out);

        let k = overlap_mult * stride;
        let pad = (k - stride) / 2u32;
        let py = oy + pad;
        let px = ox + pad;
        let base_iy = py / stride;
        let rem_y = py % stride;
        let base_ix = px / stride;
        let rem_x = px % stride;

        let k2_c = k * k * c_out;
        let mut acc = bias[ch as usize];

        for my in 0..overlap_mult {
            if base_iy >= my {
                let iy = base_iy - my;
                if iy < h_in {
                    let ky = my * stride + rem_y;
                    for mx in 0..overlap_mult {
                        if base_ix >= mx {
                            let ix = base_ix - mx;
                            if ix < w_in {
                                let kx = mx * stride + rem_x;
                                let token_idx = (batch * h_in + iy) * w_in + ix;
                                let tap_idx = (ky * k + kx) * c_out + ch;
                                let col_idx = (token_idx * k2_c + tap_idx) as usize;
                                acc += cols[col_idx];
                            }
                        }
                    }
                }
            }
        }

        output[idx] = acc;
    }
}

/// Fuses Stage 2's late pixel-skip addition:
/// `out[t, c] = x[t, c]` for `c < d_baton`, and
/// `out[t, c] = x[t, c] + patch2[t, c - d_baton] * silu(gate_logits[t, c - d_baton] + gate_bias[c - d_baton])`
/// for `c >= d_baton`.
#[cube(launch)]
pub fn add_gated_rgb_skip_kernel<F: Float>(
    x: &[F],
    patch2: &[F],
    gate_logits: &[F],
    gate_bias: &[F],
    output: &mut [F],
    total_elems: u32,
    d: u32,
    d_baton: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < total_elems as usize {
        let idx_u = idx as u32;
        let c = idx_u % d;
        let base = x[idx];
        if c < d_baton {
            output[idx] = base;
        } else {
            let token = idx_u / d;
            let d_rgb = d - d_baton;
            let c_rgb = c - d_baton;
            let rgb_idx = (token * d_rgb + c_rgb) as usize;
            let g = silu::<F>(gate_logits[rgb_idx] + gate_bias[c_rgb as usize]);
            output[idx] = base + patch2[rgb_idx] * g;
        }
    }
}
