//! Small kernels of the clip pipeline (`runtime.hann_stage1_stream`): Stage 1 input assembly, Hann
//! accumulation of batons, carried hints.

use cubecl::prelude::*;

use corridorkey_base::core::CkFloat;
use corridorkey_base::frame::kernels::prep::reflect_101;

/// `bf16(x)` as f32: round to nearest even on the upper 16 bits (the product stores carried hints
/// and hint qualities as bf16 tensors).
#[cube]
pub fn round_bf16(x: f32) -> f32 {
    let bits = u32::reinterpret(x);
    let rounded = bits + 0x7FFFu32 + ((bits >> 16u32) & 1u32);
    f32::reinterpret(rounded & 0xFFFF_0000u32)
}

/// Stage 1 input `[T, H, W, 4]`: the model view's RGB, then the hint.
#[cube(launch)]
pub fn assemble_rgb_hint_kernel<F: CkFloat>(model: &[F], hint: &[F], output: &mut [F], total: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let c = i % 4u32;
        let p = i / 4u32;
        // In-bounds indices for both arms (an `if` expression may evaluate both).
        let m_idx = if c < 3u32 { p * 3u32 + c } else { 0u32 };
        let rgb = model[m_idx as usize];
        let h = hint[p as usize];
        output[i as usize] = if c < 3u32 { rgb } else { h };
    }
}

/// `acc[i] += weight · x[offset + i]` (Hann accumulation of one frame's baton, in f32).
#[cube(launch)]
pub fn accumulate_kernel<F: CkFloat>(acc: &mut [f32], x: &[F], weight: f32, len: u32, offset: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < len {
        acc[i as usize] += weight * f32::cast_from(x[(offset + i) as usize]);
    }
}

/// `out[out_offset + i] = acc[i] / weight_sum`, rounded through f16 when `round_f16` (the product
/// writes blended batons as float16 before Stage 2).
#[cube(launch)]
pub fn finalize_baton_kernel<F: CkFloat>(
    acc: &[f32],
    out: &mut [F],
    inv_weight: f32,
    len: u32,
    out_offset: u32,
    #[comptime] round_f16: bool,
) {
    let i = ABSOLUTE_POS as u32;
    if i < len {
        let v = acc[i as usize] * inv_weight;
        if comptime!(round_f16) {
            out[(out_offset + i) as usize] = F::cast_from(f32::cast_from(half::f16::cast_from(v)));
        } else {
            out[(out_offset + i) as usize] = F::cast_from(v);
        }
    }
}

/// Carried hint of one frame `[H, W]` from the resized, clamped `aux` (`[T, H, W, 4]`, alpha first),
/// rounded through bf16 when `round_bf16_hint`.
#[cube(launch)]
pub fn carry_hint_kernel<F: CkFloat>(aux: &[F], out: &mut [F], len: u32, frame: u32, #[comptime] round_bf16_hint: bool) {
    let i = ABSOLUTE_POS as u32;
    if i < len {
        let v = f32::cast_from(aux[((frame * len + i) * 4u32) as usize]);
        if comptime!(round_bf16_hint) {
            out[i as usize] = F::cast_from(round_bf16(v));
        } else {
            out[i as usize] = F::cast_from(v);
        }
    }
}

/// Hints `[count, H, W]` from native-size hints `[count, h0, w0]` (values in [0, 1]), reflect-101
/// padded like the frames (`center_pad_to_shape`), written at `dst[dst_offset ..]`; rounded through
/// bf16 when `round_bf16_hint` (the product's hint tensor is bf16).
#[cube(launch)]
pub fn pad_hint_kernel<F: CkFloat>(
    src: &[f32],
    dst: &mut [F],
    total: u32,
    h: u32,
    w: u32,
    h0: u32,
    w0: u32,
    top: u32,
    left: u32,
    dst_offset: u32,
    #[comptime] round_bf16_hint: bool,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let x = i % w;
        let y = (i / w) % h;
        let t = i / (w * h);
        let sy = reflect_101(y, top, h0);
        let sx = reflect_101(x, left, w0);
        let v = src[((t * h0 + sy) * w0 + sx) as usize];
        if comptime!(round_bf16_hint) {
            dst[(dst_offset + i) as usize] = F::cast_from(round_bf16(v));
        } else {
            dst[(dst_offset + i) as usize] = F::cast_from(v);
        }
    }
}

/// `dst[dst_offset + i] = src[src_offset + i]` for `i < len`.
#[cube(launch)]
pub fn copy_kernel<F: CkFloat>(src: &[F], dst: &mut [F], len: u32, src_offset: u32, dst_offset: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < len {
        dst[(dst_offset + i) as usize] = src[(src_offset + i) as usize];
    }
}
