//! Frame preparation kernels (`video_io.preprocess_video_frame_views`,
//! `runtime._build_foundation_inputs`, the RVM resize of `precompute_rvm_to_memmap`).
//!
//! Decoded frames arrive as packed bytes (WGSL has no 8-bit type: four bytes per `u32` word) and
//! leave as the model input and the backbones' view at the padded model size; the backbone inputs
//! are resized from that view.

use cubecl::prelude::*;

use crate::color::kernels::{linear_to_asinh, linear_to_srgb, srgb_to_linear};
use crate::core::CkFloat;

/// Byte `i` of a buffer packed four bytes per little-endian `u32` word.
#[cube]
fn byte_at(words: &[u32], i: u32) -> u32 {
    (words[(i / 4u32) as usize] >> ((i % 4u32) * 8u32)) & 0xFFu32
}

/// cv2's `BORDER_REFLECT_101` source index of padded coordinate `p` for an image of `size`
/// starting at `start` (`gfedcb|abcdefgh|gfedcba`); the pad must be smaller than the image.
#[cube]
pub fn reflect_101(p: u32, start: u32, size: u32) -> u32 {
    // Signed arithmetic in i32: the padded coordinate may sit before the image.
    let s = p as i32 - start as i32;
    let n = size as i32;
    let r = if s < 0i32 { -s } else if s >= n { 2i32 * (n - 1i32) - s } else { s };
    r as u32
}

/// Model input and backbone view of `[T, H, W, 3]` padded frames from `[T, h0, w0, 3]` 8-bit RGB:
/// `display = byte / 255` at the reflect-101 source pixel, `linear = adjust(srgb_to_linear(display))`
/// (`scale · x + offset`), `model = asinh(linear)`, and `foundation = clip(linear_to_srgb(max(linear,
/// 0)), 0, 1)` when `adjusted` (`linear_to_vfm_sdr`), else `display`.
#[cube(launch)]
pub fn frame_views_kernel<F: CkFloat, N: Size>(
    frames: &[u32],
    model: &mut [Vector<F, N>],
    foundation: &mut [Vector<F, N>],
    total: u32,
    h: u32,
    w: u32,
    h0: u32,
    w0: u32,
    top: u32,
    left: u32,
    scale: F,
    offset: F,
    inv_knee: F,
    inv_denom: F,
    #[comptime] adjusted: bool,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let c = i % 3u32;
        let x = (i / 3u32) % w;
        let y = (i / (3u32 * w)) % h;
        let t = i / (3u32 * w * h);
        let sy = reflect_101(y, top, h0);
        let sx = reflect_101(x, left, w0);
        let byte = byte_at(frames, ((t * h0 + sy) * w0 + sx) * 3u32 + c);
        let display = F::cast_from(byte) * F::new(1.0f32 / 255.0f32);
        let v = Vector::<F, N>::new(display);
        let linear = srgb_to_linear::<F, N>(v) * Vector::new(scale) + Vector::new(offset);
        model[i as usize] = linear_to_asinh::<F, N>(linear, inv_knee, inv_denom);
        if comptime!(adjusted) {
            let zero = Vector::<F, N>::new(F::new(0.0f32));
            let one = Vector::<F, N>::new(F::new(1.0f32));
            let sdr = linear_to_srgb::<F, N>(select_many(linear.less_than(&zero), zero, linear));
            foundation[i as usize] = select_many(sdr.greater_than(&one), one, select_many(sdr.less_than(&zero), zero, sdr));
        } else {
            foundation[i as usize] = v;
        }
    }
}

/// [`frame_views_kernel`] from `[T, h0, w0, 3]` float RGB (EXR frame directories,
/// `video_io._read_frame_dir_views`). With `linear_input` the values are scene-linear: `linear =
/// adjust(raw)` and the backbones see `clip(linear_to_srgb(max(linear, 0)), 0, 1)`. Otherwise they
/// are display values: `display = clip(raw, 0, 1)`, then as from bytes.
#[cube(launch)]
pub fn frame_views_f32_kernel<F: CkFloat, N: Size>(
    frames: &[f32],
    model: &mut [Vector<F, N>],
    foundation: &mut [Vector<F, N>],
    total: u32,
    h: u32,
    w: u32,
    h0: u32,
    w0: u32,
    top: u32,
    left: u32,
    scale: F,
    offset: F,
    inv_knee: F,
    inv_denom: F,
    #[comptime] linear_input: bool,
    #[comptime] adjusted: bool,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let c = i % 3u32;
        let x = (i / 3u32) % w;
        let y = (i / (3u32 * w)) % h;
        let t = i / (3u32 * w * h);
        let sy = reflect_101(y, top, h0);
        let sx = reflect_101(x, left, w0);
        let raw = frames[(((t * h0 + sy) * w0 + sx) * 3u32 + c) as usize];
        let zero = Vector::<F, N>::new(F::new(0.0f32));
        let one = Vector::<F, N>::new(F::new(1.0f32));
        if comptime!(linear_input) {
            let linear = Vector::<F, N>::new(F::cast_from(raw)) * Vector::new(scale) + Vector::new(offset);
            model[i as usize] = linear_to_asinh::<F, N>(linear, inv_knee, inv_denom);
            let sdr = linear_to_srgb::<F, N>(select_many(linear.less_than(&zero), zero, linear));
            foundation[i as usize] = select_many(sdr.greater_than(&one), one, select_many(sdr.less_than(&zero), zero, sdr));
        } else {
            let display = Vector::<F, N>::new(F::cast_from(f32::min(f32::max(raw, 0.0f32), 1.0f32)));
            let linear = srgb_to_linear::<F, N>(display) * Vector::new(scale) + Vector::new(offset);
            model[i as usize] = linear_to_asinh::<F, N>(linear, inv_knee, inv_denom);
            if comptime!(adjusted) {
                let sdr = linear_to_srgb::<F, N>(select_many(linear.less_than(&zero), zero, linear));
                foundation[i as usize] = select_many(sdr.greater_than(&one), one, select_many(sdr.less_than(&zero), zero, sdr));
            } else {
                foundation[i as usize] = display;
            }
        }
    }
}

/// PyTorch's `interpolate(mode="area")` (`adaptive_avg_pool2d`) of `[T, h, w, C]` to
/// `[T, oh, ow, C]`: output `(oy, ox)` averages input rows `floor(oy·h/oh) .. ceil((oy+1)·h/oh)` and
/// the same for columns, in f32.
#[cube(launch)]
pub fn area_resize_kernel<F: CkFloat>(
    input: &[F],
    output: &mut [F],
    total: u32,
    h: u32,
    w: u32,
    ch: u32,
    oh: u32,
    ow: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let c = i % ch;
        let ox = (i / ch) % ow;
        let oy = (i / (ch * ow)) % oh;
        let t = i / (ch * ow * oh);
        let y0 = (oy * h) / oh;
        let y1 = ((oy + 1u32) * h + oh - 1u32) / oh;
        let x0 = (ox * w) / ow;
        let x1 = ((ox + 1u32) * w + ow - 1u32) / ow;
        let mut sum = 0.0f32;
        for y in y0..y1 {
            for x in x0..x1 {
                sum += f32::cast_from(input[(((t * h + y) * w + x) * ch + c) as usize]);
            }
        }
        output[i as usize] = F::cast_from(sum / f32::cast_from((y1 - y0) * (x1 - x0)));
    }
}

/// PyTorch's `interpolate(mode="bilinear", align_corners=False)` without antialias of `[T, h, w, C]`
/// to `[T, oh, ow, C]` (source index `(o + 0.5)·in/out - 0.5`, clamped at 0), clamped to [0, 1]
/// when `clamp`.
#[cube(launch)]
pub fn bilinear_resize_kernel<F: CkFloat>(
    input: &[F],
    output: &mut [F],
    total: u32,
    h: u32,
    w: u32,
    ch: u32,
    oh: u32,
    ow: u32,
    #[comptime] clamp: bool,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let c = i % ch;
        let ox = (i / ch) % ow;
        let oy = (i / (ch * ow)) % oh;
        let t = i / (ch * ow * oh);
        let sy = f32::max((f32::cast_from(oy) + 0.5f32) * (f32::cast_from(h) / f32::cast_from(oh)) - 0.5f32, 0.0f32);
        let sx = f32::max((f32::cast_from(ox) + 0.5f32) * (f32::cast_from(w) / f32::cast_from(ow)) - 0.5f32, 0.0f32);
        let y0 = u32::min(u32::cast_from(f32::floor(sy)), h - 1u32);
        let x0 = u32::min(u32::cast_from(f32::floor(sx)), w - 1u32);
        let y1 = u32::min(y0 + 1u32, h - 1u32);
        let x1 = u32::min(x0 + 1u32, w - 1u32);
        let ly = sy - f32::cast_from(y0);
        let lx = sx - f32::cast_from(x0);
        let row0 = (t * h + y0) * w;
        let row1 = (t * h + y1) * w;
        let p00 = f32::cast_from(input[((row0 + x0) * ch + c) as usize]);
        let p01 = f32::cast_from(input[((row0 + x1) * ch + c) as usize]);
        let p10 = f32::cast_from(input[((row1 + x0) * ch + c) as usize]);
        let p11 = f32::cast_from(input[((row1 + x1) * ch + c) as usize]);
        let top = p00 * (1.0f32 - lx) + p01 * lx;
        let bottom = p10 * (1.0f32 - lx) + p11 * lx;
        let v = top * (1.0f32 - ly) + bottom * ly;
        if comptime!(clamp) {
            output[i as usize] = F::cast_from(f32::min(f32::max(v, 0.0f32), 1.0f32));
        } else {
            output[i as usize] = F::cast_from(v);
        }
    }
}
