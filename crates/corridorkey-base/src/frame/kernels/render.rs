//! Output rendering kernels (`runtime._render_predictions_torch` and `_render_predictions_exr_torch`):
//! from Stage 2 logits (`[T, H, W, 4]`: alpha, then asinh foreground RGB, at the padded size) to the
//! writers' frames at the native size: 16-bit BGR / BGRA for video, f32 for EXR.
//!
//! 16-bit frames are packed two values per `u32` word (WGSL has no 16-bit type); one unit writes one
//! whole word, so units never share a word. Values convert like `(x * 65535).astype(np.uint16)`
//! (truncation). Everything is computed in f32.

use cubecl::prelude::*;

use crate::core::CkFloat;

/// Outputs, as comptime codes.
pub const OUT_ALPHA: u32 = 0;
pub const OUT_FG: u32 = 1;
pub const OUT_CUTOUT: u32 = 2;
pub const OUT_CHECKER: u32 = 3;

/// `make_checker`: 64-pixel squares, 0.75 where `(y / 64 + x / 64)` is even, else 0.5.
pub const CHECKER_SIZE: u32 = 64;

/// Scalar twin of [`crate::color::kernels::linear_to_srgb`] (odd-symmetric).
#[cube]
fn linear_to_srgb_f32(x: f32) -> f32 {
    let a = f32::abs(x);
    let y = if a > 0.0031308f32 { 1.055f32 * f32::powf(a, 1.0f32 / 2.4f32) - 0.055f32 } else { a * 12.92f32 };
    if x < 0.0f32 { -y } else { y }
}

/// Scalar twin of [`crate::color::kernels::srgb_to_linear`] (odd-symmetric).
#[cube]
fn srgb_to_linear_f32(x: f32) -> f32 {
    let a = f32::abs(x);
    let y = if a > 0.04045f32 { f32::powf((a + 0.055f32) / 1.055f32, 2.4f32) } else { a / 12.92f32 };
    if x < 0.0f32 { -y } else { y }
}

/// Scalar twin of [`crate::color::kernels::sinh_accurate`] (backends' `sinh` is not accurate).
#[cube]
fn sinh_f32(x: f32) -> f32 {
    let x2 = x * x;
    if f32::abs(x) < 0.5f32 {
        x * (1.0f32 + x2 * (1.0f32 / 6.0f32 + x2 * (1.0f32 / 120.0f32 + x2 * (1.0f32 / 5040.0f32))))
    } else {
        (f32::exp(x) - f32::exp(-x)) * 0.5f32
    }
}

#[cube]
fn clamp01(x: f32) -> f32 {
    f32::min(f32::max(x, 0.0f32), 1.0f32)
}

/// Channel `c` (0 = R, 1 = G, 2 = B) of `_despill_average_srgb_torch(rgb, strength)`: the green
/// excess over the red-blue mean moves half to red, half to blue.
#[cube]
fn despill(r: f32, g: f32, b: f32, c: u32, strength: f32) -> f32 {
    let spill = f32::max(g - (r + b) * 0.5f32, 0.0f32);
    let orig = pick(r, g, b, c);
    let full = if c == 1u32 { g - spill } else { orig + spill * 0.5f32 };
    if strength <= 0.0f32 {
        orig
    } else if strength >= 1.0f32 {
        full
    } else {
        orig * (1.0f32 - strength) + full * strength
    }
}

/// `(r, g, b)` component `c`.
#[cube]
fn pick(r: f32, g: f32, b: f32, c: u32) -> f32 {
    if c == 0u32 {
        r
    } else if c == 1u32 {
        g
    } else {
        b
    }
}

/// Value of output channel `ch` (BGR / BGRA order; alpha outputs repeat alpha) at native pixel
/// `(t, y, x)`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn render_value<F: CkFloat>(
    logits: &[F],
    t: u32,
    y: u32,
    x: u32,
    ch: u32,
    h: u32,
    w: u32,
    top: u32,
    left: u32,
    inv_scale: f32,
    inv_offset: f32,
    knee: f32,
    denom: f32,
    strength: f32,
    #[comptime] output: u32,
    #[comptime] exr: bool,
    #[comptime] cutout_linear: bool,
) -> f32 {
    let base = ((t * h + y + top) * w + x + left) * 4u32;
    let a = clamp01(f32::cast_from(logits[base as usize]));
    // asinh -> scene-linear -> inverse linear adjustment
    let lr = sinh_f32(f32::cast_from(logits[(base + 1u32) as usize]) * denom) * knee * inv_scale + inv_offset;
    let lg = sinh_f32(f32::cast_from(logits[(base + 2u32) as usize]) * denom) * knee * inv_scale + inv_offset;
    let lb = sinh_f32(f32::cast_from(logits[(base + 3u32) as usize]) * denom) * knee * inv_scale + inv_offset;
    let c = 2u32 - u32::min(ch, 2u32); // BGR -> RGB component (unused for the alpha channel)
    let checker = if ((y / CHECKER_SIZE) + (x / CHECKER_SIZE)) % 2u32 == 0u32 { 0.75f32 } else { 0.5f32 };
    let bg = srgb_to_linear_f32(checker);

    let mut v = a;
    if comptime!(exr) {
        if comptime!(output == OUT_FG) {
            v = pick(lr, lg, lb, c);
        } else if comptime!(output == OUT_CUTOUT) {
            if ch < 3u32 {
                v = despill(lr, lg, lb, c, strength);
            }
        } else if comptime!(output == OUT_CHECKER) {
            v = despill(lr, lg, lb, c, strength) * a + bg * (1.0f32 - a);
        }
    } else {
        let pr = f32::max(lr, 0.0f32);
        let pg = f32::max(lg, 0.0f32);
        let pb = f32::max(lb, 0.0f32);
        let sr = clamp01(linear_to_srgb_f32(pr));
        let sg = clamp01(linear_to_srgb_f32(pg));
        let sb = clamp01(linear_to_srgb_f32(pb));
        if comptime!(output == OUT_FG) {
            v = pick(sr, sg, sb, c);
        } else if comptime!(output == OUT_CUTOUT) {
            if ch < 3u32 {
                if comptime!(cutout_linear) {
                    v = clamp01(despill(pr, pg, pb, c, strength));
                } else {
                    v = clamp01(despill(sr, sg, sb, c, strength));
                }
            }
        } else if comptime!(output == OUT_CHECKER) {
            let comp = srgb_to_linear_f32(despill(sr, sg, sb, c, strength)) * a + bg * (1.0f32 - a);
            v = clamp01(linear_to_srgb_f32(f32::max(comp, 0.0f32)));
        }
    }
    v
}

/// 16-bit video frames `[T, oh, ow, channels]` (`total` values), two per word.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn render_u16_kernel<F: CkFloat>(
    logits: &[F],
    words: &mut [u32],
    total: u32,
    h: u32,
    w: u32,
    oh: u32,
    ow: u32,
    top: u32,
    left: u32,
    inv_scale: f32,
    inv_offset: f32,
    knee: f32,
    denom: f32,
    strength: f32,
    #[comptime] channels: u32,
    #[comptime] output: u32,
    #[comptime] cutout_linear: bool,
) {
    let k = ABSOLUTE_POS as u32;
    if k * 2u32 < total {
        // Starts from a run-time value: a `let mut` from a literal is comptime (dsp-kitchen §6b).
        let mut word = k * 0u32;
        for half in 0..2u32 {
            let i = k * 2u32 + half;
            if i < total {
                let ch = i % channels;
                let pixel = i / channels;
                let x = pixel % ow;
                let y = (pixel / ow) % oh;
                let t = pixel / (ow * oh);
                let v = render_value::<F>(
                    logits, t, y, x, ch, h, w, top, left, inv_scale, inv_offset, knee, denom, strength, output, false, cutout_linear,
                );
                let code = u32::cast_from(f32::floor(v * 65535.0f32));
                word = word | (code << (half * 16u32));
            }
        }
        words[k as usize] = word;
    }
}

/// f32 EXR frames `[T, oh, ow, channels]`, unclamped scene-linear colour, alpha in [0, 1].
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn render_f32_kernel<F: CkFloat>(
    logits: &[F],
    output: &mut [f32],
    total: u32,
    h: u32,
    w: u32,
    oh: u32,
    ow: u32,
    top: u32,
    left: u32,
    inv_scale: f32,
    inv_offset: f32,
    knee: f32,
    denom: f32,
    strength: f32,
    #[comptime] channels: u32,
    #[comptime] kind: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let ch = i % channels;
        let pixel = i / channels;
        let x = pixel % ow;
        let y = (pixel / ow) % oh;
        let t = pixel / (ow * oh);
        output[i as usize] = render_value::<F>(
            logits, t, y, x, ch, h, w, top, left, inv_scale, inv_offset, knee, denom, strength, kind, true, false,
        );
    }
}
