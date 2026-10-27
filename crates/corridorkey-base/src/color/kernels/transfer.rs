//! Colour transfer functions (`corridorkey/color_utils.py`) as `#[cube]` helpers, so fused
//! kernels (frame upload, output packing) call them inline, plus one elementwise conversion
//! kernel.
//!
//! Everything works on `Vector<F, N>`: the launch picks `N` (the widest load the device reports
//! as optimal for memory traffic, `io_optimized_vector_sizes`), so one unit moves 16 bytes per
//! access instead of 4. Branches are selects (`select_many`): every lane computes both sides,
//! which is cheaper than divergence for a memory-bound kernel and is the only form a vector
//! condition has.

use cubecl::prelude::*;

use crate::core::CkFloat;

/// Encodings, as comptime codes of [`convert_kernel`].
pub const SRGB: u32 = 0;
pub const LINEAR: u32 = 1;
pub const ASINH: u32 = 2;

/// Linear threshold of the sRGB encoding (IEC 61966-2-1).
pub const SRGB_LINEAR_KNEE: f32 = 0.0031308;
/// Encoded threshold of the sRGB encoding.
pub const SRGB_ENCODED_KNEE: f32 = 0.04045;

/// `y` with the sign of `x` (odd-symmetric extension of a function of `|x|`).
#[cube]
fn with_sign_of<F: Float, N: Size>(y: Vector<F, N>, x: Vector<F, N>) -> Vector<F, N> {
    select_many(x.less_than(&Vector::zero()), -y, y)
}

/// Scene-linear → sRGB, odd-symmetric like `linear_to_srgb` (negative values mirror positive ones).
#[cube]
pub fn linear_to_srgb<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let a = Vector::abs(x);
    let low = a * Vector::new(F::new(12.92f32));
    let high = Vector::new(F::new(1.055f32)) * Vector::powf(a, Vector::new(F::new(1.0f32 / 2.4f32))) - Vector::new(F::new(0.055f32));
    with_sign_of(select_many(a.greater_than(&Vector::new(F::new(SRGB_LINEAR_KNEE))), high, low), x)
}

/// sRGB → scene-linear, odd-symmetric like `srgb_to_linear`.
#[cube]
pub fn srgb_to_linear<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let a = Vector::abs(x);
    let low = a * Vector::new(F::new(1.0f32 / 12.92f32));
    let high = Vector::powf((a + Vector::new(F::new(0.055f32))) * Vector::new(F::new(1.0f32 / 1.055f32)), Vector::new(F::new(2.4f32)));
    with_sign_of(select_many(a.greater_than(&Vector::new(F::new(SRGB_ENCODED_KNEE))), high, low), x)
}

/// Cutoff where [`asinh_nonnegative`] switches from its 7th-order Taylor polynomial to
/// `ln(a + √(1 + a²))`. Below `0.2`, the 9th-order remainder `35/1152 · a⁸` is `< 8e-8` relative;
/// above `0.2`, `a + √(1 + a²) > 1.21`, so `f32` rounding before `ln` is `< 6e-7` relative.
pub const ASINH_TAYLOR_CUTOFF: f32 = 0.2;

/// `asinh(a)` for `a ≥ 0`, accurate to `< 1e-6` relative over the whole `f32` range.
///
/// Backends' `asinh` and `log1p` are not: CubeCL's CUDA `asinh` (`ln(x + √(x² + 1))`) measured
/// 5e-3 relative error for large negative arguments and 1.5e-5 for small ones against NumPy, and
/// `cubecl-core` lowers `Vector::log1p(u)` to `(u + 1).ln()` on every backend.
#[cube]
fn asinh_nonnegative<F: Float, N: Size>(a: Vector<F, N>) -> Vector<F, N> {
    let a2 = a * a;
    let c1 = Vector::new(F::new(1.0f32 / 6.0f32));
    let c2 = Vector::new(F::new(3.0f32 / 40.0f32));
    let c3 = Vector::new(F::new(5.0f32 / 112.0f32));
    let small = a - a * a2 * (c1 - a2 * (c2 - a2 * c3));
    let large = Vector::ln(a + Vector::sqrt(Vector::new(F::new(1.0f32)) + a2));
    select_many(a.less_than(&Vector::new(F::new(ASINH_TAYLOR_CUTOFF))), small, large)
}

/// Scene-linear → signed asinh: `asinh(x / knee) / asinh(ref / knee)`, with `inv_knee = 1 / knee`
/// and `inv_denom = 1 / asinh(ref / knee)` from the host. Odd-symmetric: `sign(x) · asinh(|x|)`.
#[cube]
pub fn linear_to_asinh<F: Float, N: Size>(x: Vector<F, N>, inv_knee: F, inv_denom: F) -> Vector<F, N> {
    with_sign_of(asinh_nonnegative(Vector::abs(x) * Vector::new(inv_knee)), x) * Vector::new(inv_denom)
}

/// Below this `|x|`, [`sinh_accurate`] uses its 7th-order Taylor polynomial (9th-order remainder
/// `x⁹ / 362880` is `< 6e-9` relative); above it `(eˣ - e⁻ˣ) / 2` has no cancellation to speak of.
pub const SINH_TAYLOR_CUTOFF: f32 = 0.5;

/// `sinh(x)` accurate to a few ulp over the `f32` range, independent of the backend: Vulkan's
/// `GLSL.std.450 Sinh` carries no precision guarantee (like the backends' `asinh`, see
/// [`asinh_nonnegative`]), and `(eˣ - e⁻ˣ) / 2` alone cancels near 0.
#[cube]
pub fn sinh_accurate<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let x2 = x * x;
    let small = x * (Vector::new(F::new(1.0f32))
        + x2 * (Vector::new(F::new(1.0f32 / 6.0f32))
            + x2 * (Vector::new(F::new(1.0f32 / 120.0f32)) + x2 * Vector::new(F::new(1.0f32 / 5040.0f32)))));
    let large = (Vector::exp(x) - Vector::exp(-x)) * Vector::new(F::new(0.5f32));
    select_many(Vector::abs(x).less_than(&Vector::new(F::new(SINH_TAYLOR_CUTOFF))), small, large)
}

/// Inverse of [`linear_to_asinh`]: `sinh(y · denom) · knee`.
#[cube]
pub fn asinh_to_linear<F: Float, N: Size>(y: Vector<F, N>, knee: F, denom: F) -> Vector<F, N> {
    sinh_accurate::<F, N>(y * Vector::new(denom)) * Vector::new(knee)
}

/// `output[i] = encode_to(scale · decode_from(input[i]) + offset)` for vectors `i < len`: decode
/// to scene-linear, apply the linear adjustment (`LinearAdjust` as an affine map; identity is
/// `scale = 1, offset = 0`), encode. `(knee, denom)` and their inverses parametrise the asinh
/// encoding. One unit per vector of `N` values.
#[cube(launch)]
pub fn convert_kernel<F: CkFloat, N: Size>(
    input: &[Vector<F, N>],
    output: &mut [Vector<F, N>],
    scale: F,
    offset: F,
    knee: F,
    denom: F,
    inv_knee: F,
    inv_denom: F,
    len: u32,
    #[comptime] from: u32,
    #[comptime] to: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < len {
        let mut x = input[i as usize];
        if from == SRGB {
            x = srgb_to_linear::<F, N>(x);
        } else if from == ASINH {
            x = asinh_to_linear::<F, N>(x, knee, denom);
        }
        x = x * Vector::new(scale) + Vector::new(offset);
        if to == SRGB {
            x = linear_to_srgb::<F, N>(x);
        } else if to == ASINH {
            x = linear_to_asinh::<F, N>(x, inv_knee, inv_denom);
        }
        output[i as usize] = x;
    }
}
