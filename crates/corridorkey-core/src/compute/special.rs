//! Special float values inside kernels.
//!
//! WGSL has no literal for infinity, so `F::new(f32::INFINITY)` fails to compile on the wgpu
//! runtime. These helpers build ±∞ from their IEEE-754 bits and cast to the kernel's float type
//! (an `f32` infinity casts to infinity in every float type).

use cubecl::prelude::*;

/// IEEE-754 bits of `f32` +∞.
pub const POSITIVE_INFINITY_BITS: u32 = f32::INFINITY.to_bits();

/// IEEE-754 bits of `f32` −∞.
pub const NEGATIVE_INFINITY_BITS: u32 = f32::NEG_INFINITY.to_bits();

/// +∞ in `F`.
#[cube]
pub fn positive_infinity<F: Float>() -> F {
    F::cast_from(f32::reinterpret(POSITIVE_INFINITY_BITS))
}

/// −∞ in `F`.
#[cube]
pub fn negative_infinity<F: Float>() -> F {
    F::cast_from(f32::reinterpret(NEGATIVE_INFINITY_BITS))
}
