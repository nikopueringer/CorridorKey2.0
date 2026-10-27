//! Colour transfer functions on the device: sRGB, scene-linear and the model's signed asinh
//! encoding (`corridorkey/color_utils.py`), with the clip's linear adjustment folded in.
//!
//! Frames are converted where they are; the helpers in [`kernels`] are `#[cube]` functions so the
//! fused upload and output kernels apply them inline instead of launching a pass each.

pub mod kernels;

use cubecl::prelude::*;
use cubecl::server::Handle;
use corridorkey_core::compute::LaunchGeometry;
use corridorkey_core::frame::{HDR_ASINH_KNEE, HDR_ASINH_REF};
use corridorkey_core::{ColorSpace, LinearAdjust};

use crate::core::{cast, CkFloat};
use kernels::{convert_kernel, ASINH, LINEAR, SRGB};

fn code(space: ColorSpace) -> u32 {
    match space {
        ColorSpace::Srgb => SRGB,
        ColorSpace::Linear => LINEAR,
        ColorSpace::Asinh => ASINH,
    }
}

/// `(knee, asinh(ref / knee))` of the model's asinh encoding.
pub fn asinh_params() -> (f64, f64) {
    (HDR_ASINH_KNEE, (HDR_ASINH_REF / HDR_ASINH_KNEE).asinh())
}

/// `output = encode_to(adjust(decode_from(input)))` over `len` values, on the device. `adjust`
/// is applied in scene-linear light (pass `LinearAdjust::default()` for none; to undo an
/// adjustment, pass the inverse through [`convert_affine`]). `input` and `output` may be the same
/// buffer.
pub fn convert<F: CkFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    len: usize,
    from: ColorSpace,
    to: ColorSpace,
    adjust: LinearAdjust,
) {
    convert_affine::<F>(client, input, output, len, from, to, adjust.affine());
}

/// [`convert`] with the linear-light adjustment given as `(scale, offset)`.
pub fn convert_affine<F: CkFloat>(
    client: &Client,
    input: &Handle,
    output: &Handle,
    len: usize,
    from: ColorSpace,
    to: ColorSpace,
    (scale, offset): (f64, f64),
) {
    if len == 0 {
        return;
    }
    let (knee, denom) = asinh_params();
    let vector = vector_size::<F>(client, len);
    let units = len / vector;
    let geom = LaunchGeometry::elementwise(client, units);
    unsafe {
        convert_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            vector,
            BufferArg::from_raw_parts(input.clone(), len),
            BufferArg::from_raw_parts(output.clone(), len),
            cast::<F>(scale),
            cast::<F>(offset),
            cast::<F>(knee),
            cast::<F>(denom),
            cast::<F>(1.0 / knee),
            cast::<F>(1.0 / denom),
            units as u32,
            code(from),
            code(to),
        );
    }
}

/// Widest vector the device reports as optimal for loading `F` that divides `len` (1 when none
/// does): frames are `H × W × 3` values, so 4-wide loads apply whenever `H × W` is even.
pub fn vector_size<F: CkFloat>(client: &Client, len: usize) -> usize {
    client
        .io_optimized_vector_sizes(size_of::<F>())
        .map(|v| v as usize)
        .filter(|v| len.is_multiple_of(*v))
        .max()
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::buffer::{download, empty, upload};

    // Host references in f64, written from `color_utils.py`
    fn srgb_to_linear(x: f64) -> f64 {
        let a = x.abs();
        let y = if a <= 0.04045 { a / 12.92 } else { ((a + 0.055) / 1.055).powf(2.4) };
        y.copysign(x)
    }
    fn linear_to_srgb(x: f64) -> f64 {
        let a = x.abs();
        let y = if a <= 0.0031308 { a * 12.92 } else { 1.055 * a.powf(1.0 / 2.4) - 0.055 };
        y.copysign(x)
    }
    fn linear_to_asinh(x: f64) -> f64 {
        (x / HDR_ASINH_KNEE).asinh() / (HDR_ASINH_REF / HDR_ASINH_KNEE).asinh()
    }

    fn reference(x: f64, from: ColorSpace, to: ColorSpace, adjust: LinearAdjust) -> f64 {
        let (knee, denom) = asinh_params();
        let linear = match from {
            ColorSpace::Srgb => srgb_to_linear(x),
            ColorSpace::Linear => x,
            ColorSpace::Asinh => (x * denom).sinh() * knee,
        };
        let adjusted = if adjust.is_identity() {
            linear
        } else {
            adjust.gain * (adjust.pivot + adjust.contrast * (linear - adjust.pivot))
        };
        match to {
            ColorSpace::Srgb => linear_to_srgb(adjusted),
            ColorSpace::Linear => adjusted,
            ColorSpace::Asinh => linear_to_asinh(adjusted),
        }
    }

    fn matches_python_reference(client: &Client) {
        // Prime length (vectors of 1, partial last cube) and a multiple of 4 (widest vectors);
        // negatives, both sides of both knees, HDR values
        for n in [10_007, 12_000] {
        assert_eq!(vector_size::<f32>(client, n) > 1, n % 4 == 0, "{}: vector size for {n}", client.name());
        let linear: Vec<f64> = (0..n).map(|i| -2.0 + 22.0 * (i as f64 / (n - 1) as f64).powi(3)).collect();
        let out = empty::<f32>(client, n);
        let adjust = LinearAdjust { gain: 1.2, contrast: 0.9, pivot: 0.18 };
        use ColorSpace::*;
        for (from, to, adj) in [
            (Srgb, Linear, LinearAdjust::default()),
            (Linear, Srgb, LinearAdjust::default()),
            (Linear, Asinh, LinearAdjust::default()),
            (Asinh, Linear, LinearAdjust::default()),
            (Srgb, Asinh, adjust),
        ] {
            // Inputs in the source encoding's own range (asinh values of the same scene-linear values)
            let input: Vec<f64> = linear.iter().map(|&v| if from == Asinh { linear_to_asinh(v) } else { v }).collect();
            let host: Vec<f32> = input.iter().map(|&v| v as f32).collect();
            let x = upload(client, &host);
            convert::<f32>(client, &x, &out, n, from, to, adj);
            let got: Vec<f32> = download(client, out.clone());
            for (i, (&v, &g)) in input.iter().zip(&got).enumerate() {
                let want = reference(v as f32 as f64, from, to, adj);
                let tol = 2e-5 * want.abs().max(1.0);
                assert!((g as f64 - want).abs() <= tol, "{} n={n} {from:?}→{to:?} [{i}] x={v}: {g} vs {want}", client.name());
            }
        }
        }
    }
    runtime_test!(test_convert_matches_python, matches_python_reference);

    fn round_trips(client: &Client) {
        let host: Vec<f32> = (0..4_100).map(|i| -1.0 + i as f32 * 0.004).collect();
        let x = upload(client, &host);
        let y = empty::<f32>(client, host.len());
        let adjust = LinearAdjust { gain: 0.8, contrast: 1.3, pivot: 0.18 };
        convert::<f32>(client, &x, &y, host.len(), ColorSpace::Linear, ColorSpace::Asinh, adjust);
        // In place, undoing the adjustment
        convert_affine::<f32>(client, &y, &y, host.len(), ColorSpace::Asinh, ColorSpace::Linear, adjust.inverse_affine());
        let back: Vec<f32> = download(client, y);
        for (&a, &b) in host.iter().zip(&back) {
            assert!((a - b).abs() <= 1e-5 * a.abs().max(1.0), "{}: {a} → {b}", client.name());
        }
    }
    runtime_test!(test_convert_round_trips, round_trips);
}
