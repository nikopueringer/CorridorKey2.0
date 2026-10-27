//! Probe (dsp-kitchen §6b): how each runtime converts f32 to f16 inside a kernel, against
//! round-to-nearest-even on the host.
//!
//! `cargo test --release -p corridorkey-ml --test probe_f16_convert -- --ignored --nocapture`

use corridorkey_base::core::buffer::{download, empty, upload};
use corridorkey_core::ComputeTarget;
use corridorkey_core::compute::{ComputeTask, LaunchGeometry};
use cubecl::prelude::*;
use half::f16;

#[cube(launch)]
fn convert_kernel(input: &[f32], output: &mut [f16], len: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < len {
        output[i as usize] = f16::cast_from(input[i as usize]);
    }
}

struct Probe;

impl ComputeTask for Probe {
    type Output = (usize, usize, usize, usize);
    /// `(values, round-to-nearest matches, truncated toward zero, other)`.
    fn run(self, client: Client) -> Self::Output {
        // Values spread over the f16 normal range, most not representable in f16.
        let xs: Vec<f32> = (0..200_000).map(|i| ((i as f32 * 0.618_034).fract() - 0.5) * 2.0f32.powi((i % 20) - 6)).collect();
        let n = xs.len();
        let input = upload(&client, &xs);
        let output = empty::<f16>(&client, n);
        let geom = LaunchGeometry::elementwise(&client, n);
        unsafe {
            convert_kernel::launch(
                &client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(input, n),
                BufferArg::from_raw_parts(output.clone(), n),
                n as u32,
            );
        }
        let got: Vec<f16> = download(&client, output);
        let (mut rne, mut trunc, mut other) = (0, 0, 0);
        for (&x, &g) in xs.iter().zip(&got) {
            let nearest = f16::from_f32(x); // round to nearest even
            let toward_zero = {
                // f16 of largest magnitude not exceeding |x|
                let a = f16::from_f32(x.abs());
                let a = if a.to_f32() > x.abs() { f16::from_bits(a.to_bits() - 1) } else { a };
                if x < 0.0 { -a } else { a }
            };
            if g.to_bits() == nearest.to_bits() {
                rne += 1;
            } else if g.to_bits() == toward_zero.to_bits() {
                trunc += 1;
            } else {
                other += 1;
            }
        }
        (n, rne, trunc, other)
    }
}

#[test]
#[ignore = "probe"]
fn f32_to_f16_rounding_per_runtime() {
    for target in ComputeTarget::available() {
        let (n, rne, trunc, other) = target.run(Probe).unwrap();
        println!(
            "{target}: {n} conversions: round-to-nearest {rne} ({:.1} %), toward-zero only {trunc} ({:.1} %), other {other}",
            100.0 * rne as f64 / n as f64,
            100.0 * trunc as f64 / n as f64
        );
    }
}
