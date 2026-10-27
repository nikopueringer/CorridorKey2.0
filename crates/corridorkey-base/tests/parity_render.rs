//! Output rendering against the product's `_render_predictions_torch` / `_render_predictions_exr_torch`
//! (`scripts/parity/render_reference.py`): every output, both cutout modes, 16-bit video and f32 EXR.
//!
//! ```text
//! uv run python scripts/parity/render_reference.py /tmp/corridorkey-parity-render
//! cargo test --release -p corridorkey-base --test parity_render -- --ignored --nocapture
//! ```

use std::path::Path;

use corridorkey_base::core::buffer::upload;
use corridorkey_base::frame::{RenderOutput, RenderSettings, render_exr, render_video};
use corridorkey_core::compute::ComputeTask;
use corridorkey_core::{ComputeTarget, LinearAdjust};
use cubecl::prelude::*;
use serde_json::Value;

const REF_DIR: &str = "/tmp/corridorkey-parity-render";
const OUTPUTS: [(RenderOutput, &str); 4] = [
    (RenderOutput::Alpha, "alpha"),
    (RenderOutput::Fg, "fg"),
    (RenderOutput::Cutout, "cutout"),
    (RenderOutput::Checker, "checker"),
];

fn read(name: &str) -> Vec<u8> {
    let path = Path::new(REF_DIR).join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e} (run scripts/parity/render_reference.py)", path.display()))
}

struct Render;

impl ComputeTask for Render {
    /// `(name, exact-match fraction, max code difference, fraction more than one code apart)` per
    /// video output and
    /// `(name, max |error| / max(1, |reference|), max |reference|)` per EXR output.
    type Output = (Vec<(String, f64, u16, f64)>, Vec<(String, f64, f64)>);
    fn run(self, client: Client) -> Self::Output {
        let meta: Value = serde_json::from_str(&String::from_utf8(read("meta.json")).unwrap()).unwrap();
        let num = |k: &str| meta[k].as_f64().unwrap();
        let t = num("t") as usize;
        let hw = |k: &str| (meta[k][0].as_u64().unwrap() as usize, meta[k][1].as_u64().unwrap() as usize);
        let pad: Vec<usize> = meta["pad"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let mut settings = RenderSettings {
            native_hw: hw("native_hw"),
            padded_hw: (num("h") as usize, num("w") as usize),
            pad: (pad[2], pad[0]),
            adjust: LinearAdjust { gain: num("linear_brightness"), contrast: num("linear_contrast"), pivot: num("linear_contrast_pivot") },
            despill: num("despill"),
            cutout_linear: false,
        };
        let logits: Vec<f32> = read("logits_nhwc.f32").chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let handle = upload(&client, &logits);

        let mut video = Vec::new();
        for linear in [false, true] {
            settings.cutout_linear = linear;
            for (out, name) in OUTPUTS {
                if linear && out != RenderOutput::Cutout {
                    continue;
                }
                let file = format!("video_{name}{}.u16", if linear { "_linear" } else { "" });
                let want: Vec<u16> = read(&file).chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
                let got = render_video::<f32>(&client, &handle, t, &settings, out).unwrap();
                assert_eq!(got.len(), want.len(), "{file}");
                let same = got.iter().zip(&want).filter(|(g, w)| g == w).count();
                if std::env::var("RENDER_DEBUG").is_ok() {
                    for (i, (&g, &w)) in got.iter().zip(&want).enumerate().filter(|(_, (g, w))| g.abs_diff(**w) > 1).take(8) {
                        let ch = i % 4;
                        let px = i / 4;
                        println!("  {file} value {i} (pixel {px}, channel {ch}): got {g}, want {w}");
                    }
                }
                let max_diff = got.iter().zip(&want).map(|(&g, &w)| g.abs_diff(w)).max().unwrap();
                let beyond_one = got.iter().zip(&want).filter(|(g, w)| g.abs_diff(**w) > 1).count();
                video.push((file, same as f64 / want.len() as f64, max_diff, beyond_one as f64 / want.len() as f64));
            }
        }
        settings.cutout_linear = false;
        let mut exr = Vec::new();
        for (out, name) in OUTPUTS {
            let file = format!("exr_{name}.f32");
            let want: Vec<f32> = read(&file).chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            let got = render_exr::<f32>(&client, &handle, t, &settings, out).unwrap();
            assert_eq!(got.len(), want.len(), "{file}");
            let rel = got.iter().zip(&want).fold(0f64, |m, (&g, &w)| m.max((g as f64 - w as f64).abs() / (w.abs() as f64).max(1.0)));
            let peak = want.iter().fold(0f64, |m, &w| m.max(w.abs() as f64));
            exr.push((file, rel, peak));
        }
        (video, exr)
    }
}

#[test]
#[ignore = "needs scripts/parity/render_reference.py output"]
fn rendering_matches_the_product() {
    for target in ComputeTarget::available() {
        let (video, exr) = target.run(Render).unwrap();
        for (file, exact, max_diff, beyond_one) in &video {
            println!(
                "{target} {file:28}: {:.4} % identical codes, max difference {max_diff} code(s), {:.4} % > 1 code",
                exact * 100.0,
                beyond_one * 100.0
            );
            // Truncation to 16 bits flips by one code where device and CPU transcendentals differ in
            // the last bit of f32 just below a code boundary (more often for the checker's two `pow`
            // round trips, and on Vulkan). Larger differences only where the linear cutout despills
            // unclamped HDR values (hundreds) into [0, 1]: `g - (r + b) / 2` cancels numbers whose
            // f32 ulp is 3e-5..6e-5, so any two f32 evaluations differ by a few codes there.
            assert!(*exact > 0.995 && *beyond_one < 1e-4 && *max_diff <= 8, "{target} {file}: {exact} identical, max {max_diff}");
        }
        for (file, rel, peak) in &exr {
            println!("{target} {file:28}: max relative error {rel:.3e} (max |value| {peak:.2})");
            // Unclamped scene-linear HDR from `sinh`: judged relative to the value. The product
            // writes half-float EXR by default (~1e-3 relative); f32 device transcendentals land far below.
            assert!(*rel < 1e-4, "{target} {file}: {rel:.3e}");
        }
    }
}
