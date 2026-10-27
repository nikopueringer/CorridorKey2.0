//! Frame preparation against the product's own functions (`scripts/parity/frame_reference.py`):
//! reflect-101 padding, sRGB → linear → adjust → asinh, the backbones' view, and the area /
//! bilinear backbone inputs.
//!
//! ```text
//! uv run python scripts/parity/frame_reference.py /tmp/corridorkey-parity-frame
//! cargo test --release -p corridorkey-base --test parity_frame -- --ignored --nocapture
//! ```

use std::path::Path;

use corridorkey_base::core::CkFloat;
use corridorkey_base::core::buffer::download;
use corridorkey_base::core::to_f64;
use corridorkey_base::frame::{BackboneInputs, frame_views, upload_frames};
use corridorkey_core::compute::ComputeTask;
use corridorkey_core::{ComputeTarget, FrameMeta, LinearAdjust, Padding};
use cubecl::prelude::*;
use half::f16;
use serde_json::Value;

const REF_DIR: &str = "/tmp/corridorkey-parity-frame";

fn read_f32(name: &str) -> Vec<f32> {
    let path = Path::new(REF_DIR).join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e} (run scripts/parity/frame_reference.py)", path.display()));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn max_err<F: CkFloat>(got: &[F], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).fold(0f64, |m, (&g, &w)| m.max((to_f64(g) - w as f64).abs()))
}

struct Case<F> {
    _f: std::marker::PhantomData<F>,
}

impl<F: CkFloat> ComputeTask for Case<F> {
    /// Max |error| of model, foundation, cradio, moge, rvm.
    type Output = [f64; 5];
    fn run(self, client: Client) -> Self::Output {
        let meta_json: Value = serde_json::from_str(&std::fs::read_to_string(Path::new(REF_DIR).join("meta.json")).unwrap()).unwrap();
        let get = |k: &str| meta_json[k].as_f64().unwrap();
        let (t, h0, w0) = (get("t") as usize, get("h0") as usize, get("w0") as usize);
        let meta = FrameMeta::new((h0, w0), 1.0, 128, 24.0, t).unwrap();
        let pad: Vec<usize> = meta_json["pad"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(meta.pad, Padding { left: pad[0], right: pad[1], top: pad[2], bottom: pad[3] }, "padding geometry");
        let adjust = LinearAdjust { gain: get("linear_brightness"), contrast: get("linear_contrast"), pivot: get("linear_contrast_pivot") };

        let bytes = std::fs::read(Path::new(REF_DIR).join("frames_u8.u8")).unwrap();
        let frames = upload_frames(&client, &bytes);
        let views = frame_views::<F>(&client, &frames, t, &meta, adjust).unwrap();
        let inputs = BackboneInputs::from_views::<F>(&client, &views);
        let get_buf = |h: &cubecl::server::Handle| -> Vec<F> { download(&client, h.clone()) };
        [
            max_err(&get_buf(&views.model), &read_f32("model_nhwc.f32")),
            max_err(&get_buf(&views.foundation), &read_f32("foundation_nhwc.f32")),
            max_err(&get_buf(&inputs.cradio), &read_f32("cradio_nhwc.f32")),
            max_err(&get_buf(&inputs.moge), &read_f32("moge_nhwc.f32")),
            max_err(&get_buf(&inputs.rvm), &read_f32("rvm_nhwc.f32")),
        ]
    }
}

#[test]
#[ignore = "needs scripts/parity/frame_reference.py output"]
fn frame_preparation_matches_the_product() {
    for target in ComputeTarget::available() {
        let e32 = target.run(Case::<f32> { _f: std::marker::PhantomData }).unwrap();
        let e16 = target.run(Case::<f16> { _f: std::marker::PhantomData }).unwrap();
        let names = ["model", "foundation", "cradio", "moge", "rvm"];
        for (i, name) in names.iter().enumerate() {
            println!("{target} {name:10}: f32 max {:.3e}, f16 max {:.3e}", e32[i], e16[i]);
        }
        // f32: transcendental functions on the device (~1e-6 relative; asinh model values ≲ 1.5).
        assert!(e32.iter().all(|&e| e < 1e-5), "{target} f32: {e32:?}");
        // f16: one rounding of values ≤ ~1.5 (≤ 1e-3), resizes of rounded views.
        assert!(e16.iter().all(|&e| e < 2e-3), "{target} f16: {e16:?}");
    }
}
