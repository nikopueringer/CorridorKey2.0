//! MoGe-2 encoder against PyTorch fp32 (`scripts/parity/moge_reference.py`).
//!
//! ```text
//! uv run python scripts/convert_foundations.py
//! uv run python scripts/parity/moge_reference.py /tmp/corridorkey-parity-moge
//! cargo test --release -p corridorkey-ml --test parity_moge -- --ignored --nocapture
//! ```

use std::path::Path;

use corridorkey_base::core::{cast_f32, to_f64};
use corridorkey_core::ComputeTarget;
use corridorkey_ml::models::foundations::resize_pos_embed_bicubic;
use corridorkey_ml::{FoundationStats, MlContext, MogeEncoder, SafetensorsFile};
use half::f16;
use serde::Deserialize;

const REF_DIR: &str = "/tmp/corridorkey-parity-moge";

#[derive(Deserialize)]
struct Meta {
    n: usize,
    hm: usize,
    wm: usize,
    rows: usize,
    cols: usize,
    dim: usize,
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e} (run scripts/parity/moge_reference.py)", path.display()));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

/// `(max |error|, mean |error|)`.
fn errors(got: impl Iterator<Item = f64>, want: &[f32]) -> (f64, f64) {
    let (mut max, mut sum) = (0f64, 0f64);
    for (g, &w) in got.zip(want) {
        let e = (g - w as f64).abs();
        max = max.max(e);
        sum += e;
    }
    (max, sum / want.len() as f64)
}

#[test]
#[ignore = "needs scripts/convert_foundations.py and scripts/parity/moge_reference.py output"]
fn moge_encoder_matches_pytorch() {
    let dir = Path::new(REF_DIR);
    let meta: Meta = serde_json::from_str(
        &std::fs::read_to_string(dir.join("meta.json")).expect("run scripts/parity/moge_reference.py first"),
    )
    .unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let st = SafetensorsFile::open(root.join("weights/foundations/moge2_vitb_encoder.safetensors"))
        .expect("run scripts/convert_foundations.py first");
    let stats = FoundationStats::from_file(root.join("weights/foundation_stats.safetensors")).unwrap();
    let input = read_f32(&dir.join("input_nhwc.f32"));
    let want = read_f32(&dir.join("normalized_nhwc.f32"));

    // Position embedding: host bicubic resize vs PyTorch's interpolate.
    let (_, pos) = st.read_f32("backbone.pos_embed").unwrap();
    let pos: Vec<f64> = pos.into_iter().map(f64::from).collect();
    let grid = ((pos.len() / meta.dim - 1) as f64).sqrt() as usize;
    let resized = resize_pos_embed_bicubic(&pos[meta.dim..], grid, meta.dim, (meta.rows, meta.cols), 0.1);
    let want_pos = read_f32(&dir.join("pos_embed.f32"));
    let (pos_max, _) = errors(resized.iter().copied(), &want_pos[meta.dim..]);
    println!("pos_embed {}x{} from {grid}x{grid}: max |err| {pos_max:.3e}", meta.rows, meta.cols);
    assert!(pos_max < 1e-5, "bicubic position-embedding resize differs from PyTorch: {pos_max:.3e}");

    let scale = want.iter().fold(0f64, |m, &v| m.max(v.abs() as f64));
    let shape = [meta.n, meta.hm, meta.wm, 3];
    for target in ComputeTarget::available() {
        let ctx = MlContext::new(target).unwrap();

        let enc = MogeEncoder::load::<f32>(&ctx, &st, &stats).unwrap();
        let x = ctx.upload_tensor(&input, shape).unwrap();
        let got: Vec<f32> = ctx.download_tensor(enc.forward::<f32>(&ctx, x).unwrap()).unwrap();
        assert_eq!(got.len(), want.len());
        let (max32, mean32) = errors(got.iter().map(|&v| v as f64), &want);
        println!("{target} f32: max |err| {max32:.3e}, mean {mean32:.3e} (max |ref| {scale:.2})");

        let enc16 = MogeEncoder::load::<f16>(&ctx, &st, &stats).unwrap();
        let x16 = ctx.upload_tensor(&cast_f32::<f16>(&input), shape).unwrap();
        let got16: Vec<f16> = ctx.download_tensor(enc16.forward::<f16>(&ctx, x16).unwrap()).unwrap();
        let (max16, mean16) = errors(got16.iter().map(|&v| to_f64(v)), &want);
        println!("{target} f16: max |err| {max16:.3e}, mean {mean16:.3e}");

        // f32: 12 ViT blocks of f32 GEMMs and f32 reductions; features are standardised (O(1)).
        assert!(max32 / scale < 1e-4 && mean32 < 1e-4, "{target} f32 MoGe features: {max32:.3e} / {mean32:.3e}");
        assert!(mean16 < 2e-2, "{target} f16 MoGe features: mean {mean16:.3e}");
    }
}
