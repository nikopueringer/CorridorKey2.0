//! C-RADIOv4-SO400M against PyTorch fp32 (`scripts/parity/cradio_reference.py`).
//!
//! ```text
//! uv run python scripts/parity/cradio_reference.py /tmp/corridorkey-parity-cradio
//! cargo test --release -p corridorkey-ml --test parity_cradio -- --ignored --nocapture
//! ```

use std::path::Path;

use corridorkey_base::core::{cast_f32, to_f64};
use corridorkey_core::ComputeTarget;
use corridorkey_ml::models::foundations::resize_table_bilinear;
use corridorkey_ml::{CRadio, FoundationStats, MlContext, SafetensorsFile};
use half::f16;
use serde::Deserialize;

const REF_DIR: &str = "/tmp/corridorkey-parity-cradio";

#[derive(Deserialize)]
struct Meta {
    n: usize,
    he: usize,
    we: usize,
    rows: usize,
    cols: usize,
    dim: usize,
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e} (run scripts/parity/cradio_reference.py)", path.display()));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

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
#[ignore = "needs vfms/cradio and scripts/parity/cradio_reference.py output"]
fn cradio_matches_pytorch() {
    let dir = Path::new(REF_DIR);
    let m: Meta = serde_json::from_str(
        &std::fs::read_to_string(dir.join("meta.json")).expect("run scripts/parity/cradio_reference.py first"),
    )
    .unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let st = SafetensorsFile::open(root.join("vfms/cradio/C-RADIOv4-SO400M/model.safetensors")).unwrap();
    let stats = FoundationStats::from_file(root.join("weights/foundation_stats.safetensors")).unwrap();
    let input = read_f32(&dir.join("input_nhwc.f32"));
    let want = read_f32(&dir.join("normalized_nhwc.f32"));

    // CPE position table: host bilinear resize + top-left crop vs PyTorch's eval path.
    let (pshape, pos) = st.read_f32("radio_model.model.patch_generator.pos_embed").unwrap();
    let grid = (pshape[1] as f64).sqrt() as usize;
    let pos: Vec<f64> = pos.into_iter().map(f64::from).collect();
    let side = m.rows.max(m.cols);
    let square = resize_table_bilinear(&pos, (grid, grid), m.dim, (side, side));
    let table: Vec<f64> = (0..m.rows).flat_map(|r| square[r * side * m.dim..(r * side + m.cols) * m.dim].to_vec()).collect();
    let (pos_max, _) = errors(table.into_iter(), &read_f32(&dir.join("pos_embed.f32")));
    println!("CPE position table {}x{} from {grid}x{grid}: max |err| {pos_max:.3e}", m.rows, m.cols);
    assert!(pos_max < 1e-6, "CPE position table differs from PyTorch: {pos_max:.3e}");

    let scale = want.iter().fold(0f64, |a, &v| a.max(v.abs() as f64));
    let shape = [m.n, m.he, m.we, 3];
    for target in ComputeTarget::available() {
        let ctx = MlContext::new(target).unwrap();
        let model = CRadio::load::<f32>(&ctx, &st, &stats).unwrap();
        let got: Vec<f32> = ctx.download_tensor(model.forward::<f32>(&ctx, ctx.upload_tensor(&input, shape).unwrap()).unwrap()).unwrap();
        assert_eq!(got.len(), want.len());
        let (max32, mean32) = errors(got.iter().map(|&v| v as f64), &want);
        println!("{target} f32: max |err| {max32:.3e}, mean {mean32:.3e} (max |ref| {scale:.2})");

        let model16 = CRadio::load::<f16>(&ctx, &st, &stats).unwrap();
        let x16 = ctx.upload_tensor(&cast_f32::<f16>(&input), shape).unwrap();
        let got16: Vec<f16> = ctx.download_tensor(model16.forward::<f16>(&ctx, x16).unwrap()).unwrap();
        let (max16, mean16) = errors(got16.iter().map(|&v| to_f64(v)), &want);
        println!("{target} f16: max |err| {max16:.3e}, mean {mean16:.3e}");

        // 27 blocks of 1152-wide f32 GEMMs; features standardised (O(1), tails to `scale`).
        assert!(max32 / scale < 1e-3 && mean32 < 1e-3, "{target} f32 C-RADIO features: {max32:.3e} / {mean32:.3e}");
        assert!(mean16 < 5e-2, "{target} f16 C-RADIO features: mean {mean16:.3e}");
    }
}
