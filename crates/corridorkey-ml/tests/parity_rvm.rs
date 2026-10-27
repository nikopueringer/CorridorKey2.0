//! RVM-small against PyTorch fp32 (`scripts/parity/rvm_reference.py`), including the recurrent
//! state carried from one call to the next (the product feeds a clip in chunks of frames).
//!
//! ```text
//! uv run python scripts/convert_foundations.py
//! uv run python scripts/parity/rvm_reference.py /tmp/corridorkey-parity-rvm
//! cargo test --release -p corridorkey-ml --test parity_rvm -- --ignored --nocapture
//! ```

use std::path::Path;

use corridorkey_base::core::{CkFloat, cast_f32, to_f64};
use corridorkey_core::ComputeTarget;
use corridorkey_ml::{FoundationStats, MlContext, RvmSmall, SafetensorsFile};
use half::f16;
use serde::Deserialize;

const REF_DIR: &str = "/tmp/corridorkey-parity-rvm";

#[derive(Deserialize)]
struct Meta {
    t: usize,
    he: usize,
    we: usize,
    rows: usize,
    cols: usize,
    dim: usize,
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e} (run scripts/parity/rvm_reference.py)", path.display()));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn errors<F: CkFloat>(got: &[F], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let (mut max, mut sum) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let e = (to_f64(g) - w as f64).abs();
        max = max.max(e);
        sum += e;
    }
    (max, sum / want.len() as f64)
}

/// Runs all frames in one call, then as `[0, split)` + `[split, t)` with the state handed over;
/// returns `(features, final state, features of the split run)`.
fn run<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, stats: &FoundationStats, input: &[f32], m: &Meta, split: usize) -> (Vec<F>, Vec<F>, Vec<F>) {
    let rvm = RvmSmall::load::<F>(ctx, st, stats).unwrap();
    let frame = m.he * m.we * 3;
    let upload = |range: std::ops::Range<usize>| {
        let n = range.len();
        ctx.upload_tensor(&cast_f32::<F>(&input[range.start * frame..range.end * frame]), [n, m.he, m.we, 3]).unwrap()
    };
    let (feats, state) = rvm.forward::<F>(ctx, upload(0..m.t), rvm.zero_state::<F>(ctx, m.rows, m.cols)).unwrap();
    let feats: Vec<F> = ctx.download_tensor(feats).unwrap();
    let state_len = (1 + m.rows * m.cols) * m.dim;
    let state: Vec<F> = ctx.download_tensor(ctx.tensor_from_handle::<1>(state, [state_len], corridorkey_ml::elem_dtype::<F>())).unwrap();

    let (fa, sa) = rvm.forward::<F>(ctx, upload(0..split), rvm.zero_state::<F>(ctx, m.rows, m.cols)).unwrap();
    let (fb, _) = rvm.forward::<F>(ctx, upload(split..m.t), sa).unwrap();
    let mut split_feats: Vec<F> = ctx.download_tensor(fa).unwrap();
    split_feats.extend(ctx.download_tensor::<F, 4>(fb).unwrap());
    (feats, state, split_feats)
}

#[test]
#[ignore = "needs scripts/convert_foundations.py and scripts/parity/rvm_reference.py output"]
fn rvm_matches_pytorch_and_carries_state() {
    let dir = Path::new(REF_DIR);
    let m: Meta = serde_json::from_str(
        &std::fs::read_to_string(dir.join("meta.json")).expect("run scripts/parity/rvm_reference.py first"),
    )
    .unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let st = SafetensorsFile::open(root.join("weights/foundations/rvm_small.safetensors"))
        .expect("run scripts/convert_foundations.py first");
    let stats = FoundationStats::from_file(root.join("weights/foundation_stats.safetensors")).unwrap();
    let input = read_f32(&dir.join("input_nhwc.f32"));
    let want = read_f32(&dir.join("normalized_nhwc.f32"));
    let want_state = read_f32(&dir.join(format!("state_{}.f32", m.t - 1)));
    let split = m.t - 1;

    for target in ComputeTarget::available() {
        let ctx = MlContext::new(target).unwrap();
        let (f32_feats, f32_state, f32_split) = run::<f32>(&ctx, &st, &stats, &input, &m, split);
        let (fmax, fmean) = errors(&f32_feats, &want);
        let (smax, _) = errors(&f32_state, &want_state);
        let (cmax, _) = errors(&f32_split, &f32_feats.iter().map(|&v| v).collect::<Vec<f32>>());
        let (split_max, split_mean) = errors(&f32_split, &want);
        println!(
            "{target} f32: features max {fmax:.3e} mean {fmean:.3e}; state max {smax:.3e}; \
             split call vs PyTorch max {split_max:.3e} mean {split_mean:.3e} (vs single call max {cmax:.3e})"
        );

        let (f16_feats, _, _) = run::<f16>(&ctx, &st, &stats, &input, &m, split);
        let (hmax, hmean) = errors(&f16_feats, &want);
        println!("{target} f16: features max {hmax:.3e} mean {hmean:.3e}");

        assert!(fmax < 1e-3 && fmean < 1e-4, "{target} f32 RVM features: {fmax:.3e} / {fmean:.3e}");
        assert!(smax < 1e-3, "{target} f32 RVM state: {smax:.3e}");
        // Split calls batch the encoder over fewer frames, so GEMM shapes (and the autotuned
        // routines' summation order) differ: last-bit differences, then three recurrent steps. A
        // wrong or reset hand-over would differ at the scale of the features themselves (O(1)).
        assert!(split_max < 1e-3 && split_mean < 1e-4, "{target}: split calls vs PyTorch {split_max:.3e} / {split_mean:.3e}");
        assert!(cmax < 1e-4, "{target}: state carried across calls changes the features ({cmax:.3e})");
        assert!(hmean < 2e-2, "{target} f16 RVM features: mean {hmean:.3e}");
    }
}
