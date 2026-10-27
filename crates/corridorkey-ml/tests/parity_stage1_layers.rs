use std::{fs, path::Path};

use corridorkey_core::ComputeTarget;
use corridorkey_ml::{MlContext, SafetensorsFile, Stage1};
use serde::Deserialize;

#[derive(Deserialize)]
struct Stage1Meta {
    b: usize,
    t: usize,
    h: usize,
    w: usize,
    vfm_channels: usize,
}

fn read_f32_file(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn diff_stats(got: &[f32], want: &[f32]) -> (f64, f64) {
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    for (&g, &r) in got.iter().zip(want) {
        let err = (g as f64 - r as f64).abs();
        max_abs = max_abs.max(err);
        sum_abs += err;
    }
    (sum_abs / (want.len() as f64), max_abs)
}

#[test]
#[ignore = "diagnostic: needs a one-off dump of Stage 1 intermediates (fuse_tokens_bthwc.f32, ...) that no reference script writes; parity_stage1 covers Stage 1 end to end"]
fn test_stage1_layer_by_layer() {
    let dir = Path::new("/tmp/corridorkey-parity-stage1");
    if !dir.join("stage1_meta.json").exists() {
        return;
    }
    let ckpt_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/corridorkey_v2.safetensors");
    if !ckpt_path.exists() {
        return;
    }

    let meta: Stage1Meta = serde_json::from_str(&fs::read_to_string(dir.join("stage1_meta.json")).unwrap()).unwrap();
    let rgb_hint = read_f32_file(&dir.join("rgb_hint_bthwc.f32"));
    let (b, t, h, w) = (meta.b, meta.t, meta.h, meta.w);
    let bt = b * t;
    let (h16, w16) = (h / 16, w / 16);

    let st = SafetensorsFile::open(&ckpt_path).unwrap();
    let header = st.checkpoint_header().unwrap();

    let target = ComputeTarget::available().into_iter().next().unwrap();
    let ctx = MlContext::new(target).unwrap();
    let stage1 = Stage1::load::<f32>(&ctx, &st, &header.model_config, "stage1").unwrap();

    let t_rgb_hint = ctx.upload_tensor(&rgb_hint, [bt, h, w, 4]).unwrap();

    // 1. Stem
    let stem_out = stage1.stem.forward::<f32>(&ctx, t_rgb_hint).unwrap();
    let got_stem = ctx.download_tensor::<f32, 4>(stem_out.clone()).unwrap();
    let want_stem = read_f32_file(&dir.join("stem_bthwc.f32"));
    let (mean_stem, max_stem) = diff_stats(&got_stem, &want_stem);
    println!("Stem: mean={mean_stem:.3e}, max={max_stem:.3e}");

    // 2. Enc 16
    let hq = read_f32_file(&dir.join("hint_quality_bt.f32"));
    let t_hq = ctx.upload_tensor(&hq, [b * t, 1]).unwrap();
    let mut f_16 = stem_out.reshape([b, t, h16, w16, stage1.d_16]);
    for (i, blk) in stage1.enc_16.iter().enumerate() {
        f_16 = blk.forward::<f32>(&ctx, f_16, Some(t_hq.clone())).unwrap();
        let got = ctx.download_tensor::<f32, 5>(f_16.clone()).unwrap();
        let want = read_f32_file(&dir.join(format!("enc_16_{i}_bthwc.f32")));
        let (mean, max) = diff_stats(&got, &want);
        println!("Enc 16 [{i}]: mean={mean:.3e}, max={max:.3e}");
    }

    // 3. Down
    let (h64, w64) = (h / 64, w / 64);
    let f_64_down = stage1.down.forward::<f32>(&ctx, f_16.clone().reshape([bt, h16, w16, stage1.d_16])).unwrap();
    let got_down = ctx.download_tensor::<f32, 4>(f_64_down.clone()).unwrap();
    let want_down = read_f32_file(&dir.join("down_bthwc.f32"));
    let (mean_down, max_down) = diff_stats(&got_down, &want_down);
    println!("Down: mean={mean_down:.3e}, max={max_down:.3e}");

    // 4. Fuse
    let vfm_64 = read_f32_file(&dir.join("vfm_64_bthwc.f32"));
    let t_vfm = ctx.upload_tensor(&vfm_64, [b, t, h64, w64, meta.vfm_channels]).unwrap();
    let combined = burn_tensor::Tensor::cat(vec![t_vfm, f_64_down.reshape([b, t, h64, w64, stage1.d_16])], 4);
    let got_combined = ctx.download_tensor::<f32, 5>(combined.clone()).unwrap();
    let want_tokens = read_f32_file(&dir.join("fuse_tokens_bthwc.f32"));
    let (mean_tok, max_tok) = diff_stats(&got_combined, &want_tokens);
    println!("Tokens: mean={mean_tok:.3e}, max={max_tok:.3e}");

    let mut f_64 = if let Some(fuse) = &stage1.fuse {
        let f = fuse.forward::<f32, 5>(&ctx, combined).unwrap();
        let got_fuse = ctx.download_tensor::<f32, 5>(f.clone()).unwrap();
        let want_fuse = read_f32_file(&dir.join("fuse_bthwc.f32"));
        let (mean_fuse, max_fuse) = diff_stats(&got_fuse, &want_fuse);
        println!("Fuse: mean={mean_fuse:.3e}, max={max_fuse:.3e}");
        f
    } else {
        combined
    };

    // 5. Body 64
    let split = stage1.body_64.len() / 2;
    for (i, blk) in stage1.body_64[..split].iter().enumerate() {
        f_64 = blk.forward::<f32>(&ctx, f_64, 1.0).unwrap();
        let got = ctx.download_tensor::<f32, 5>(f_64.clone()).unwrap();
        let want = read_f32_file(&dir.join(format!("body_64_{i}_bthwc.f32")));
        let (mean, max) = diff_stats(&got, &want);
        println!("Body 64 [{i}]: mean={mean:.3e}, max={max:.3e}");
    }
}
