//! Stage 1 end-to-end parity and resident timing comparison against PyTorch CUDA
//! (`scripts/parity/stage1_reference.py`).

use std::{fs, path::Path, time::Instant};

use corridorkey_core::{ComputeTarget, compute::bench::time_device};
use corridorkey_ml::{MlContext, SafetensorsFile, Stage1};
use serde::Deserialize;

#[derive(Deserialize)]
struct Stage1Meta {
    device: String,
    torch: String,
    b: usize,
    t: usize,
    h: usize,
    w: usize,
    vfm_channels: usize,
    resident_ms: f64,
}

fn read_f32_file(path: &Path) -> Vec<f32> {
    let bytes =
        fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn compare_stage1_with_python() {
    let dir = Path::new("/tmp/corridorkey-parity-stage1");
    if !dir.join("stage1_meta.json").exists() {
        eprintln!(
            "skipping compare_stage1_with_python: /tmp/corridorkey-parity-stage1 not generated"
        );
        return;
    }
    let ckpt_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/corridorkey_v2.safetensors");
    if !ckpt_path.exists() {
        return;
    }

    let meta: Stage1Meta =
        serde_json::from_str(&fs::read_to_string(dir.join("stage1_meta.json")).unwrap()).unwrap();
    let rgb_hint = read_f32_file(&dir.join("rgb_hint_bthwc.f32"));
    let vfm_64 = read_f32_file(&dir.join("vfm_64_bthwc.f32"));
    let hq = read_f32_file(&dir.join("hint_quality_bt.f32"));
    let want_baton = read_f32_file(&dir.join("baton_bthwc_torch.f32"));
    let want_aux = read_f32_file(&dir.join("aux_bthwc_torch.f32"));

    let (b, t, h, w) = (meta.b, meta.t, meta.h, meta.w);
    let h64 = h / 64;
    let w64 = w / 64;

    let st = SafetensorsFile::open(&ckpt_path).unwrap();
    let header = st.checkpoint_header().unwrap();

    println!(
        "\n=== Stage 1 (B={b}, T={t}, {h}×{w}) Parity vs PyTorch {} on {} (PyTorch resident: {:.3} ms) ===",
        meta.torch, meta.device, meta.resident_ms
    );

    for target in ComputeTarget::available() {
        let ctx = MlContext::new(target).unwrap();
        let t_load = Instant::now();
        let stage1 = Stage1::load::<f32>(&ctx, &st, &header.model_config, "stage1").unwrap();
        let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

        let t_rgb_hint = ctx.upload_tensor(&rgb_hint, [b, t, h, w, 4]).unwrap();
        let t_vfm = ctx
            .upload_tensor(&vfm_64, [b, t, h64, w64, meta.vfm_channels])
            .unwrap();
        let t_hq = ctx.upload_tensor(&hq, [b * t, 1]).unwrap();

        let out = stage1
            .forward::<f32>(
                &ctx,
                t_rgb_hint.clone(),
                t_vfm.clone(),
                Some(t_hq.clone()),
                1.0,
            )
            .unwrap();
        let got_baton = ctx.download_tensor::<f32, 5>(out.baton).unwrap();
        let got_aux = ctx.download_tensor::<f32, 5>(out.aux).unwrap();
        assert_eq!(got_baton.len(), want_baton.len());
        assert_eq!(got_aux.len(), want_aux.len());

        let diff_stats = |got: &[f32], want: &[f32]| -> (f64, f64) {
            let mut max_abs = 0.0f64;
            let mut sum_abs = 0.0f64;
            for (&g, &r) in got.iter().zip(want) {
                let err = (g as f64 - r as f64).abs();
                max_abs = max_abs.max(err);
                sum_abs += err;
            }
            (sum_abs / (want.len() as f64), max_abs)
        };
        let (baton_mean, baton_max) = diff_stats(&got_baton, &want_baton);
        let (aux_mean, aux_max) = diff_stats(&got_aux, &want_aux);

        let res_ms = time_device(ctx.client(), 5, || {
            let baton = stage1
                .forward_baton::<f32>(
                    &ctx,
                    t_rgb_hint.clone(),
                    t_vfm.clone(),
                    Some(t_hq.clone()),
                    1.0,
                )
                .unwrap();
            let _ = corridorkey_ml::into_contiguous_cube(baton).unwrap();
        })
        .as_secs_f64()
            * 1e3;

        println!(
            "  {target:12}: load={load_ms:7.1} ms | resident={res_ms:7.3} ms ({:.2}× vs torch) | baton mean={baton_mean:.3e} max={baton_max:.3e} | aux mean={aux_mean:.3e} max={aux_max:.3e}",
            meta.resident_ms / res_ms
        );
        assert!(
            baton_max < 5e-3 && baton_mean < 5e-4 && aux_max < 5e-3 && aux_mean < 5e-4,
            "{target} Stage 1 parity exceeded tolerance: baton_max={baton_max:.3e}, aux_max={aux_max:.3e}"
        );
    }
}
