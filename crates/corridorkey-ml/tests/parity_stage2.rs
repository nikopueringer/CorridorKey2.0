//! Stage 2 end-to-end parity and resident timing comparison against PyTorch CUDA
//! (`scripts/parity/stage2_reference.py`).

use std::{fs, path::Path, time::Instant};

use corridorkey_core::{ComputeTarget, compute::bench::time_device};
use corridorkey_ml::{MlContext, STAGE2_PATCH_SIZE, SafetensorsFile, Stage2};
use serde::Deserialize;

#[derive(Deserialize)]
struct Stage2Meta {
    device: String,
    torch: String,
    h: usize,
    w: usize,
    d_baton: usize,
    resident_ms: f64,
}

fn read_f32_file(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn compare_stage2_with_python() {
    let dir = Path::new("/tmp/corridorkey-parity-stage2");
    if !dir.join("stage2_meta.json").exists() {
        eprintln!("skipping compare_stage2_with_python: /tmp/corridorkey-parity-stage2 not generated");
        return;
    }
    let ckpt_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/corridorkey_v2.safetensors");
    if !ckpt_path.exists() {
        return;
    }

    let meta: Stage2Meta =
        serde_json::from_str(&fs::read_to_string(dir.join("stage2_meta.json")).unwrap()).unwrap();
    let rgb = read_f32_file(&dir.join("rgb_nhwc.f32"));
    let baton = read_f32_file(&dir.join("baton_nhwc.f32"));
    let want = read_f32_file(&dir.join("out_nhwc_torch.f32"));

    let h = meta.h;
    let w = meta.w;
    let h16 = h / STAGE2_PATCH_SIZE;
    let w16 = w / STAGE2_PATCH_SIZE;

    let st = SafetensorsFile::open(&ckpt_path).unwrap();
    let header = st.checkpoint_header().unwrap();

    println!(
        "\n=== Stage 2 ({h}×{w}) Parity vs PyTorch {} on {} (PyTorch resident: {:.3} ms) ===",
        meta.torch, meta.device, meta.resident_ms
    );

    for target in ComputeTarget::available() {
        let ctx = MlContext::new(target).unwrap();
        let t_load = Instant::now();
        let stage2 = Stage2::load::<f32>(&ctx, &st, &header.model_config, "stage2").unwrap();
        let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

        let t_rgb = ctx.upload_tensor(&rgb, [1, h, w, 3]).unwrap();
        let t_baton = ctx.upload_tensor(&baton, [1, h16, w16, meta.d_baton]).unwrap();

        // Warmup + parity check
        let t_out = stage2.forward::<f32>(&ctx, t_rgb.clone(), t_baton.clone()).unwrap();
        let got = ctx.download_tensor::<f32, 4>(t_out).unwrap();
        assert_eq!(got.len(), want.len());

        let mut max_abs = 0.0f64;
        let mut sum_abs = 0.0f64;
        for (&g, &r) in got.iter().zip(&want) {
            let err = (g as f64 - r as f64).abs();
            max_abs = max_abs.max(err);
            sum_abs += err;
        }
        let mean_abs = sum_abs / (want.len() as f64);

        // Per-component breakdown (single synced pass after warmup)
        let step_ms = |f: &mut dyn FnMut()| -> f64 {
            time_device(ctx.client(), 3, f).as_secs_f64() * 1e3
        };
        let p1_ms = step_ms(&mut || {
            let rf = stage2.patch_1.forward::<f32>(&ctx, t_rgb.clone()).unwrap();
            let _ = corridorkey_ml::into_contiguous_cube(rf).unwrap();
        });
        let rgb_feat = stage2.patch_1.forward::<f32>(&ctx, t_rgb.clone()).unwrap();
        let mut x = burn_tensor::Tensor::cat(vec![t_baton.clone(), rgb_feat], 3);

        // Sub-block breakdown of pre_blocks[0] (FullSpatialBlock2d)
        if let corridorkey_ml::Stage2Block2d::Full(b0) = &stage2.pre_blocks[0] {
            let n1_ms = step_ms(&mut || {
                let y = b0.norm1.forward::<f32, 4>(&ctx, x.clone()).unwrap();
                let _ = corridorkey_ml::into_contiguous_cube(y).unwrap();
            });
            let u1 = b0.norm1.forward::<f32, 4>(&ctx, x.clone()).unwrap();
            let qkv_ms = step_ms(&mut || {
                let (y, _) = b0.attn.core.project_qkv_spatial::<f32>(&ctx, u1.clone()).unwrap();
                let _ = corridorkey_ml::into_contiguous_cube(y).unwrap();
            });
            let (qkv_4d, bias_done) = b0.attn.core.project_qkv_spatial::<f32>(&ctx, u1.clone()).unwrap();
            let c_qkv = corridorkey_ml::into_contiguous_cube(qkv_4d).unwrap();
            let (cos_h, sin_h) = b0.attn.rope.cos_sin_handles::<f32>(&ctx, h16, w16);
            let (q, k, v) = b0.attn.core.prep_qkv_norm_rope::<f32>(
                &ctx,
                c_qkv.handle,
                &cos_h,
                &sin_h,
                1,
                h16 * w16,
                bias_done,
            );
            let sdpa_ms = step_ms(&mut || {
                let y = corridorkey_ml::models::blocks::attn_core::run_unmasked_sdpa::<f32>(
                    &ctx,
                    q.clone(),
                    k.clone(),
                    v.clone(),
                )
                .unwrap();
                let _ = corridorkey_ml::into_contiguous_cube(y).unwrap();
            });
            let attn_ms = step_ms(&mut || {
                let y = b0.attn.forward::<f32>(&ctx, u1.clone()).unwrap();
                let _ = corridorkey_ml::into_contiguous_cube(y).unwrap();
            });
            let lm_ms = if let Some(lm) = &b0.local_mix {
                step_ms(&mut || {
                    let y = lm.forward_residual::<f32, 4>(&ctx, u1.clone()).unwrap();
                    let _ = corridorkey_ml::into_contiguous_cube(y).unwrap();
                })
            } else {
                0.0
            };
            let mlp_ms = step_ms(&mut || {
                let y = b0.mlp.forward::<f32, 4>(&ctx, u1.clone()).unwrap();
                let _ = corridorkey_ml::into_contiguous_cube(y).unwrap();
            });
            println!(
                "    pre[0] sub-ops: norm1={n1_ms:.2} ms | attn={attn_ms:.2} ms (qkv+dw={qkv_ms:.2}, sdpa={sdpa_ms:.2}) | local_mix={lm_ms:.2} ms | mlp={mlp_ms:.2} ms"
            );
        }
        let mut blk_times = Vec::new();
        for (bi, blk) in stage2.pre_blocks.iter().enumerate() {
            let x_in = x.clone();
            let ms = step_ms(&mut || {
                let y = blk.forward::<f32>(&ctx, x_in.clone()).unwrap();
                let _ = corridorkey_ml::into_contiguous_cube(y).unwrap();
            });
            let kind = match blk {
                corridorkey_ml::Stage2Block2d::Full(_) => "full",
                corridorkey_ml::Stage2Block2d::Swin(_) => "swin",
                corridorkey_ml::Stage2Block2d::CSwin(_) => "cswin",
            };
            blk_times.push(format!("pre[{bi}:{kind}]={ms:.1}"));
            x = blk.forward::<f32>(&ctx, x).unwrap();
        }
        let skip_ms = if let Some(skip) = &stage2.patch_2_skip {
            let x_in = x.clone();
            let ms = step_ms(&mut || {
                let y = skip.forward::<f32>(&ctx, x_in.clone(), t_rgb.clone()).unwrap();
                let _ = corridorkey_ml::into_contiguous_cube(y).unwrap();
            });
            x = skip.forward::<f32>(&ctx, x, t_rgb.clone()).unwrap();
            ms
        } else {
            0.0
        };
        for (bi, blk) in stage2.final_blocks.iter().enumerate() {
            let x_in = x.clone();
            let ms = step_ms(&mut || {
                let y = blk.forward::<f32>(&ctx, x_in.clone()).unwrap();
                let _ = corridorkey_ml::into_contiguous_cube(y).unwrap();
            });
            blk_times.push(format!("fin[{bi}:swin]={ms:.1}"));
            x = blk.forward::<f32>(&ctx, x).unwrap();
        }
        let dep_ms = step_ms(&mut || {
            let y = stage2.depatchify.forward::<f32>(&ctx, x.clone()).unwrap();
            let _ = corridorkey_ml::into_contiguous_cube(y).unwrap();
        });

        // Resident timing (5 iterations)
        let res_ms = time_device(ctx.client(), 5, || {
            let out = stage2.forward::<f32>(&ctx, t_rgb.clone(), t_baton.clone()).unwrap();
            let _ = corridorkey_ml::into_contiguous_cube(out).unwrap();
        })
        .as_secs_f64()
            * 1e3;
        println!(
            "  {target:12}: load={load_ms:7.1} ms | resident={res_ms:7.3} ms ({:.2}× vs torch) | mean_abs={mean_abs:.3e} | max_abs={max_abs:.3e}",
            meta.resident_ms / res_ms
        );
        println!(
            "    breakdown: patch_1={p1_ms:.1} ms | patch_2_skip={skip_ms:.1} ms | depatchify={dep_ms:.1} ms | {}",
            blk_times.join(", ")
        );
        assert!(
            max_abs < 5e-3 && mean_abs < 5e-4,
            "{target} Stage 2 parity exceeded tolerance: mean_abs={mean_abs:.3e}, max_abs={max_abs:.3e}"
        );
    }
}
