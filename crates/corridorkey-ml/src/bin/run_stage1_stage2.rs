use std::{fs, path::Path, time::Instant};

use corridorkey_base::core::{CkFloat, cast_f32, to_f64};
use corridorkey_core::{ComputeTarget, compute::bench::time_device};
use burn_tensor::DType;
use corridorkey_ml::{MlContext, SafetensorsFile, Stage1, Stage2};
use half::f16;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct RunMeta {
    b: usize,
    t: usize,
    h: usize,
    w: usize,
    vfm_channels: usize,
}

#[derive(Serialize)]
struct Timing {
    target: String,
    dtype: String,
    load_ms: f64,
    cold_ms: f64,
    warm_stage1_ms: f64,
    warm_stage2_ms: f64,
    warm_total_ms: f64,
}

fn read_f32_file(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: run_stage1_stage2 <in_dir> <out_dir> [--target <cuda|wgpu>] [--dtype <auto|f32|f16>]"
        );
        std::process::exit(1);
    }
    let in_dir = Path::new(&args[1]);
    let out_dir = Path::new(&args[2]);
    fs::create_dir_all(out_dir).unwrap();

    let target_str = args
        .iter()
        .position(|a| a == "--target" || a == "--runtime")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    let target = match target_str {
        Some(name) => ComputeTarget::parse(name).expect("valid target"),
        None => ComputeTarget::from_env().unwrap_or(ComputeTarget::Cuda),
    };
    let use_flex32 = args.iter().any(|a| a == "--flex32");
    let ctx = MlContext::new(target).unwrap().with_flex32_matmul(use_flex32);
    // `auto` (default): the device's fastest supported precision (`MlContext::preferred_dtype`).
    let requested = args
        .iter()
        .position(|a| a == "--dtype")
        .and_then(|i| args.get(i + 1))
        .map_or("auto", |s| s.as_str());
    let dtype = match requested {
        "auto" => match ctx.preferred_dtype() {
            DType::F16 => "f16",
            _ => "f32",
        },
        other => other,
    };
    println!(
        "Running CorridorKey Rust inference on target: {target:?} (dtype={dtype}, requested {requested}, flex32={use_flex32})"
    );
    match dtype {
        "f32" => run::<f32>(in_dir, out_dir, &ctx, &args),
        "f16" => run::<f16>(in_dir, out_dir, &ctx, &args),
        other => panic!("unsupported --dtype {other} (expected auto, f32 or f16)"),
    }
    // Exit cleanly without waiting for thousands of tensor pool handle drops
    std::process::exit(0);
}

/// Loads both stages as `F`, runs them on the staged inputs (converted to `F` once) and writes the
/// logits as f32.
fn run<F: CkFloat>(in_dir: &Path, out_dir: &Path, ctx: &MlContext, args: &[String]) {
    let target = ctx.target();
    let meta: RunMeta =
        serde_json::from_str(&fs::read_to_string(in_dir.join("meta.json")).unwrap()).unwrap();
    let (b, t, h, w) = (meta.b, meta.t, meta.h, meta.w);
    let bt = b * t;
    let h64 = h / 64;
    let w64 = w / 64;
    let h16 = h / 16;
    let w16 = w / 16;

    let rgb_hint: Vec<F> = cast_f32(&read_f32_file(&in_dir.join("rgb_hint_bthwc.f32")));
    let rgb: Vec<F> = cast_f32(&read_f32_file(&in_dir.join("rgb_bthwc.f32")));
    let vfm_64: Vec<F> = cast_f32(&read_f32_file(&in_dir.join("vfm_64_bthwc.f32")));
    let hq: Vec<F> = cast_f32(&read_f32_file(&in_dir.join("hint_quality_bt.f32")));

    let ckpt_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/corridorkey_v2.safetensors");
    let st = SafetensorsFile::open(&ckpt_path).unwrap();
    let header = st.checkpoint_header().unwrap();

    let t0 = Instant::now();
    let stage1 = Stage1::load::<F>(&ctx, &st, &header.model_config, "stage1").unwrap();
    let stage2 = Stage2::load::<F>(&ctx, &st, &header.model_config, "stage2").unwrap();
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("Loaded weights in {load_ms:.2} ms");

    let t_rgb_hint = ctx.upload_tensor(&rgb_hint, [b, t, h, w, 4]).unwrap();
    let t_rgb = ctx.upload_tensor(&rgb, [bt, h, w, 3]).unwrap();
    let t_vfm = ctx
        .upload_tensor(&vfm_64, [b, t, h64, w64, meta.vfm_channels])
        .unwrap();
    let t_hq = ctx.upload_tensor(&hq, [bt, 1]).unwrap();

    // 1. Cold pass (JIT compiles all kernels)
    println!("Running cold execution (kernel compilation)...");
    let t_cold = Instant::now();
    let cold_baton = stage1
        .forward_baton::<F>(
            &ctx,
            t_rgb_hint.clone(),
            t_vfm.clone(),
            Some(t_hq.clone()),
            1.0,
        )
        .unwrap();
    let cold_baton_nhwc = cold_baton.reshape([bt, h16, w16, stage1.d_16]);
    let cold_logits = stage2
        .forward::<F>(&ctx, t_rgb.clone(), cold_baton_nhwc)
        .unwrap();
    let _ = corridorkey_ml::into_contiguous_cube(cold_logits).unwrap();
    let cold_ms = t_cold.elapsed().as_secs_f64() * 1e3;
    println!("Cold execution (with kernel compile): {cold_ms:.2} ms");

    // 2. Warm Stage 1
    println!("Measuring warm Stage 1 latency (median of 5 runs)...");
    let mut baton_ref = None;
    let warm_s1_dur = time_device(ctx.client(), 5, || {
        let b = stage1
            .forward_baton::<F>(
                &ctx,
                t_rgb_hint.clone(),
                t_vfm.clone(),
                Some(t_hq.clone()),
                1.0,
            )
            .unwrap();
        let _ = corridorkey_ml::into_contiguous_cube(b.clone()).unwrap();
        baton_ref = Some(b);
    });
    let warm_stage1_ms = warm_s1_dur.as_secs_f64() * 1e3;
    println!("Warm Stage 1: {warm_stage1_ms:.2} ms");

    // 3. Warm Stage 2
    println!("Measuring warm Stage 2 latency (median of 5 runs)...");
    let baton_nhwc = baton_ref.unwrap().reshape([bt, h16, w16, stage1.d_16]);
    let mut final_logits = None;
    let warm_s2_dur = time_device(ctx.client(), 5, || {
        let l = stage2
            .forward::<F>(&ctx, t_rgb.clone(), baton_nhwc.clone())
            .unwrap();
        let _ = corridorkey_ml::into_contiguous_cube(l.clone()).unwrap();
        final_logits = Some(l);
    });
    let warm_stage2_ms = warm_s2_dur.as_secs_f64() * 1e3;
    let warm_total_ms = warm_stage1_ms + warm_stage2_ms;
    println!("Warm Stage 2: {warm_stage2_ms:.2} ms");
    println!("Warm Total (Stage 1 + Stage 2): {warm_total_ms:.2} ms");

    if args.iter().any(|a| a == "--profile") {
        println!("\n=== Internal Layer Profiling Breakdown ===");
        stage1
            .profile_breakdown::<F>(
                &ctx,
                t_rgb_hint.clone(),
                t_vfm.clone(),
                Some(t_hq.clone()),
                1.0,
            )
            .unwrap();
        stage2
            .profile_breakdown::<F>(&ctx, t_rgb.clone(), baton_nhwc.clone())
            .unwrap();
        println!("==========================================\n");
    }

    if args.iter().any(|a| a == "--profile-ops") {
        const RUNS: usize = 3;
        let pctx = ctx.clone().with_op_profile(true);
        for _ in 0..RUNS {
            let l = stage2.forward::<F>(&pctx, t_rgb.clone(), baton_nhwc.clone()).unwrap();
            let _ = corridorkey_ml::into_contiguous_cube(l).unwrap();
        }
        println!("\n=== Stage 2 ops (synced spans, per forward) ===");
        for (name, total, calls) in pctx.take_op_profile() {
            let per_run = total.as_secs_f64() * 1e3 / RUNS as f64;
            println!("    {name:28} {per_run:8.2} ms  ({} calls)", calls / RUNS);
        }
    }

    // 4. Download and save outputs
    let logits = final_logits.unwrap();
    let dl_logits: Vec<f32> = ctx
        .download_tensor::<F, 4>(logits)
        .unwrap()
        .into_iter()
        .map(|v| to_f64(v) as f32)
        .collect();
    let raw_bytes: &[u8] = bytemuck::cast_slice(&dl_logits);
    fs::write(out_dir.join("rust_logits_bthwc.f32"), raw_bytes).unwrap();
    println!("Wrote rust_logits_bthwc.f32 ({} elements)", dl_logits.len());

    let timing = Timing {
        target: format!("{target:?}"),
        dtype: if size_of::<F>() == 2 { "f16" } else { "f32" }.to_string(),
        load_ms,
        cold_ms,
        warm_stage1_ms,
        warm_stage2_ms,
        warm_total_ms,
    };
    fs::write(
        out_dir.join("rust_timing.json"),
        serde_json::to_string_pretty(&timing).unwrap(),
    )
    .unwrap();
    println!("Wrote rust_timing.json");
}
