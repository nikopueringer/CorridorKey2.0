//! Stage 2 block operators at model shapes: GEMM throughput through `MlContext::matmul` and the
//! unmasked attention dispatch, per runtime. Compare with `scripts/parity/bench_stage2_ops.py`
//! (cuBLAS / PyTorch SDPA on the same shapes).
//!
//! `cargo test --release -p corridorkey-ml --test bench_stage2_ops -- --ignored --nocapture`

use corridorkey_base::core::CkFloat;
use corridorkey_core::ComputeTarget;
use corridorkey_core::compute::bench::time_device;
use corridorkey_ml::MlContext;
use burn_tensor::module::attention;
use burn_tensor::ops::AttentionModuleOptions;
use corridorkey_ml::models::blocks::attn_core::run_unmasked_sdpa;
use corridorkey_ml::models::blocks::kernels::launch_sdpa_window;
use corridorkey_ml::into_contiguous_cube;
use half::f16;

const ITERS: usize = 10;

/// Tokens of one 768×512 frame at Stage 2's /16 grid (48 × 32).
const TOKENS: usize = 1536;
const DIM: usize = 1280;
const HIDDEN: usize = 3200;
const HEADS: usize = 16;
const HEAD_DIM: usize = 80;

/// `(name, m, k, n)` of every GEMM in one Stage 2 block.
const GEMMS: [(&str, usize, usize, usize); 5] = [
    ("qkv", TOKENS, DIM, 3 * DIM),
    ("attn gate", TOKENS, DIM, HEADS),
    ("proj", TOKENS, DIM, DIM),
    ("mlp gate|up", TOKENS, DIM, 2 * HIDDEN),
    ("mlp down", TOKENS, HIDDEN, DIM),
];

fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i % 97) as f32 / 97.0 - 0.5) * scale)
        .collect()
}

/// Times every Stage 2 GEMM with `F` operands; returns the summed median milliseconds.
fn time_gemms<F: CkFloat>(ctx: &MlContext, to_f: impl Fn(f32) -> F) -> f64 {
    let mut total_ms = 0.0;
    for (name, m, k, n) in GEMMS {
        let a_host: Vec<F> = ramp(m * k, 1.0).into_iter().map(&to_f).collect();
        let b_host: Vec<F> = ramp(k * n, 0.05).into_iter().map(&to_f).collect();
        let a = ctx.upload_tensor(&a_host, [m, k]).unwrap();
        let b = ctx.upload_tensor(&b_host, [k, n]).unwrap();
        let ms = time_device(ctx.client(), ITERS, || {
            let _ = ctx.matmul(a.clone(), b.clone());
        })
        .as_secs_f64()
            * 1e3;
        total_ms += ms;
        let tflops = 2.0 * (m * k * n) as f64 / (ms * 1e-3) / 1e12;
        println!("  {name:12} [{m}×{k}]·[{k}×{n}]  {ms:7.3} ms  {tflops:5.2} TFLOP/s");
    }
    total_ms
}

/// f16 GEMM (`[m, k]·[k, n]`, f32 accumulate expected) against an f64 host reference of the same
/// f16-rounded operands; returns `max |err| / max |out|`.
fn f16_gemm_error(ctx: &MlContext, m: usize, k: usize, n: usize) -> f64 {
    let a32: Vec<f32> = ramp(m * k, 1.0)
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    let b32: Vec<f32> = ramp(k * n, 0.05)
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    let a: Vec<f16> = a32.iter().map(|&v| f16::from_f32(v)).collect();
    let b: Vec<f16> = b32.iter().map(|&v| f16::from_f32(v)).collect();
    let ta = ctx.upload_tensor(&a, [m, k]).unwrap();
    let tb = ctx.upload_tensor(&b, [k, n]).unwrap();
    let got: Vec<f16> = ctx.download_tensor(ctx.matmul(ta, tb)).unwrap();
    let (mut max_err, mut scale) = (0f64, 0f64);
    for i in 0..m {
        for j in 0..n {
            let want: f64 = (0..k)
                .map(|p| a32[i * k + p] as f64 * b32[p * n + j] as f64)
                .sum();
            max_err = max_err.max((got[i * n + j].to_f64() - want).abs());
            scale = scale.max(want.abs());
        }
    }
    max_err / scale
}

/// Attention at `[batches, HEADS, tokens, HEAD_DIM]` through our `sdpa_window_kernel` and through
/// Burn/cubek `attention`, with `F` operands: `(ours ms, cubek ms, max |ours - cubek|)`.
fn time_attention<F: CkFloat>(
    ctx: &MlContext,
    batches: usize,
    tokens: usize,
    to_f: impl Fn(f32) -> F,
) -> (f64, f64, f64) {
    let shape = [batches, HEADS, tokens, HEAD_DIM];
    let len = batches * HEADS * tokens * HEAD_DIM;
    let upload = |scale: f32| {
        let host: Vec<F> = ramp(len, scale).into_iter().map(&to_f).collect();
        ctx.upload_tensor(&host, shape).unwrap()
    };
    let (q, k, v) = (upload(1.0), upload(0.9), upload(0.8));
    let ours = || {
        let (cq, ck, cv) = (
            into_contiguous_cube(q.clone()).unwrap(),
            into_contiguous_cube(k.clone()).unwrap(),
            into_contiguous_cube(v.clone()).unwrap(),
        );
        launch_sdpa_window::<F>(ctx.client(), cq.handle, ck.handle, cv.handle, None, batches, HEADS, tokens, HEAD_DIM)
            .unwrap()
    };
    let theirs = || attention(q.clone(), k.clone(), v.clone(), None, None, AttentionModuleOptions::default());
    let ours_ms = time_device(ctx.client(), ITERS, || {
        let _ = ours();
    })
    .as_secs_f64()
        * 1e3;
    let theirs_ms = time_device(ctx.client(), ITERS, || {
        let _ = theirs();
    })
    .as_secs_f64()
        * 1e3;
    let a: Vec<F> = ctx.download_tensor(ctx.tensor_from_handle::<4>(ours(), shape, corridorkey_ml::elem_dtype::<F>())).unwrap();
    let b: Vec<F> = ctx.download_tensor(theirs()).unwrap();
    let diff = a.iter().zip(&b).fold(0f64, |m, (&x, &y)| {
        m.max((corridorkey_base::core::to_f64(x) - corridorkey_base::core::to_f64(y)).abs())
    });
    (ours_ms, theirs_ms, diff)
}

/// Stage 2 attention shapes: 6 unshifted 16×16 windows and the full 1536-token sequence.
#[test]
#[ignore = "benchmark"]
fn stage2_attention_paths() {
    for target in ComputeTarget::available() {
        let ctx = MlContext::new(target).unwrap();
        for (label, batches, tokens) in [("windows 6×256", 6, 256), ("full 1×1536", 1, TOKENS)] {
            let (o32, t32, d32) = time_attention::<f32>(&ctx, batches, tokens, |v| v);
            let (o16, t16, d16) = time_attention::<f16>(&ctx, batches, tokens, f16::from_f32);
            println!(
                "{target} {label:14} f32: ours {o32:7.3} ms, cubek {t32:7.3} ms (|Δ| {d32:.1e}) | f16: ours {o16:7.3} ms, cubek {t16:7.3} ms (|Δ| {d16:.1e})"
            );
        }
    }
}

#[test]
#[ignore = "benchmark"]
fn stage2_block_operators() {
    for target in ComputeTarget::available() {
        let ctx = MlContext::new(target).unwrap();
        println!("\n{target} f32:");
        let total_ms = time_gemms::<f32>(&ctx, |v| v);
        let shape = [1, HEADS, TOKENS, HEAD_DIM];
        let len = HEADS * TOKENS * HEAD_DIM;
        let q = ctx.upload_tensor(&ramp(len, 1.0), shape).unwrap();
        let kk = ctx.upload_tensor(&ramp(len, 0.9), shape).unwrap();
        let v = ctx.upload_tensor(&ramp(len, 0.8), shape).unwrap();
        let ms = time_device(ctx.client(), ITERS, || {
            let _ = run_unmasked_sdpa::<f32>(&ctx, q.clone(), kk.clone(), v.clone()).unwrap();
        })
        .as_secs_f64()
            * 1e3;
        let tflops = 4.0 * (HEADS * TOKENS * TOKENS * HEAD_DIM) as f64 / (ms * 1e-3) / 1e12;
        println!(
            "  {:12} [{HEADS}×{TOKENS}×{HEAD_DIM}]       {ms:7.3} ms  {tflops:5.2} TFLOP/s",
            "full sdpa"
        );
        println!("  GEMMs per block: {total_ms:.2} ms");
        println!("{target} f16:");
        let total_f16 = time_gemms::<f16>(&ctx, f16::from_f32);
        println!("  GEMMs per block: {total_f16:.2} ms");
        println!(
            "  f16 error (proj shape, odd m): {:.3e}",
            f16_gemm_error(&ctx, 1537, DIM, DIM)
        );
    }
}

/// One attention shape through `sdpa_rows_kernel`, `sdpa_window_kernel` and (unmasked) cubek flash:
/// `(rows ms, window ms, cubek ms or NaN)`.
fn time_attention_kernels<F: CkFloat>(
    ctx: &MlContext,
    (windows, heads, tokens, head_dim): (usize, usize, usize, usize),
    masked: bool,
    to_f: impl Fn(f32) -> F,
) -> (f64, f64, f64) {
    use corridorkey_base::core::buffer::upload;
    use corridorkey_ml::models::blocks::kernels::launch_sdpa_rows;
    let groups = windows * heads;
    let shape = [groups, 1, tokens, head_dim];
    let len = groups * tokens * head_dim;
    let host = |s: f32| -> Vec<F> { ramp(len, s).into_iter().map(&to_f).collect() };
    let (qh, kh, vh) = (host(1.0), host(0.9), host(0.8));
    let regions: Vec<i32> = (0..windows * tokens).map(|i| ((i / 37) % 3) as i32).collect();
    let r = upload(ctx.client(), &regions);
    let reg = masked.then_some((&r, windows));
    let (q, k, v) = (
        upload(ctx.client(), &qh),
        upload(ctx.client(), &kh),
        upload(ctx.client(), &vh),
    );
    let rows = time_device(ctx.client(), ITERS, || {
        let _ = launch_sdpa_rows::<F>(ctx.client(), q.clone(), k.clone(), v.clone(), reg, groups, heads, tokens, head_dim)
            .unwrap();
    })
    .as_secs_f64()
        * 1e3;
    let window = time_device(ctx.client(), ITERS, || {
        let _ = launch_sdpa_window::<F>(ctx.client(), q.clone(), k.clone(), v.clone(), reg, groups / heads, heads, tokens, head_dim)
            .unwrap();
    })
    .as_secs_f64()
        * 1e3;
    let cubek = if masked {
        f64::NAN
    } else {
        let t = |h: &Vec<F>| ctx.upload_tensor(h, shape).unwrap();
        let (tq, tk, tv) = (t(&qh), t(&kh), t(&vh));
        time_device(ctx.client(), ITERS, || {
            let _ = attention(tq.clone(), tk.clone(), tv.clone(), None, None, AttentionModuleOptions::default());
        })
        .as_secs_f64()
            * 1e3
    };
    (rows, window, cubek)
}

/// `sdpa_rows_kernel` against the previous kernel and cubek at the model's attention shapes.
#[test]
#[ignore = "benchmark"]
fn attention_kernels_at_model_shapes() {
    let shapes = [
        ("stage1 3D T=8", (6, 8, 2048, 64)),
        ("stage2 windows", (6, 16, 256, 80)),
        ("stage2 full", (1, 16, 1536, 80)),
    ];
    for target in ComputeTarget::available() {
        let ctx = MlContext::new(target).unwrap();
        for (label, shape) in shapes {
            for masked in [false, true] {
                let (r32, w32, c32) = time_attention_kernels::<f32>(&ctx, shape, masked, |v| v);
                let (r16, w16, c16) = time_attention_kernels::<f16>(&ctx, shape, masked, f16::from_f32);
                println!(
                    "{target} {label:15} masked={masked:5} | f32 rows {r32:8.3} window {w32:8.3} cubek {c32:8.3} | f16 rows {r16:8.3} window {w16:8.3} cubek {c16:8.3} ms"
                );
            }
        }
    }
}

/// f16 GEMM error against an f64 reference as the reduction length grows: f32 accumulation (then
/// f16 output rounding) stays flat in `k`, f16 accumulation grows with it.
#[test]
#[ignore = "benchmark"]
fn f16_gemm_error_vs_reduction_length() {
    for target in ComputeTarget::available() {
        let ctx = MlContext::new(target).unwrap();
        for k in [256, 1280, 4304, 8192] {
            println!("{target} k={k:5}: max |err| / max |out| = {:.3e}", f16_gemm_error(&ctx, 385, k, 512));
        }
    }
}
