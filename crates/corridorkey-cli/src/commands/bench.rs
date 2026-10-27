//! `bench`: times the Rust engine end to end (frame preparation, backbones, Stage 1 windows with the
//! Hann blend, Stage 2, rendering; frames in host memory, rendered outputs back in host memory, no
//! encoding) and reads its peak device memory from the driver (NVML, in process).
//!
//! The clip is the test portrait (`data/green_screen_portrait.png`) resized to `--size` and moving
//! 6 px per frame, so the networks see natural content. `--sweep` runs a grid of sizes × windows ×
//! precisions, each in its own process (CubeCL keeps its memory pools for the life of a process),
//! after a warm-up process for the same shapes so autotuning neither costs time nor raises the peak.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Args;
use corridorkey_base::frame::RenderOutput;
use corridorkey_core::{ComputeTarget, FrameMeta};
use corridorkey_io::memory::{PeakSampler, free_and_total};
use corridorkey_io::{ClipKeyer, ClipOptions, MemoryHints};
use corridorkey_ml::pipeline::ModelPaths;
use serde_json::json;

use super::infer::Precision;

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Frame size, WIDTHxHEIGHT
    #[arg(long, default_value = "768x512")]
    pub size: String,
    /// Frames keyed per run
    #[arg(long, default_value_t = 8)]
    pub frames: usize,
    /// Stage 1 window [default: --frames, one window]; the stride is half of it
    #[arg(long)]
    pub chunk: Option<usize>,
    #[arg(long, value_enum, default_value_t = Precision::Auto)]
    pub precision: Precision,
    /// Timed runs after the first (the first is reported apart: it includes one-time setup)
    #[arg(long, default_value_t = 3)]
    pub runs: usize,
    /// Backbones a few frames at a time (`infer --low-vram`)
    #[arg(long)]
    pub low_vram: bool,
    /// Append the result as one JSON line to this file
    #[arg(long)]
    pub json: Option<PathBuf>,
    /// Run a grid of --sweep-sizes × --sweep-chunks × --sweep-precisions (one process each)
    #[arg(long)]
    pub sweep: bool,
    #[arg(long, value_delimiter = ',', default_values = ["768x512", "1280x768", "1920x1080"])]
    pub sweep_sizes: Vec<String>,
    /// Windows of the sweep; each run keys exactly one window
    #[arg(long, value_delimiter = ',', default_values = ["2", "4", "8", "16"])]
    pub sweep_chunks: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_values = ["f16", "f32"])]
    pub sweep_precisions: Vec<String>,
}

fn parse_size(s: &str) -> Result<(usize, usize)> {
    let (w, h) = s.split_once('x').with_context(|| format!("size {s:?} is not WIDTHxHEIGHT"))?;
    Ok((w.parse()?, h.parse()?))
}

/// `frames` RGB frames of `(w, h)`: the portrait resized, moving 6 px per frame.
fn clip(root: &Path, (w, h): (usize, usize), frames: usize) -> Result<Vec<u8>> {
    let path = root.join("data/green_screen_portrait.png");
    let img = image::open(&path).with_context(|| format!("reading {}", path.display()))?.into_rgb8();
    let img = image::imageops::resize(&img, w as u32, h as u32, image::imageops::FilterType::Lanczos3);
    let row = w * 3;
    let mut out = Vec::with_capacity(frames * h * row);
    for t in 0..frames {
        let shift = (6 * t) % w * 3;
        for y in 0..h {
            let src = &img.as_raw()[y * row..(y + 1) * row];
            // np.roll(frame, shift, axis=1)
            out.extend_from_slice(&src[row - shift..]);
            out.extend_from_slice(&src[..row - shift]);
        }
    }
    Ok(out)
}

/// One measurement in this process.
fn bench_once(root: &Path, target: ComputeTarget, args: &BenchArgs) -> Result<serde_json::Value> {
    let (w, h) = parse_size(&args.size)?;
    let chunk = args.chunk.unwrap_or(args.frames).clamp(1, args.frames.max(1));
    let rgb = clip(root, (w, h), args.frames)?;
    let free_before = free_and_total();
    let sampler = PeakSampler::start(Duration::from_millis(10));

    let precision = match args.precision {
        Precision::Auto => corridorkey_io::Precision::Auto,
        Precision::F32 => corridorkey_io::Precision::F32,
        Precision::F16 => corridorkey_io::Precision::F16,
    };
    let t = Instant::now();
    let keyer = ClipKeyer::load(target, &ModelPaths::from_root(root), precision)?;
    let load_s = t.elapsed().as_secs_f64();
    let opts = ClipOptions {
        outputs: vec![RenderOutput::Alpha, RenderOutput::Fg],
        hann_chunk: Some(chunk),
        hann_stride: Some((chunk / 2).max(1)),
        low_vram: args.low_vram,
        ..ClipOptions::new("", "")
    };
    let mut times = vec![];
    for _ in 0..=args.runs {
        let t = Instant::now();
        keyer.key_frames(&rgb, args.frames, (h, w), MemoryHints::default(), &opts)?;
        times.push(t.elapsed().as_secs_f64());
    }
    let peak = sampler.and_then(PeakSampler::stop);
    let meta = FrameMeta::new((h, w), 1.0, corridorkey_core::frame::SHAPE_QUANTUM, 24.0, args.frames)?;
    let (ph, pw) = meta.shape_hw;
    let steady = &times[1..];
    let best = steady.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = steady.iter().sum::<f64>() / steady.len().max(1) as f64;
    Ok(json!({
        "engine": "rust",
        "runtime": keyer.target().name(),
        "precision": match keyer.precision() { corridorkey_io::Precision::F16 => "f16", _ => "f32" },
        "width": w, "height": h, "padded_width": pw, "padded_height": ph,
        "megapixels": (pw * ph) as f64 / 1e6,
        "frames": args.frames, "chunk": chunk, "stride": (chunk / 2).max(1), "low_vram": args.low_vram,
        "load_s": load_s,
        "first_run_s": times[0],
        "runs_s": steady,
        "best_s": best,
        "mean_s": mean,
        "fps": args.frames as f64 / best,
        "peak_vram_bytes": peak,
        "free_before_bytes": free_before.map(|(f, _)| f),
        "total_bytes": free_before.map(|(_, t)| t),
    }))
}

fn append_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).with_context(|| format!("opening {}", path.display()))?;
    writeln!(f, "{value}")?;
    Ok(())
}

/// One child process of the sweep; `Ok(false)` when it failed (e.g. out of memory).
#[allow(clippy::too_many_arguments)]
fn run_child(root: &Path, target: ComputeTarget, size: &str, chunk: usize, precision: &str, runs: usize, low_vram: bool, json: Option<&Path>) -> Result<bool> {
    let mut cmd = Command::new(std::env::current_exe()?);
    cmd.args(["--root", &root.display().to_string(), "--runtime", target.name(), "bench"])
        .args(["--size", size, "--frames", &chunk.to_string(), "--chunk", &chunk.to_string()])
        .args(["--precision", precision, "--runs", &runs.to_string()]);
    if low_vram {
        cmd.arg("--low-vram");
    }
    if let Some(j) = json {
        cmd.arg("--json").arg(j);
    }
    let out = cmd.output()?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let last = err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
        println!("    failed: {last}");
        if let Some(j) = json {
            append_json(j, &json!({"engine": "rust", "runtime": target.name(), "precision": precision, "size": size, "chunk": chunk, "error": last}))?;
        }
        return Ok(false);
    }
    print!("{}", String::from_utf8_lossy(&out.stdout));
    Ok(true)
}

fn sweep(root: &Path, target: ComputeTarget, args: &BenchArgs) -> Result<()> {
    let json = args.json.as_deref().context("--sweep needs --json")?;
    for precision in &args.sweep_precisions {
        for size in &args.sweep_sizes {
            for &chunk in &args.sweep_chunks {
                println!("  {} {precision} {size} window {chunk}", target.name());
                // Warm-up: tunes and caches every shape of this configuration
                if !run_child(root, target, size, chunk, precision, 0, args.low_vram, None)? {
                    break; // larger windows of this size will not fit either
                }
                if !run_child(root, target, size, chunk, precision, args.runs, args.low_vram, Some(json))? {
                    break;
                }
            }
        }
    }
    Ok(())
}

pub fn run(root: &Path, target: ComputeTarget, args: &BenchArgs) -> Result<()> {
    if args.frames == 0 {
        bail!("--frames must be at least 1");
    }
    if args.sweep {
        return sweep(root, target, args);
    }
    let result = bench_once(root, target, args)?;
    let gb = |v: &serde_json::Value| v.as_f64().map_or("n/a".into(), |b| format!("{:.2} GB", b / 2f64.powi(30)));
    println!(
        "    {} {} {} x{} window {}: best {:.3} s ({:.2} frames/s), first {:.2} s, load {:.1} s, peak {}",
        result["runtime"].as_str().unwrap_or("?"),
        result["precision"].as_str().unwrap_or("?"),
        args.size,
        args.frames,
        result["chunk"],
        result["best_s"].as_f64().unwrap_or(f64::NAN),
        result["fps"].as_f64().unwrap_or(f64::NAN),
        result["first_run_s"].as_f64().unwrap_or(f64::NAN),
        result["load_s"].as_f64().unwrap_or(f64::NAN),
        gb(&result["peak_vram_bytes"])
    );
    if let Some(j) = &args.json {
        append_json(j, &result)?;
    }
    Ok(())
}
