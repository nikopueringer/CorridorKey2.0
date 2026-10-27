//! `infer` on the Rust engine: the command-line options as [`ClipOptions`], keyed by
//! `corridorkey_io`'s [`ClipKeyer`] (`runtime.run_inference`).

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};
use corridorkey_base::frame::RenderOutput;
use corridorkey_core::{ComputeTarget, LinearAdjust};
use corridorkey_io::writers::EncoderConfig;
use corridorkey_io::{ClipKeyer, ClipOptions};
use corridorkey_ml::pipeline::ModelPaths;

use super::infer::{ExrDtype, InferArgs, Output, OutputFormat, Precision, StartMode};

/// Options neither engine supports. The product refuses a model scale other than 1 too
/// ("optimized runtime render path requires native model_scale=1.0").
fn unsupported(args: &InferArgs) -> Result<()> {
    if args.frames.model_scale != 1.0 {
        bail!("--model-scale other than 1 is not supported (the PyTorch product's render path refuses it too)");
    }
    Ok(())
}

fn clip_options(args: &InferArgs) -> ClipOptions {
    let (f, h, c, e) = (&args.frames, &args.hints, &args.color, &args.encoding);
    ClipOptions {
        outputs: args
            .outputs
            .iter()
            .map(|o| match o {
                Output::Alpha => RenderOutput::Alpha,
                Output::Fg => RenderOutput::Fg,
                Output::Checker => RenderOutput::Checker,
                Output::Cutout => RenderOutput::Cutout,
            })
            .collect(),
        num_frames: f.num_frames,
        start_mode: match f.start_mode {
            StartMode::Begin => corridorkey_io::StartMode::Begin,
            StartMode::Middle => corridorkey_io::StartMode::Middle,
            StartMode::RandomMiddle => corridorkey_io::StartMode::RandomMiddle,
        },
        seed: f.seed,
        frame_dir_fps: f.frame_dir_fps,
        frame_dir_linear: f.frame_dir_linear,
        hann_chunk: f.hann_chunk,
        hann_stride: f.hann_stride,
        hint_first_frame: h.hint_first_frame.clone(),
        hint_video: h.hint_video.clone(),
        hint_quality: h.hint_quality,
        carry_hint: !h.no_carry_hint,
        carry_hint_quality: h.carry_hint_quality,
        stage2_batch: args.memory.stage2_batch,
        low_vram: args.memory.low_vram,
        low_vram_foundation_chunk: args.memory.low_vram_foundation_chunk,
        exr: (e.output_format == OutputFormat::Exr).then_some(e.exr_dtype == ExrDtype::Half),
        despill: c.despill_strength,
        cutout_linear: c.cutout_linear,
        linear: LinearAdjust { gain: c.linear_brightness, contrast: c.linear_contrast, pivot: c.linear_contrast_pivot },
        encoder: EncoderConfig {
            ffmpeg: e.ffmpeg_bin.clone(),
            codec: e.ffmpeg_codec.clone(),
            crf: e.crf,
            preset: e.preset.clone(),
            pix_fmt: e.pix_fmt.clone(),
            bitrate: e.bitrate.clone(),
            threads: e.ffmpeg_threads,
        },
        ..ClipOptions::new(&args.input, &args.output_dir)
    }
}

/// Runs `infer` on the Rust engine with the runtime `target`.
pub fn run(root: &Path, target: ComputeTarget, args: &InferArgs) -> Result<()> {
    unsupported(args)?;
    let started = Instant::now();
    let mut paths = ModelPaths::from_root(root);
    if let Some(w) = &args.weights {
        paths.checkpoint = w.clone();
    }
    let precision = match args.engine.precision {
        Precision::Auto => corridorkey_io::Precision::Auto,
        Precision::F32 => corridorkey_io::Precision::F32,
        Precision::F16 => corridorkey_io::Precision::F16,
    };
    let t = Instant::now();
    let keyer = ClipKeyer::load(target, &paths, precision)?;
    println!("  loaded networks in {:.1} s", t.elapsed().as_secs_f64());
    let report = keyer.key_clip(&clip_options(args), &mut |line| println!("  {line}"))?;
    println!("  total {:.1} s", started.elapsed().as_secs_f64());
    for p in &report.written {
        println!("    {}", p.display());
    }
    Ok(())
}
