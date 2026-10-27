//! Keying a whole clip (`runtime.run_inference`): decode the input, run the pipeline, encode the
//! outputs, write the settings log. [`ClipKeyer`] loads the networks once and keys any number of
//! clips (from files with [`ClipKeyer::key_clip`], from memory with [`ClipKeyer::key_frames`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use burn_tensor::DType;
use corridorkey_base::frame::RenderOutput;
use corridorkey_core::{CkResult, ComputeTarget, FrameMeta, LinearAdjust};
use corridorkey_ml::MlContext;
use corridorkey_ml::pipeline::{FrameReader, FrameSink, HintReader, Hints, Keyer, ModelPaths, PipelineConfig, RenderedFrames};
use half::f16;

use crate::hints::{load_first_frame_hint, open_hint_reader};
use crate::images::{ImageDirReader, frame_paths};
use crate::numpy_rng::Pcg64;
use crate::video::{VideoReader, ffprobe_for, probe};
use crate::writers::{EncoderConfig, OutputSink};
use crate::{OpenedInput, clip_name};

/// `hann_chunk` without one given (`InferenceConfig` default; the product can instead pick it
/// from free GPU memory).
pub const DEFAULT_CHUNK: usize = 80;

/// Where the processed frames start when fewer than the clip are asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartMode {
    Begin,
    Middle,
    RandomMiddle,
}

/// Compute precision of the networks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// The device's fastest supported type (f16 on GPUs that support it)
    Auto,
    F32,
    F16,
}

/// Everything `infer` takes besides the networks, with `infer.py`'s defaults.
#[derive(Debug, Clone)]
pub struct ClipOptions {
    pub input: PathBuf,
    pub output_dir: PathBuf,
    pub outputs: Vec<RenderOutput>,
    /// Frames to process (all when `None`).
    pub num_frames: Option<usize>,
    pub start_mode: StartMode,
    /// Seed of [`StartMode::RandomMiddle`].
    pub seed: u64,
    /// Frame rate of a frame-directory input.
    pub frame_dir_fps: f64,
    /// Frame-directory values are scene-linear (`--frame-dir-linear`).
    pub frame_dir_linear: bool,
    /// Frames per Stage 1 window ([`DEFAULT_CHUNK`] when `None`).
    pub hann_chunk: Option<usize>,
    /// Frames between Stage 1 windows (half the chunk when `None`).
    pub hann_stride: Option<usize>,
    /// Matte image (or directory: its first frame) guiding frame 0 (`--hint-first-frame`).
    pub hint_first_frame: Option<PathBuf>,
    /// Hint video or frame directory for every frame (`--hint-video`).
    pub hint_video: Option<PathBuf>,
    /// Quality (trust) of both external hints.
    pub hint_quality: f64,
    pub carry_hint: bool,
    pub carry_hint_quality: f64,
    pub stage2_batch: usize,
    /// Staged low-memory inference (`--low-vram`): backbones run `low_vram_foundation_chunk`
    /// frames at a time; same result, lower peak memory.
    pub low_vram: bool,
    pub low_vram_foundation_chunk: usize,
    /// EXR sequences instead of videos: `Some(true)` half, `Some(false)` float.
    pub exr: Option<bool>,
    pub despill: f64,
    pub cutout_linear: bool,
    /// Applied to video inputs only (frame directories ignore it, like the product).
    pub linear: LinearAdjust,
    pub encoder: EncoderConfig,
}

impl ClipOptions {
    pub fn new(input: impl Into<PathBuf>, output_dir: impl Into<PathBuf>) -> Self {
        Self {
            input: input.into(),
            output_dir: output_dir.into(),
            outputs: vec![RenderOutput::Alpha, RenderOutput::Fg],
            num_frames: None,
            start_mode: StartMode::Begin,
            seed: 0,
            frame_dir_fps: 24.0,
            frame_dir_linear: false,
            hann_chunk: None,
            hann_stride: None,
            hint_first_frame: None,
            hint_video: None,
            hint_quality: 0.95,
            carry_hint: true,
            carry_hint_quality: 0.95,
            stage2_batch: 1,
            low_vram: false,
            low_vram_foundation_chunk: 4,
            exr: None,
            despill: 0.5,
            cutout_linear: false,
            linear: LinearAdjust::default(),
            encoder: EncoderConfig::default(),
        }
    }

    /// The pipeline settings for a clip of `frames` frames.
    fn pipeline(&self, frames: usize, exr: bool) -> PipelineConfig {
        let chunk = self.hann_chunk.unwrap_or(DEFAULT_CHUNK).min(frames).max(1);
        let stride = self.hann_stride.unwrap_or((chunk / 2).max(1)).clamp(1, chunk);
        PipelineConfig {
            chunk,
            stride,
            carry_hint: self.carry_hint,
            carry_hint_quality: self.carry_hint_quality,
            stage2_batch: self.stage2_batch.max(1),
            outputs: self.outputs.clone(),
            exr,
            despill: self.despill,
            cutout_linear: self.cutout_linear,
            product_rounding: true,
            foundation_chunk: self.low_vram.then_some(self.low_vram_foundation_chunk.max(1)),
        }
    }
}

/// What [`ClipKeyer::key_clip`] wrote.
#[derive(Debug, Clone)]
pub struct ClipReport {
    pub clip: String,
    pub frames: usize,
    /// Native `(width, height)`.
    pub size: (usize, usize),
    pub fps: f64,
    /// One video file or EXR directory per output, in the order asked for.
    pub written: Vec<PathBuf>,
    /// Inference and encoding, without decoding setup or network loading.
    pub seconds: f64,
}

/// Outputs of [`ClipKeyer::key_frames`]: per output, `frames × height × width × channels` f32
/// values in R G B (A) order (alpha alone: one channel), the values the EXR outputs hold
/// (scene-linear, unclamped).
pub type KeyedFrames = HashMap<RenderOutput, Vec<f32>>;

/// The networks on one device, ready to key clips.
pub struct ClipKeyer {
    ctx: MlContext,
    keyer: Keyer,
    /// `F16` or `F32`, never `Auto`.
    precision: Precision,
    quantum: usize,
    /// Device memory available to this keyer: free memory before the networks were loaded (every
    /// later allocation is ours), else the device's capacity. `None` when neither is known.
    budget_bytes: Option<u64>,
}

impl ClipKeyer {
    /// Loads the networks of `paths` on `target` at `precision`.
    pub fn load(target: ComputeTarget, paths: &ModelPaths, precision: Precision) -> Result<Self> {
        let quantum = corridorkey_ml::SafetensorsFile::open(&paths.checkpoint)?.checkpoint_header()?.metadata.shape_quantum;
        let free_before = crate::memory::free_and_total().map(|(free, _)| free);
        let ctx = MlContext::new(target)?;
        let budget_bytes = free_before.or_else(|| ctx.client().properties().memory.max_memory());
        let precision = match precision {
            Precision::Auto => match ctx.preferred_dtype() {
                DType::F16 => Precision::F16,
                _ => Precision::F32,
            },
            p => p,
        };
        let loaded = match precision {
            Precision::F16 => Keyer::load::<f16>(&ctx, paths),
            _ => Keyer::load::<f32>(&ctx, paths),
        };
        let keyer = loaded.context("loading the networks (did you run scripts/convert_foundations.py?)")?;
        Ok(Self { ctx, keyer, precision, quantum, budget_bytes })
    }

    /// `F16` or `F32`: what [`Precision::Auto`] resolved to.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub fn target(&self) -> ComputeTarget {
        self.ctx.target()
    }

    /// Device memory this keyer may use, in bytes (see the field).
    pub fn budget_bytes(&self) -> Option<u64> {
        self.budget_bytes
    }

    /// The Stage 1 window of a clip of `meta`: `opts`' when given, else the largest the memory
    /// budget fits (`apply_auto_vram`, with this engine's own peak model), else [`DEFAULT_CHUNK`].
    /// Returns the pipeline settings and a log line when chosen automatically.
    ///
    /// # Errors
    /// A window given in `opts` whose estimated peak exceeds the budget. Such a run must not start:
    /// when device memory runs out, CubeCL skips the work it cannot allocate for, and in this
    /// pipeline the failure does not always reach the read (2026-10-27, CubeCL 0.11.3: a 1080p
    /// window of 32 returned an alpha of zeros without an error).
    fn pipeline_for(&self, opts: &ClipOptions, meta: &FrameMeta, exr: bool) -> Result<(PipelineConfig, Option<String>)> {
        let (h, w) = meta.shape_hw;
        let megapixels = (h * w) as f64 / 1e6;
        let auto = opts.hann_chunk.is_none() && opts.hann_stride.is_none();
        let Some(budget) = self.budget_bytes else {
            return Ok((opts.pipeline(meta.num_frames, exr), None));
        };
        let budget_gb = budget as f64 / 2f64.powi(30);
        if !auto {
            let cfg = opts.pipeline(meta.num_frames, exr);
            let model = if self.precision == Precision::F16 { crate::vram::F16_PEAK } else { crate::vram::F32_PEAK };
            let estimate = model.estimate_gb(megapixels, cfg.chunk);
            if estimate > budget_gb {
                let fits = crate::vram::select_window(megapixels, meta.num_frames, budget_gb, self.precision);
                bail!(
                    "a Stage 1 window of {} frames at {w}x{h} needs an estimated {estimate:.1} GB of device memory, {budget_gb:.1} GB available; \
                     use --hann-chunk {} or less (or leave it out to choose automatically)",
                    cfg.chunk,
                    fits.chunk
                );
            }
            return Ok((cfg, None));
        }
        let choice = crate::vram::select_window(megapixels, meta.num_frames, budget_gb, self.precision);
        let mut cfg = opts.pipeline(meta.num_frames, exr);
        cfg.chunk = choice.chunk.min(meta.num_frames).max(1);
        cfg.stride = choice.stride.clamp(1, cfg.chunk);
        let line = format!(
            "auto window: {budget_gb:.1} GB available at {w}x{h} -> hann_chunk={}, hann_stride={} (estimated peak {:.1} GB)",
            cfg.chunk, cfg.stride, choice.estimate_gb
        );
        if !choice.fits {
            bail!("{line}: even a single-frame window does not fit the available device memory");
        }
        Ok((cfg, Some(line)))
    }

    fn run(&self, meta: &FrameMeta, reader: &mut dyn FrameReader, hints: &mut Hints<'_>, sink: &mut dyn FrameSink, cfg: &PipelineConfig) -> CkResult<()> {
        match self.precision {
            Precision::F16 => self.keyer.run_with_hints::<f16>(&self.ctx, meta, reader, hints, sink, cfg),
            _ => self.keyer.run_with_hints::<f32>(&self.ctx, meta, reader, hints, sink, cfg),
        }
    }

    /// Keys the clip `opts.input` (video file or frame directory) into `opts.output_dir`, named
    /// like the product. `log` receives progress lines.
    pub fn key_clip(&self, opts: &ClipOptions, log: &mut dyn FnMut(&str)) -> Result<ClipReport> {
        if opts.outputs.is_empty() {
            bail!("no outputs asked for");
        }
        let mut input = open_input(opts)?;
        let mut meta = FrameMeta::new(input.native_hw, 1.0, self.quantum, input.fps, input.frames)?;
        if !input.is_frame_dir {
            meta.linear_adjust = opts.linear;
        }
        let (h0, w0) = meta.native_hw;
        log(&format!(
            "{} frames from frame {} @ {}x{} (native {w0}x{h0}, fps {:.2}) on {} ({:?})",
            meta.num_frames,
            input.start_frame,
            meta.shape_hw.1,
            meta.shape_hw.0,
            meta.fps,
            self.target(),
            self.precision
        ));
        let (cfg, auto_line) = self.pipeline_for(opts, &meta, opts.exr.is_some())?;
        if let Some(line) = auto_line {
            log(&line);
        }
        let clip = clip_name(&opts.input);
        let (mut sink, written) =
            OutputSink::open(&opts.output_dir, &clip, &opts.outputs, (w0, h0), meta.fps, cfg.despill, opts.exr, cfg.cutout_linear, &opts.encoder)?;
        write_settings(&opts.output_dir.join(format!("{clip}_settings.txt")), opts, &meta, &cfg)?;

        let mut video_hints = match &opts.hint_video {
            Some(p) => Some(open_hint_reader(p, &opts.encoder.ffmpeg, meta.native_hw, input.start_frame, meta.num_frames)?),
            None => None,
        };
        let mut hints = Hints {
            first_frame: opts.hint_first_frame.as_deref().map(|p| load_first_frame_hint(p, meta.native_hw)).transpose()?,
            first_frame_quality: opts.hint_quality,
            video: video_hints.as_deref_mut().map(|r| r as &mut dyn HintReader),
            video_quality: opts.hint_quality,
        };
        let t = Instant::now();
        self.run(&meta, input.reader.as_mut(), &mut hints, &mut sink, &cfg)?;
        sink.finish()?;
        let seconds = t.elapsed().as_secs_f64();
        log(&format!(
            "wrote {} output(s) for {clip} ({} frames @ {w0}x{h0}) in {seconds:.1} s ({:.2} frames/s)",
            written.len(),
            meta.num_frames,
            meta.num_frames as f64 / seconds
        ));
        Ok(ClipReport { clip, frames: meta.num_frames, size: (w0, h0), fps: meta.fps, written, seconds })
    }

    /// Keys `frames` 8-bit RGB frames of `(height, width)` held in memory (`frames × height ×
    /// width × 3` bytes, display-referred like a decoded video: `opts.linear` applies). Only the
    /// keying options of `opts` are read (outputs, windows, hint qualities, despill, cutout,
    /// linear); hints come from `hints`, not from `opts`' paths.
    pub fn key_frames(&self, rgb: &[u8], frames: usize, (h, w): (usize, usize), hints: MemoryHints<'_>, opts: &ClipOptions) -> Result<KeyedFrames> {
        if opts.outputs.is_empty() {
            bail!("no outputs asked for");
        }
        let frame_len = h * w * 3;
        if frames == 0 || rgb.len() != frames * frame_len {
            bail!("expected {frames} frames of {h}x{w}x3 bytes ({} bytes), got {}", frames * frame_len, rgb.len());
        }
        let mut meta = FrameMeta::new((h, w), 1.0, self.quantum, 24.0, frames)?;
        meta.linear_adjust = opts.linear;
        let (cfg, _) = self.pipeline_for(opts, &meta, true)?;
        let px = h * w;
        if hints.first_frame.is_some_and(|f| f.len() != px) || hints.frames.is_some_and(|f| f.len() != frames * px) {
            bail!("hints must be {h}x{w} per frame (and one per frame for the hint frames)");
        }
        let mut reader = SliceReader { rgb, frame_len };
        let mut hint_frames = hints.frames.map(|values| SliceHints { values, px });
        let mut pipeline_hints = Hints {
            first_frame: hints.first_frame.map(<[f32]>::to_vec),
            first_frame_quality: opts.hint_quality,
            video: hint_frames.as_mut().map(|r| r as &mut dyn HintReader),
            video_quality: opts.hint_quality,
        };
        let mut sink = Collect { parts: HashMap::new() };
        self.run(&meta, &mut reader, &mut pipeline_hints, &mut sink, &cfg)?;

        let mut out = KeyedFrames::new();
        for &o in &opts.outputs {
            let mut parts = sink.parts.remove(&o).unwrap_or_default();
            parts.sort_by_key(|(first, _)| *first);
            let mut values: Vec<f32> = parts.into_iter().flat_map(|(_, v)| v).collect();
            let channels = o.channels(true);
            if values.len() != frames * h * w * channels {
                bail!("{o:?}: {} values rendered, expected {}", values.len(), frames * h * w * channels);
            }
            // The renderer writes B G R (A), the order of the EXR channels; NumPy users expect RGB.
            if channels >= 3 {
                for px in values.chunks_exact_mut(channels) {
                    px.swap(0, 2);
                }
            }
            out.insert(o, values);
        }
        Ok(out)
    }
}

/// Hints held in memory for [`ClipKeyer::key_frames`], values in [0, 1] at the frames' size.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryHints<'a> {
    /// Hint of frame 0, `[h, w]`.
    pub first_frame: Option<&'a [f32]>,
    /// Hints of every frame, `[frames, h, w]`.
    pub frames: Option<&'a [f32]>,
}

struct SliceHints<'a> {
    values: &'a [f32],
    px: usize,
}

impl HintReader for SliceHints<'_> {
    fn read_hints(&mut self, start: usize, count: usize) -> CkResult<Vec<f32>> {
        Ok(self.values[start * self.px..(start + count) * self.px].to_vec())
    }
}

struct SliceReader<'a> {
    rgb: &'a [u8],
    frame_len: usize,
}

impl FrameReader for SliceReader<'_> {
    fn read_rgb8(&mut self, start: usize, count: usize) -> CkResult<Vec<u8>> {
        Ok(self.rgb[start * self.frame_len..(start + count) * self.frame_len].to_vec())
    }
}

struct Collect {
    parts: HashMap<RenderOutput, Vec<(usize, Vec<f32>)>>,
}

impl FrameSink for Collect {
    fn write(&mut self, output: RenderOutput, first_frame: usize, _frames: usize, data: RenderedFrames) -> CkResult<()> {
        let RenderedFrames::Exr(values) = data else {
            return Err(corridorkey_core::CkError::InvalidConfig("expected float frames".into()));
        };
        self.parts.entry(output).or_default().push((first_frame, values));
        Ok(())
    }
}

/// Opens `opts.input` (`probe_video_meta` / `probe_frame_dir_meta`): the frames to load and, for
/// a video, the start frame of the chosen window.
pub fn open_input(opts: &ClipOptions) -> Result<OpenedInput> {
    let wanted = opts.num_frames.filter(|&n| n > 0);
    if opts.input.is_dir() {
        let mut paths = frame_paths(&opts.input)?;
        if let Some(n) = wanted {
            paths.truncate(n);
        }
        let reader = ImageDirReader::new(paths, opts.frame_dir_linear)?;
        return Ok(OpenedInput {
            native_hw: reader.native_hw(),
            fps: opts.frame_dir_fps,
            frames: reader.len(),
            start_frame: 0,
            reader: Box::new(reader),
            is_frame_dir: true,
        });
    }
    if !opts.input.exists() {
        bail!("input {} does not exist", opts.input.display());
    }
    let ffmpeg = &opts.encoder.ffmpeg;
    let info = probe(&ffprobe_for(ffmpeg), &opts.input)?;
    let n_total = info.frames;
    let n = wanted.map_or(n_total, |n| n.min(n_total));
    let max_start = n_total.saturating_sub(n);
    let start = match (opts.start_mode, wanted.is_some() && max_start > 0) {
        (StartMode::Begin, _) | (_, false) => 0,
        (StartMode::Middle, true) => max_start / 2,
        (StartMode::RandomMiddle, true) => {
            let (lo, hi) = (max_start / 4, 3 * max_start / 4);
            Pcg64::from_seed(opts.seed).integers(lo as i64, hi as i64 + 1) as usize
        }
    };
    let reader = VideoReader::new(ffmpeg, &opts.input, &info, start);
    Ok(OpenedInput {
        native_hw: (info.height, info.width),
        fps: info.fps,
        frames: n,
        start_frame: start,
        reader: Box::new(reader),
        is_frame_dir: false,
    })
}

/// `<clip>_settings.txt`, the run log `run_inference` writes next to the outputs.
fn write_settings(path: &Path, opts: &ClipOptions, meta: &FrameMeta, cfg: &PipelineConfig) -> Result<()> {
    std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
    let (h0, w0) = meta.native_hw;
    let l = &opts.linear;
    let start_mode = match opts.start_mode {
        StartMode::Begin => "begin",
        StartMode::Middle => "middle",
        StartMode::RandomMiddle => "random_middle",
    };
    let text = format!(
        "CorridorKey v2 Inference Settings Log\n====================================\n\
         Input: {}\nOutput Directory: {}\nResolution: {w0}x{h0}\nFPS: {}\nOutputs: {}\nOutput Format: {}\n\
         Hint First Frame: {}\nHint Video: {}\nHint Quality (Trust): {}\nCarry Hint: {}\nCarry Hint Quality: {}\n\
         Despill Strength: {}\nCutout Linear: {}\nFrame Dir Linear: {}\nHann Chunk: {}\nHann Stride: {}\nLow VRAM: {}\n\
         Model Scale: 1\nNum Frames: {}\nStart Mode: {start_mode}\nSeed: {}\nLinear Brightness: {}\nLinear Contrast: {}\n\
         Linear Contrast Pivot: {}\nEngine: rust\n",
        opts.input.display(),
        opts.output_dir.display(),
        meta.fps,
        cfg.outputs.iter().map(|o| format!("{o:?}").to_lowercase()).collect::<Vec<_>>().join(", "),
        if cfg.exr { "exr" } else { "video" },
        opts.hint_first_frame.as_ref().map_or("None".into(), |p| p.display().to_string()),
        opts.hint_video.as_ref().map_or("None".into(), |p| p.display().to_string()),
        opts.hint_quality,
        cfg.carry_hint,
        cfg.carry_hint_quality,
        cfg.despill,
        cfg.cutout_linear,
        opts.frame_dir_linear,
        cfg.chunk,
        cfg.stride,
        opts.low_vram,
        meta.num_frames,
        opts.seed,
        l.gain,
        l.contrast,
        l.pivot,
    );
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}
