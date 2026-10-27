//! `infer`: key a clip (`infer.py`). Every option of the Python entry point, typed and checked
//! here. The run is the Rust engine (`commands::keying`) by default; `--engine python` runs
//! `infer.py` instead (options the Rust engine does not support yet are refused with that hint).

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::{Args, ValueEnum};

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    Alpha,
    /// Foreground (also accepted: raw-fg, foreground)
    #[value(alias = "raw-fg", alias = "raw_fg", alias = "foreground")]
    Fg,
    /// Foreground composited over a checkerboard
    Checker,
    /// Straight RGBA cutout
    Cutout,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartMode {
    Begin,
    Middle,
    #[value(name = "random-middle", alias = "random_middle")]
    RandomMiddle,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// ffmpeg-encoded preview videos (display-referred, clipped to [0, 1])
    Video,
    /// Unclamped scene-linear OpenEXR sequences (full HDR float)
    Exr,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExrDtype {
    Half,
    Float,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// The CubeCL / Burn pipeline (this binary)
    Rust,
    /// `infer.py` in the package's Python environment
    Python,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// The device's fastest supported type (f16 on GPUs that support it)
    Auto,
    F32,
    F16,
}

#[derive(Args, Debug)]
pub struct EngineArgs {
    #[arg(long, value_enum, default_value_t = Engine::Rust)]
    pub engine: Engine,
    /// Compute precision of the Rust engine
    #[arg(long, value_enum, default_value_t = Precision::Auto)]
    pub precision: Precision,
}

#[derive(Args, Debug)]
pub struct InferArgs {
    /// Input video file or directory of image frames
    #[arg(short, long)]
    pub input: PathBuf,
    /// Directory for the outputs
    #[arg(short, long)]
    pub output_dir: PathBuf,
    /// Runtime checkpoint (.safetensors or legacy .pt) [default: weights/corridorkey_v2.safetensors]
    #[arg(long)]
    pub weights: Option<PathBuf>,
    /// Outputs to write
    #[arg(long, value_delimiter = ',', default_values = ["alpha", "fg"])]
    pub outputs: Vec<Output>,

    #[command(flatten, next_help_heading = "Engine")]
    pub engine: EngineArgs,
    #[command(flatten, next_help_heading = "Hints")]
    pub hints: HintArgs,
    #[command(flatten, next_help_heading = "Frames")]
    pub frames: FrameArgs,
    #[command(flatten, next_help_heading = "Colour")]
    pub color: ColorArgs,
    #[command(flatten, next_help_heading = "Output encoding")]
    pub encoding: EncodingArgs,
    #[command(flatten, next_help_heading = "Memory")]
    pub memory: MemoryArgs,
}

#[derive(Args, Debug)]
pub struct HintArgs {
    /// Matte image or directory guiding the first frame only
    #[arg(long)]
    pub hint_first_frame: Option<PathBuf>,
    /// Full-frame hint video or frame directory
    #[arg(long)]
    pub hint_video: Option<PathBuf>,
    #[arg(long, default_value_t = 0.95)]
    pub hint_quality: f64,
    #[arg(long, default_value_t = 0.95)]
    pub carry_hint_quality: f64,
    /// Do not carry the hint from window to window
    #[arg(long)]
    pub no_carry_hint: bool,
}

#[derive(Args, Debug)]
pub struct FrameArgs {
    /// Frames to process (all when omitted)
    #[arg(long)]
    pub num_frames: Option<usize>,
    /// Where the processed frames start when --num-frames is less than the clip
    #[arg(long, value_enum, default_value_t = StartMode::Begin)]
    pub start_mode: StartMode,
    /// Seed of --start-mode random-middle
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Lanczos scale before the model; 1.0 is native resolution
    #[arg(long, default_value_t = 1.0)]
    pub model_scale: f64,
    /// Frame rate of a frame-directory input
    #[arg(long, default_value_t = 24.0)]
    pub frame_dir_fps: f64,
    /// Treat frame-directory inputs as scene-linear RGB
    #[arg(long)]
    pub frame_dir_linear: bool,
    /// Frames per stage-1 window [default: chosen from free GPU memory]
    #[arg(long)]
    pub hann_chunk: Option<usize>,
    /// Frames between stage-1 windows [default: half of --hann-chunk]
    #[arg(long)]
    pub hann_stride: Option<usize>,
}

#[derive(Args, Debug)]
pub struct ColorArgs {
    /// Despill strength, 0 (none) to 1 (full)
    #[arg(long, default_value_t = 0.5)]
    pub despill_strength: f64,
    /// Export the cutout as straight linear RGB instead of straight sRGB
    #[arg(long)]
    pub cutout_linear: bool,
    #[arg(long, default_value_t = 1.0)]
    pub linear_brightness: f64,
    #[arg(long, default_value_t = 1.0)]
    pub linear_contrast: f64,
    #[arg(long, default_value_t = 0.18)]
    pub linear_contrast_pivot: f64,
}

#[derive(Args, Debug)]
pub struct EncodingArgs {
    #[arg(long, value_enum, default_value_t = OutputFormat::Video)]
    pub output_format: OutputFormat,
    /// EXR pixel type for --output-format exr
    #[arg(long, value_enum, default_value_t = ExrDtype::Half)]
    pub exr_dtype: ExrDtype,
    #[arg(long, default_value = "ffmpeg")]
    pub ffmpeg_bin: String,
    #[arg(long, default_value = "libx264")]
    pub ffmpeg_codec: String,
    #[arg(long, default_value_t = 12)]
    pub crf: u32,
    #[arg(long, default_value = "medium")]
    pub preset: String,
    /// Pixel format of x264 previews (10-bit by default; yuv444p for 8-bit compatibility)
    #[arg(long, default_value = "yuv444p10le")]
    pub pix_fmt: String,
    #[arg(long)]
    pub bitrate: Option<String>,
    #[arg(long, default_value_t = 2)]
    pub ffmpeg_threads: u32,
}

#[derive(Args, Debug)]
pub struct MemoryArgs {
    /// Staged low-memory inference (turns off automatic selection, as does --hann-chunk)
    #[arg(long)]
    pub low_vram: bool,
    /// Directory for intermediate files [default: <system temp>/corridorkey_v2]
    #[arg(long)]
    pub temp_dir: Option<PathBuf>,
    /// Output frames per stage-2 batch
    #[arg(long, default_value_t = 1)]
    pub stage2_batch: usize,
    #[arg(long, default_value_t = 4)]
    pub low_vram_foundation_chunk: usize,
    #[arg(long, default_value_t = 65536)]
    pub low_vram_mlp_chunk_tokens: usize,
    #[arg(long, default_value_t = 1)]
    pub low_vram_enc_frame_chunk: usize,
    #[arg(long, default_value_t = 16)]
    pub low_vram_swin_window_batch: usize,
    #[arg(long, default_value_t = 2048)]
    pub low_vram_full_attn_query_chunk_tokens: usize,
}

impl InferArgs {
    fn validate(&self) -> Result<()> {
        if !self.input.exists() {
            bail!("input {} does not exist", self.input.display());
        }
        if !(0.0..=1.0).contains(&self.color.despill_strength) {
            bail!("--despill-strength must be within 0..1, got {}", self.color.despill_strength);
        }
        if !(self.frames.model_scale > 0.0) {
            bail!("--model-scale must be > 0, got {}", self.frames.model_scale);
        }
        if let (Some(chunk), Some(stride)) = (self.frames.hann_chunk, self.frames.hann_stride)
            && (stride == 0 || stride > chunk)
        {
            bail!("--hann-stride must be within 1..=--hann-chunk ({chunk}), got {stride}");
        }
        if self.memory.stage2_batch == 0 {
            bail!("--stage2-batch must be at least 1");
        }
        Ok(())
    }

    /// The same options as `infer.py` arguments.
    pub fn to_python_args(&self) -> Vec<String> {
        let mut a: Vec<String> = vec![];
        let mut push = |flag: &str, value: String| a.extend([format!("--{flag}"), value]);
        let path = |p: &Path| p.display().to_string();
        push("input", path(&self.input));
        push("output_dir", path(&self.output_dir));
        if let Some(w) = &self.weights {
            push("weights", path(w));
        }
        let h = &self.hints;
        if let Some(p) = &h.hint_first_frame {
            push("hint_first_frame", path(p));
        }
        if let Some(p) = &h.hint_video {
            push("hint_video", path(p));
        }
        push("hint_quality", h.hint_quality.to_string());
        push("carry_hint_quality", h.carry_hint_quality.to_string());
        let f = &self.frames;
        push("num_frames", f.num_frames.map_or("-1".into(), |n| n.to_string()));
        push("start_mode", match f.start_mode {
            StartMode::Begin => "begin",
            StartMode::Middle => "middle",
            StartMode::RandomMiddle => "random_middle",
        }.into());
        push("seed", f.seed.to_string());
        push("model_scale", f.model_scale.to_string());
        push("frame_dir_fps", f.frame_dir_fps.to_string());
        if let Some(n) = f.hann_chunk {
            push("hann_chunk", n.to_string());
        }
        if let Some(n) = f.hann_stride {
            push("hann_stride", n.to_string());
        }
        let c = &self.color;
        push("despill_strength", c.despill_strength.to_string());
        push("linear_brightness", c.linear_brightness.to_string());
        push("linear_contrast", c.linear_contrast.to_string());
        push("linear_contrast_pivot", c.linear_contrast_pivot.to_string());
        let e = &self.encoding;
        push("output_format", match e.output_format {
            OutputFormat::Video => "video",
            OutputFormat::Exr => "exr",
        }.into());
        push("exr_dtype", match e.exr_dtype {
            ExrDtype::Half => "half",
            ExrDtype::Float => "float",
        }.into());
        push("ffmpeg_bin", e.ffmpeg_bin.clone());
        push("ffmpeg_codec", e.ffmpeg_codec.clone());
        push("crf", e.crf.to_string());
        push("preset", e.preset.clone());
        push("pix_fmt", e.pix_fmt.clone());
        if let Some(b) = &e.bitrate {
            push("bitrate", b.clone());
        }
        push("ffmpeg_threads", e.ffmpeg_threads.to_string());
        let m = &self.memory;
        if let Some(t) = &m.temp_dir {
            push("temp_dir", path(t));
        }
        push("stage2_batch", m.stage2_batch.to_string());
        push("low_vram_foundation_chunk", m.low_vram_foundation_chunk.to_string());
        push("low_vram_mlp_chunk_tokens", m.low_vram_mlp_chunk_tokens.to_string());
        push("low_vram_enc_frame_chunk", m.low_vram_enc_frame_chunk.to_string());
        push("low_vram_swin_window_batch", m.low_vram_swin_window_batch.to_string());
        push("low_vram_full_attn_query_chunk_tokens", m.low_vram_full_attn_query_chunk_tokens.to_string());
        // `--outputs` takes several values: last, so nothing after it is read as an output
        let flags = [
            (h.no_carry_hint, "--no_carry_hint"),
            (f.frame_dir_linear, "--frame_dir_linear"),
            (c.cutout_linear, "--cutout_linear"),
            (m.low_vram, "--low_vram"),
        ];
        a.extend(flags.into_iter().filter(|(on, _)| *on).map(|(_, f)| f.to_string()));
        a.push("--outputs".into());
        for o in &self.outputs {
            a.push(match o {
                Output::Alpha => "alpha",
                Output::Fg => "fg",
                Output::Checker => "checker",
                Output::Cutout => "cutout",
            }.into());
        }
        a
    }
}

pub fn run(root: &Path, target: anyhow::Result<corridorkey_core::ComputeTarget>, args: &InferArgs) -> Result<()> {
    args.validate()?;
    let missing = super::assets::AssetPaths::new(root).missing();
    if args.weights.is_none() && !missing.is_empty() {
        bail!("missing asset(s): {}; run `corridorkey weights download`", missing.join(", "));
    }
    match args.engine.engine {
        Engine::Rust => super::keying::run(root, target?, args),
        Engine::Python => super::python::run_script(root, "infer.py", &args.to_python_args()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        infer: InferArgs,
    }

    fn parse(extra: &[&str]) -> InferArgs {
        let mut argv = vec!["corridorkey", "--input", "clip.mp4", "--output-dir", "out"];
        argv.extend(extra);
        Cli::parse_from(argv).infer
    }

    #[test]
    fn defaults_match_infer_py() {
        let a = parse(&[]).to_python_args();
        let value = |flag: &str| a.iter().position(|x| x == flag).map(|i| a[i + 1].clone());
        assert_eq!(value("--num_frames").as_deref(), Some("-1"));
        assert_eq!(value("--start_mode").as_deref(), Some("begin"));
        assert_eq!(value("--despill_strength").as_deref(), Some("0.5"));
        assert_eq!(value("--pix_fmt").as_deref(), Some("yuv444p10le"));
        assert_eq!(&a[a.len() - 3..], ["--outputs", "alpha", "fg"]);
        assert!(!a.iter().any(|x| x == "--low_vram" || x == "--hann_chunk"));
    }

    #[test]
    fn aliases_lists_and_flags_translate() {
        let args = parse(&["--outputs", "raw_fg,checker", "--start-mode", "random_middle", "--low-vram", "--hann-chunk", "40"]);
        assert_eq!(args.outputs, vec![Output::Fg, Output::Checker]);
        let a = args.to_python_args();
        assert!(a.windows(2).any(|w| w == ["--start_mode", "random_middle"]));
        assert!(a.windows(2).any(|w| w == ["--hann_chunk", "40"]));
        assert!(a.contains(&"--low_vram".to_string()));
        assert_eq!(&a[a.len() - 3..], ["--outputs", "fg", "checker"]);
    }
}
