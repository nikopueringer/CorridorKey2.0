//! Output writers (`ThreadedVideoWriter`, `ExrSequenceWriter`) and the sink that routes rendered
//! frames to them. Each writer runs on its own thread behind a small bounded queue, like the
//! Python writers, so encoding overlaps inference.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::JoinHandle;

use anyhow::{Context, Result, anyhow, bail};
use corridorkey_base::frame::RenderOutput;
use corridorkey_core::{CkError, CkResult};
use corridorkey_ml::pipeline::{FrameSink, RenderedFrames};

/// Queue depth of each writer thread (`maxsize=2`).
const WRITER_QUEUE: usize = 2;

/// ffmpeg encoding options (`WriterConfig`).
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub ffmpeg: String,
    pub codec: String,
    pub crf: u32,
    pub preset: String,
    pub pix_fmt: String,
    pub bitrate: Option<String>,
    pub threads: u32,
}

impl Default for EncoderConfig {
    /// `infer.py`'s defaults.
    fn default() -> Self {
        Self {
            ffmpeg: "ffmpeg".into(),
            codec: "libx264".into(),
            crf: 12,
            preset: "medium".into(),
            pix_fmt: "yuv444p10le".into(),
            bitrate: None,
            threads: 2,
        }
    }
}

/// The file-name suffix of an output (`suffix` in `run_inference`).
pub fn output_suffix(output: RenderOutput, despill: f64) -> String {
    match output {
        RenderOutput::Alpha => "alpha".into(),
        RenderOutput::Fg => "fg".into(),
        RenderOutput::Cutout => "cutout".into(),
        RenderOutput::Checker => format!("checker_comp_despill{:03}", (despill * 100.0) as i64),
    }
}

enum Job {
    Video(Vec<u16>),
    Exr(usize, Vec<f32>),
}

/// One output stream on its own thread.
struct Writer {
    tx: Option<SyncSender<Job>>,
    thread: Option<JoinHandle<Result<()>>>,
    next_frame: usize,
}

impl Writer {
    fn send(&mut self, job: Job) -> Result<()> {
        let tx = self.tx.as_ref().expect("writer open");
        if tx.send(job).is_err() {
            return self.finish().and(Err(anyhow!("writer thread stopped")));
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.tx.take();
        match self.thread.take() {
            Some(t) => t.join().map_err(|_| anyhow!("writer thread panicked"))?,
            None => Ok(()),
        }
    }
}

/// ffmpeg encoder of 16-bit frames (`ThreadedVideoWriter._start`, same arguments).
fn video_writer(path: &Path, fps: f64, (w, h): (usize, usize), channels: usize, cfg: &EncoderConfig, color_trc: &str) -> Result<Writer> {
    let in_fmt = if channels == 4 { "bgra64le" } else { "bgr48le" };
    let mut cmd = Command::new(&cfg.ffmpeg);
    cmd.args(["-y", "-hide_banner", "-loglevel", "error", "-nostats", "-f", "rawvideo", "-pix_fmt", in_fmt])
        .args(["-s:v", &format!("{w}x{h}"), "-r", &format!("{fps:.8}"), "-i", "pipe:0", "-an"]);
    if channels == 4 {
        cmd.args(["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]);
    } else {
        cmd.args(["-c:v", &cfg.codec, "-preset", &cfg.preset, "-pix_fmt", &cfg.pix_fmt]);
        match &cfg.bitrate {
            Some(b) => cmd.args(["-b:v", b]),
            None => cmd.args(["-crf", &cfg.crf.to_string()]),
        };
    }
    cmd.args(["-threads", &cfg.threads.max(1).to_string()])
        .args(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", color_trc, "-movflags", "+write_colr"])
        .arg(path)
        .stdin(Stdio::piped());
    let mut child: Child = cmd.spawn().with_context(|| format!("starting {}", cfg.ffmpeg))?;
    let mut stdin = child.stdin.take().expect("piped stdin");
    let (tx, rx) = sync_channel::<Job>(WRITER_QUEUE);
    let path = path.to_path_buf();
    let thread = std::thread::spawn(move || -> Result<()> {
        for job in rx {
            let Job::Video(values) = job else { unreachable!() };
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            stdin.write_all(&bytes).with_context(|| format!("writing to ffmpeg for {}", path.display()))?;
        }
        drop(stdin);
        let status = child.wait()?;
        if !status.success() {
            bail!("ffmpeg writer failed ({status}): {}", path.display());
        }
        Ok(())
    });
    Ok(Writer { tx: Some(tx), thread: Some(thread), next_frame: 0 })
}

/// EXR sequence `<dir>/<name>_NNNNNN.exr`, ZIP-compressed, channels B G R (A) or Y, half (clamped
/// to ±65504 to stay finite) or float (`ExrSequenceWriter`).
fn exr_writer(dir: &Path, name: &str, (w, h): (usize, usize), channels: usize, half: bool) -> Result<Writer> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let (tx, rx) = sync_channel::<Job>(WRITER_QUEUE);
    let (dir, name) = (dir.to_path_buf(), name.to_string());
    let thread = std::thread::spawn(move || -> Result<()> {
        for job in rx {
            let Job::Exr(first, values) = job else { unreachable!() };
            let frame_len = w * h * channels;
            for (j, frame) in values.chunks_exact(frame_len).enumerate() {
                let path = dir.join(format!("{name}_{:06}.exr", first + j));
                write_exr(&path, frame, (w, h), channels, half).with_context(|| format!("writing {}", path.display()))?;
            }
        }
        Ok(())
    });
    Ok(Writer { tx: Some(tx), thread: Some(thread), next_frame: 0 })
}

fn write_exr(path: &Path, frame: &[f32], (w, h): (usize, usize), channels: usize, half: bool) -> Result<()> {
    use exr::prelude::*;
    // Interleaved B G R (A) input; EXR stores named planar channels.
    let names: &[&str] = match channels {
        1 => &["Y"],
        3 => &["B", "G", "R"],
        _ => &["B", "G", "R", "A"],
    };
    let plane = |c: usize| frame.iter().skip(c).step_by(channels).copied();
    let list: Vec<AnyChannel<FlatSamples>> = names
        .iter()
        .enumerate()
        .map(|(c, n)| {
            let samples = if half {
                FlatSamples::F16(plane(c).map(|v| f16::from_f32(v.clamp(-65504.0, 65504.0))).collect())
            } else {
                FlatSamples::F32(plane(c).collect())
            };
            AnyChannel::new(*n, samples)
        })
        .collect();
    let layer = Layer::new((w, h), LayerAttributes::default(), Encoding { compression: Compression::ZIP16, ..Encoding::default() }, AnyChannels::sort(list.into()));
    Image::from_layer(layer).write().to_file(path)?;
    Ok(())
}

/// Routes rendered outputs to their writers.
pub struct OutputSink {
    writers: HashMap<RenderOutput, Writer>,
    exr: bool,
}

impl OutputSink {
    /// Opens one writer per output under `out_dir`, named like `run_inference`.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        out_dir: &Path,
        clip: &str,
        outputs: &[RenderOutput],
        size: (usize, usize),
        fps: f64,
        despill: f64,
        exr: Option<bool>,
        cutout_linear: bool,
        cfg: &EncoderConfig,
    ) -> Result<(Self, Vec<PathBuf>)> {
        std::fs::create_dir_all(out_dir)?;
        let mut writers = HashMap::new();
        let mut paths = Vec::new();
        for &out in outputs {
            let suffix = output_suffix(out, despill);
            let channels = out.channels(exr.is_some());
            let writer = match exr {
                Some(half) => {
                    let dir = out_dir.join(format!("{clip}_{suffix}_exr"));
                    paths.push(dir.clone());
                    exr_writer(&dir, &suffix, size, channels, half)?
                }
                None => {
                    let ext = if out == RenderOutput::Cutout { "mov" } else { "mp4" };
                    let path = out_dir.join(format!("{clip}_{suffix}.{ext}"));
                    let trc = if out == RenderOutput::Cutout && cutout_linear { "linear" } else { "iec61966-2-1" };
                    paths.push(path.clone());
                    video_writer(&path, fps, size, channels, cfg, trc)?
                }
            };
            writers.insert(out, writer);
        }
        Ok((Self { writers, exr: exr.is_some() }, paths))
    }

    /// Closes every writer, waiting for its encoder.
    pub fn finish(mut self) -> Result<()> {
        for (_, mut w) in self.writers.drain() {
            w.finish()?;
        }
        Ok(())
    }
}

impl FrameSink for OutputSink {
    fn write(&mut self, output: RenderOutput, first_frame: usize, frames: usize, data: RenderedFrames) -> CkResult<()> {
        let w = self.writers.get_mut(&output).ok_or_else(|| CkError::InvalidConfig(format!("no writer for {output:?}")))?;
        if first_frame != w.next_frame {
            return Err(CkError::InvalidConfig(format!("{output:?}: frame {first_frame} arrived, expected {}", w.next_frame)));
        }
        w.next_frame += frames;
        let job = match (data, self.exr) {
            (RenderedFrames::Video(v), false) => Job::Video(v),
            (RenderedFrames::Exr(v), true) => Job::Exr(first_frame, v),
            _ => return Err(CkError::InvalidConfig("rendered data does not match the output format".into())),
        };
        w.send(job).map_err(|e| CkError::Io(format!("{e:#}")))
    }
}
