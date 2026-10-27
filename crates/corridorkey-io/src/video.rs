//! Video input through ffmpeg (`probe_video_meta`, `SequentialVideoWindowReader`).
//!
//! Frames are decoded to 8-bit RGB through `ffmpeg -f rawvideo -pix_fmt rgb24` and read in order;
//! the keyer reads overlapping windows (kept in a cache) and runs two passes (RVM, then Stage 1),
//! so a read before the cache restarts the decoder. `start_frame` is reached by decoding and
//! dropping frames: exact for every codec (OpenCV's seek can land near, not on, the frame).

use std::collections::VecDeque;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, bail};
use corridorkey_core::{CkError, CkResult};
use corridorkey_ml::pipeline::FrameReader;
use serde_json::Value;

/// Size, frame rate and frame count of a video (`cv2.VideoCapture` properties).
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub frames: usize,
}

/// ffprobe next to `ffmpeg` when that is a path, else `ffprobe` from `PATH`.
pub fn ffprobe_for(ffmpeg: &str) -> String {
    if ffmpeg.contains('/') {
        Path::new(ffmpeg).with_file_name("ffprobe").display().to_string()
    } else {
        "ffprobe".into()
    }
}

/// Probes the first video stream with ffprobe. The frame count is the container's `nb_frames`,
/// else `duration × fps` (OpenCV's `CAP_PROP_FRAME_COUNT` reads the same metadata).
pub fn probe(ffprobe: &str, path: &Path) -> Result<VideoInfo> {
    let out = Command::new(ffprobe)
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration", "-of", "json"])
        .arg(path)
        .output()
        .with_context(|| format!("running {ffprobe}"))?;
    if !out.status.success() {
        bail!("{ffprobe} failed on {}: {}", path.display(), String::from_utf8_lossy(&out.stderr));
    }
    let json: Value = serde_json::from_slice(&out.stdout)?;
    let s = &json["streams"][0];
    let rate = |key: &str| -> Option<f64> {
        let (n, d) = s[key].as_str()?.split_once('/')?;
        let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
        (d > 0.0 && n > 0.0).then_some(n / d)
    };
    let fps = rate("avg_frame_rate").or_else(|| rate("r_frame_rate")).unwrap_or(24.0);
    let count = s["nb_frames"].as_str().and_then(|v| v.parse::<usize>().ok()).or_else(|| {
        let d: f64 = s["duration"].as_str()?.parse().ok()?;
        Some((d * fps).round() as usize)
    });
    let (Some(width), Some(height), Some(frames)) = (s["width"].as_u64(), s["height"].as_u64(), count) else {
        bail!("{}: no video stream with size and frame count", path.display());
    };
    Ok(VideoInfo { width: width as usize, height: height as usize, fps, frames })
}

/// Sequential ffmpeg decoder with a window cache (see the module docs).
pub struct VideoReader {
    ffmpeg: String,
    path: PathBuf,
    frame_len: usize,
    start_frame: usize,
    child: Option<(Child, ChildStdout)>,
    /// Clip-relative index of the next frame the decoder yields.
    next: usize,
    cache: VecDeque<Vec<u8>>,
    /// Clip-relative index of `cache[0]`.
    cache_start: usize,
}

impl VideoReader {
    pub fn new(ffmpeg: &str, path: &Path, info: &VideoInfo, start_frame: usize) -> Self {
        Self {
            ffmpeg: ffmpeg.into(),
            path: path.into(),
            frame_len: info.width * info.height * 3,
            start_frame,
            child: None,
            next: 0,
            cache: VecDeque::new(),
            cache_start: 0,
        }
    }

    fn restart(&mut self) -> Result<()> {
        if let Some((mut child, _)) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        let mut child = Command::new(&self.ffmpeg)
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(&self.path)
            .args(["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdout(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting {}", self.ffmpeg))?;
        let stdout = child.stdout.take().expect("piped stdout");
        self.child = Some((child, stdout));
        self.next = 0;
        self.cache.clear();
        self.cache_start = 0;
        // Decode up to the clip's first frame.
        let mut skip = vec![0u8; self.frame_len];
        for i in 0..self.start_frame {
            self.read_raw(&mut skip).with_context(|| format!("skipping to frame {} (at {i})", self.start_frame))?;
        }
        Ok(())
    }

    fn read_raw(&mut self, buf: &mut [u8]) -> Result<()> {
        let (_, stdout) = self.child.as_mut().expect("decoder running");
        stdout.read_exact(buf).with_context(|| format!("{}: decoding ended early", self.path.display()))
    }
}

impl FrameReader for VideoReader {
    fn read_rgb8(&mut self, start: usize, count: usize) -> CkResult<Vec<u8>> {
        let io = |e: anyhow::Error| CkError::Io(format!("{e:#}"));
        if self.child.is_none() || start < self.cache_start {
            self.restart().map_err(io)?;
        }
        // Drop frames before `start`, decode up to `start + count`.
        while self.cache_start < start && !self.cache.is_empty() {
            self.cache.pop_front();
            self.cache_start += 1;
        }
        if self.cache.is_empty() {
            self.cache_start = self.next;
        }
        while self.next < start + count {
            let mut frame = vec![0u8; self.frame_len];
            self.read_raw(&mut frame).map_err(io)?;
            if self.next >= start {
                self.cache.push_back(frame);
            } else {
                self.cache_start = self.next + 1;
            }
            self.next += 1;
        }
        let first = start - self.cache_start;
        Ok(self.cache.iter().skip(first).take(count).flatten().copied().collect())
    }
}

impl Drop for VideoReader {
    fn drop(&mut self) {
        if let Some((mut child, _)) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
