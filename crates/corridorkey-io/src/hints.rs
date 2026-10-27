//! Hint inputs (`load_initial_hint_frame`, `read_hint_image`, `make_hint_reader`,
//! `preprocess_hint_frame`): mattes guiding Stage 1, as `[h0, w0]` values in [0, 1] at the native
//! size (the pipeline pads them like the frames).
//!
//! The product reads them two ways, reproduced here:
//! - the first-frame hint (`read_hint_image`): the alpha channel of a 4-channel image, else the
//!   first channel in OpenCV's order (blue of an RGB image), / 255 or / 65535 for integer images,
//!   as is for EXR; clipped to [0, 1]. It must be at the frames' size (the product accepts the
//!   native or the model size, the same at model scale 1);
//! - hint videos and hint frame directories (`preprocess_hint_frame`): OpenCV's BGR→gray (integer
//!   `(4899 R + 9617 G + 1868 B + 8192) >> 14` then / 255 or / 65535, float `0.114 B + 0.587 G +
//!   0.299 R` for EXR), resized to the frames' size with OpenCV's Lanczos4 when it differs
//!   ([`crate::resize`]), clipped to [0, 1].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use corridorkey_core::{CkError, CkResult};
use corridorkey_ml::pipeline::{FrameReader, HintReader};
use image::DynamicImage;

use crate::images::{frame_paths, read_exr_cv};
use crate::resize::{bgr_to_gray_f32, resize_lanczos4};
use crate::video::{VideoInfo, VideoReader, ffprobe_for, probe};

/// OpenCV's `COLOR_BGR2GRAY` on integer pixels.
fn cv_gray(r: u32, g: u32, b: u32) -> u32 {
    (4899 * r + 9617 * g + 1868 * b + 8192) >> 14
}

fn is_exr(path: &Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("exr"))
}

fn open_image(path: &Path) -> Result<DynamicImage> {
    image::open(path).with_context(|| format!("reading hint {}", path.display()))
}

fn clip01(values: &mut [f32]) {
    values.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
}

/// `read_hint_image` of a decoded integer image (see the module docs), before clipping.
fn hint_image_values(img: &DynamicImage) -> Result<Vec<f32>> {
    let v8 = |v: u8| v as f32 / 255.0;
    let v16 = |v: u16| v as f32 / 65535.0;
    Ok(match img {
        DynamicImage::ImageLuma8(i) => i.pixels().map(|p| v8(p[0])).collect(),
        DynamicImage::ImageLumaA8(i) => i.pixels().map(|p| v8(p[1])).collect(),
        DynamicImage::ImageRgb8(i) => i.pixels().map(|p| v8(p[2])).collect(),
        DynamicImage::ImageRgba8(i) => i.pixels().map(|p| v8(p[3])).collect(),
        DynamicImage::ImageLuma16(i) => i.pixels().map(|p| v16(p[0])).collect(),
        DynamicImage::ImageLumaA16(i) => i.pixels().map(|p| v16(p[1])).collect(),
        DynamicImage::ImageRgb16(i) => i.pixels().map(|p| v16(p[2])).collect(),
        DynamicImage::ImageRgba16(i) => i.pixels().map(|p| v16(p[3])).collect(),
        other => bail!("unsupported hint pixel format {:?}", other.color()),
    })
}

/// `preprocess_hint_frame`'s gray values of a decoded integer image, before resizing and clipping.
fn hint_frame_values(img: &DynamicImage) -> Result<Vec<f32>> {
    let v8 = |v: u32| v as f32 / 255.0;
    let v16 = |v: u32| v as f32 / 65535.0;
    let gray = |p: &[u8]| cv_gray(p[0] as u32, p[1] as u32, p[2] as u32);
    let gray16 = |p: &[u16]| cv_gray(p[0] as u32, p[1] as u32, p[2] as u32);
    Ok(match img {
        DynamicImage::ImageLuma8(i) => i.pixels().map(|p| v8(p[0] as u32)).collect(),
        // OpenCV decodes gray + alpha as BGRA: gray of equal channels is the gray value
        DynamicImage::ImageLumaA8(i) => i.pixels().map(|p| v8(p[0] as u32)).collect(),
        DynamicImage::ImageRgb8(i) => i.pixels().map(|p| v8(gray(&p.0))).collect(),
        DynamicImage::ImageRgba8(i) => i.pixels().map(|p| v8(gray(&p.0))).collect(),
        DynamicImage::ImageLuma16(i) => i.pixels().map(|p| v16(p[0] as u32)).collect(),
        DynamicImage::ImageLumaA16(i) => i.pixels().map(|p| v16(p[0] as u32)).collect(),
        DynamicImage::ImageRgb16(i) => i.pixels().map(|p| v16(gray16(&p.0))).collect(),
        DynamicImage::ImageRgba16(i) => i.pixels().map(|p| v16(gray16(&p.0))).collect(),
        other => bail!("unsupported hint pixel format {:?}", other.color()),
    })
}

/// A hint frame file read as `preprocess_hint_frame` reads it, before resizing: `((h, w), gray)`.
fn read_hint_frame(path: &Path) -> Result<((usize, usize), Vec<f32>)> {
    if is_exr(path) {
        let (hw, channels, values) = read_exr_cv(path)?;
        return Ok((hw, if channels == 1 { values } else { bgr_to_gray_f32(&values, channels) }));
    }
    let img = open_image(path)?;
    Ok(((img.height() as usize, img.width() as usize), hint_frame_values(&img)?))
}

/// `preprocess_hint_frame` after reading: to the frames' size (Lanczos4), clipped to [0, 1].
fn fit_hint(hw: (usize, usize), values: Vec<f32>, native_hw: (usize, usize)) -> Vec<f32> {
    let mut out = if hw == native_hw { values } else { resize_lanczos4(&values, hw, native_hw) };
    clip01(&mut out);
    out
}

/// The first-frame hint (`load_initial_hint_frame`): `path`, or the first frame of the directory
/// `path`, as `[h0, w0]` values.
pub fn load_first_frame_hint(path: &Path, native_hw: (usize, usize)) -> Result<Vec<f32>> {
    let file = if path.is_dir() { frame_paths(path)?.remove(0) } else { path.to_path_buf() };
    let (hw, mut values) = if is_exr(&file) {
        let (hw, channels, values) = read_exr_cv(&file)?;
        // `img[:, :, 3] if 4 channels else img[:, :, 0]`: alpha, else blue
        let pick = if channels == 4 { 3 } else { 0 };
        (hw, values.chunks_exact(channels).map(|p| p[pick]).collect())
    } else {
        let img = open_image(&file)?;
        ((img.height() as usize, img.width() as usize), hint_image_values(&img)?)
    };
    if hw != native_hw {
        bail!("{}: hint is {}x{}, the frames are {}x{} (the first-frame hint must be at the frames' size)", file.display(), hw.1, hw.0, native_hw.1, native_hw.0);
    }
    clip01(&mut values);
    Ok(values)
}

/// A hint frame directory (`SequentialHintFrameDirWindowReader`).
pub struct HintDirReader {
    paths: Vec<PathBuf>,
    native_hw: (usize, usize),
}

impl HintReader for HintDirReader {
    fn read_hints(&mut self, start: usize, count: usize) -> CkResult<Vec<f32>> {
        let mut out = Vec::with_capacity(count * self.native_hw.0 * self.native_hw.1);
        for p in &self.paths[start..start + count] {
            let (hw, values) = read_hint_frame(p).map_err(|e| CkError::Io(format!("{e:#}")))?;
            out.extend(fit_hint(hw, values, self.native_hw));
        }
        Ok(out)
    }
}

/// A hint video (`SequentialHintVideoWindowReader`): decoded like the input video, from the same
/// start frame, at its own size.
pub struct HintVideoReader {
    video: VideoReader,
    hw: (usize, usize),
    native_hw: (usize, usize),
}

impl HintReader for HintVideoReader {
    fn read_hints(&mut self, start: usize, count: usize) -> CkResult<Vec<f32>> {
        let rgb = self.video.read_rgb8(start, count)?;
        let frame_px = self.hw.0 * self.hw.1;
        let mut out = Vec::with_capacity(count * self.native_hw.0 * self.native_hw.1);
        for frame in rgb.chunks_exact(frame_px * 3) {
            let gray = frame.chunks_exact(3).map(|p| cv_gray(p[0] as u32, p[1] as u32, p[2] as u32) as f32 / 255.0).collect();
            out.extend(fit_hint(self.hw, gray, self.native_hw));
        }
        Ok(out)
    }
}

/// The hint reader of `path` (`make_hint_reader`): a frame directory or a video, checked to cover
/// `start_frame + frames` frames.
pub fn open_hint_reader(path: &Path, ffmpeg: &str, native_hw: (usize, usize), start_frame: usize, frames: usize) -> Result<Box<dyn HintReader>> {
    if path.is_dir() {
        let paths = frame_paths(path)?;
        if paths.len() < frames {
            bail!("{}: only {} hint frames for {frames} source frames", path.display(), paths.len());
        }
        return Ok(Box::new(HintDirReader { paths, native_hw }));
    }
    let info: VideoInfo = probe(&ffprobe_for(ffmpeg), path)?;
    if info.frames < start_frame + frames {
        bail!(
            "hint video {} has only {} frames, but the source requires {} (start={start_frame}, num={frames})",
            path.display(),
            info.frames,
            start_frame + frames
        );
    }
    let hw = (info.height, info.width);
    Ok(Box::new(HintVideoReader { video: VideoReader::new(ffmpeg, path, &info, start_frame), hw, native_hw }))
}
