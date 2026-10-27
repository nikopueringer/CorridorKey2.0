//! Frame-directory input (`frame_paths`, `probe_frame_dir_meta`, `read_image_rgb`,
//! `_read_frame_dir_views`): PNG / JPEG frames as 8-bit RGB, EXR frames as float RGB, sorted by
//! path. With `linear` (`--frame-dir-linear`) the values are scene-linear (bytes / 255 for PNG /
//! JPEG, like the product); otherwise display values.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use corridorkey_core::{CkError, CkResult};
use corridorkey_ml::pipeline::{FrameData, FrameReader};

fn is_exr(p: &Path) -> bool {
    p.extension().is_some_and(|e| e == "exr")
}

/// Frame files of `dir` with the product's extensions (matched as written, lower case), sorted.
/// EXR and PNG / JPEG frames are not mixed.
pub fn frame_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("listing {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| ["exr", "png", "jpg", "jpeg"].contains(&e)))
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no frames (exr, png, jpg, jpeg) under {}", dir.display());
    }
    let exr = paths.iter().filter(|p| is_exr(p)).count();
    if exr != 0 && exr != paths.len() {
        bail!("{} mixes EXR and PNG / JPEG frames", dir.display());
    }
    Ok(paths)
}

/// Reads one EXR frame as interleaved float RGB (the first RGB(A) layer; alpha dropped, like
/// `read_image_rgb`). Returns `((h, w), values)`.
fn read_exr_rgb(path: &Path) -> Result<((usize, usize), Vec<f32>)> {
    use exr::prelude::*;
    let image = read_first_rgba_layer_from_file(
        path,
        |size, _| (vec![0f32; size.width() * size.height() * 3], size.width()),
        |(buf, w): &mut (Vec<f32>, usize), pos, (r, g, b, _a): (f32, f32, f32, f32)| {
            let i = (pos.y() * *w + pos.x()) * 3;
            buf[i] = r;
            buf[i + 1] = g;
            buf[i + 2] = b;
        },
    )
    .with_context(|| format!("reading {}", path.display()))?;
    let size = image.layer_data.size;
    Ok(((size.height(), size.width()), image.layer_data.channel_data.pixels.0))
}

/// An EXR file as `cv2.imread(IMREAD_UNCHANGED)` (the product's `exr_io.read_exr`) lays it out:
/// `((h, w), channels, values)`, values interleaved in B G R (A) order for a color image
/// (channels 3 or 4), one channel for a `Y` or single-channel image.
pub fn read_exr_cv(path: &Path) -> Result<((usize, usize), usize, Vec<f32>)> {
    let image = exr::prelude::read_first_flat_layer_from_file(path).with_context(|| format!("reading {}", path.display()))?;
    let layer = image.layer_data;
    let (w, h) = (layer.size.width(), layer.size.height());
    let planes: Vec<(String, Vec<f32>)> =
        layer.channel_data.list.iter().map(|c| (c.name.to_string(), c.sample_data.values_as_f32().collect())).collect();
    let plane = |name: &str| planes.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice());
    let interleave = |order: &[&[f32]]| -> Vec<f32> { (0..w * h).flat_map(|i| order.iter().map(move |p| p[i])).collect() };
    if let (Some(b), Some(g), Some(r)) = (plane("B"), plane("G"), plane("R")) {
        return Ok(match plane("A") {
            Some(a) => ((h, w), 4, interleave(&[b, g, r, a])),
            None => ((h, w), 3, interleave(&[b, g, r])),
        });
    }
    if let Some(y) = plane("Y") {
        return Ok(((h, w), 1, y.to_vec()));
    }
    match planes.as_slice() {
        [(_, only)] => Ok(((h, w), 1, only.clone())),
        _ => bail!("{}: no RGB, Y or single channel to read (channels: {:?})", path.display(), planes.iter().map(|(n, _)| n).collect::<Vec<_>>()),
    }
}

/// Size `(h, w)` of a frame file.
fn frame_hw(path: &Path) -> Result<(usize, usize)> {
    if is_exr(path) {
        return Ok(read_exr_rgb(path)?.0);
    }
    let (w, h) = image::image_dimensions(path).with_context(|| format!("reading {}", path.display()))?;
    Ok((h as usize, w as usize))
}

/// Reads a frame directory (see the module docs).
pub struct ImageDirReader {
    paths: Vec<PathBuf>,
    hw: (usize, usize),
    exr: bool,
    linear: bool,
}

impl ImageDirReader {
    pub fn new(paths: Vec<PathBuf>, linear: bool) -> Result<Self> {
        let hw = frame_hw(&paths[0])?;
        Ok(Self { exr: is_exr(&paths[0]), hw, paths, linear })
    }

    pub fn native_hw(&self) -> (usize, usize) {
        self.hw
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    fn check_hw(&self, p: &Path, hw: (usize, usize)) -> CkResult<()> {
        if hw != self.hw {
            return Err(CkError::Shape(format!("{}: {}x{} differs from the first frame", p.display(), hw.1, hw.0)));
        }
        Ok(())
    }
}

impl FrameReader for ImageDirReader {
    fn read_rgb8(&mut self, start: usize, count: usize) -> CkResult<Vec<u8>> {
        if self.exr {
            return Err(CkError::InvalidConfig("EXR frames are float: read them with `read`".into()));
        }
        let mut out = Vec::with_capacity(count * self.hw.0 * self.hw.1 * 3);
        for p in &self.paths[start..start + count] {
            let img = image::open(p).map_err(|e| CkError::Io(format!("{}: {e}", p.display())))?.into_rgb8();
            self.check_hw(p, (img.height() as usize, img.width() as usize))?;
            out.extend_from_slice(img.as_raw());
        }
        Ok(out)
    }

    fn read(&mut self, start: usize, count: usize) -> CkResult<FrameData> {
        if !self.exr {
            let bytes = self.read_rgb8(start, count)?;
            if !self.linear {
                return Ok(FrameData::Rgb8(bytes));
            }
            let values = bytes.into_iter().map(|b| b as f32 / 255.0).collect();
            return Ok(FrameData::RgbF32 { values, linear: true });
        }
        let mut values = Vec::with_capacity(count * self.hw.0 * self.hw.1 * 3);
        for p in &self.paths[start..start + count] {
            let (hw, frame) = read_exr_rgb(p).map_err(|e| CkError::Io(format!("{e:#}")))?;
            self.check_hw(p, hw)?;
            values.extend_from_slice(&frame);
        }
        Ok(FrameData::RgbF32 { values, linear: self.linear })
    }
}
