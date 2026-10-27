//! From decoded frames to the network inputs, on the device (`video_io.py`,
//! `runtime._build_foundation_inputs`).
//!
//! Frames go up once, as the decoder's 8-bit RGB (half the bytes of `f16`, a quarter of `f32`;
//! dsp-kitchen's "cross in the most compact form"). One kernel then writes the model input and
//! the backbones' view at the padded size; the backbone inputs are resized from that view. Nothing
//! comes back to the host.

pub mod kernels;

use corridorkey_core::compute::LaunchGeometry;
use corridorkey_core::frame::{FrameMeta, LinearAdjust};
use corridorkey_core::{CkError, CkResult};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::color::asinh_params;
use crate::core::buffer::{empty, upload};
use crate::core::{CkFloat, cast};
use kernels::{
    OUT_ALPHA, OUT_CHECKER, OUT_CUTOUT, OUT_FG, area_resize_kernel, bilinear_resize_kernel, frame_views_f32_kernel,
    frame_views_kernel, render_f32_kernel, render_u16_kernel,
};

/// Packs bytes four per little-endian `u32` word (zero-padded), the form kernels read 8-bit data in.
pub fn pack_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}

/// Uploads `[T, h0, w0, 3]` 8-bit RGB frames as packed words.
pub fn upload_frames(client: &Client, bytes: &[u8]) -> Handle {
    upload(client, &pack_bytes(bytes))
}

/// Device views of `t` frames at the padded model size `[t, H, W, 3]`.
#[derive(Debug, Clone)]
pub struct FrameViews {
    /// Stage 1 / Stage 2 input: `asinh(adjust(srgb_to_linear(display)))`.
    pub model: Handle,
    /// What the backbones see: the display frame, or `linear_to_vfm_sdr(adjusted)` when a
    /// linear adjustment is set.
    pub foundation: Handle,
    pub t: usize,
    pub hw: (usize, usize),
}

/// Decoded frames as a reader hands them over: `[T, h0, w0, 3]` RGB.
#[derive(Debug, Clone)]
pub enum FrameData {
    /// 8-bit display values (video, PNG / JPEG frames).
    Rgb8(Vec<u8>),
    /// Float values (EXR frames): scene-linear when `linear`, else display values clipped to [0, 1]
    /// (`video_io._read_frame_dir_views`).
    RgbF32 { values: Vec<f32>, linear: bool },
}

/// Uploads `frames` (`t` frames of `meta`'s native size) in their compact form and prepares their
/// [`FrameViews`]: bytes through [`frame_views`], floats through [`frame_views_f32`].
pub fn upload_frame_views<F: CkFloat>(client: &Client, frames: &FrameData, t: usize, meta: &FrameMeta, adjust: LinearAdjust) -> CkResult<FrameViews> {
    match frames {
        FrameData::Rgb8(bytes) => frame_views::<F>(client, &upload_frames(client, bytes), t, meta, adjust),
        FrameData::RgbF32 { values, linear } => frame_views_f32::<F>(client, &upload(client, values), t, meta, adjust, *linear),
    }
}

/// Checks that `meta` can be prepared on the device; returns `(h0, w0)`, `(h, w)`.
fn check_geometry(meta: &FrameMeta) -> CkResult<((usize, usize), (usize, usize))> {
    if meta.model_native_hw != meta.native_hw {
        return Err(CkError::InvalidConfig(format!(
            "model_scale {} is not supported: frames must be prepared at their native size",
            meta.model_scale
        )));
    }
    let (h0, w0) = meta.native_hw;
    let p = meta.pad;
    if p.top.max(p.bottom) >= h0 || p.left.max(p.right) >= w0 {
        return Err(CkError::InvalidConfig(format!("padding {p:?} exceeds the {h0}x{w0} frame (reflect-101)")));
    }
    Ok(((h0, w0), meta.shape_hw))
}

/// [`FrameViews`] of `t` float frames (an f32 buffer, `[t, h0, w0, 3]`), see
/// [`frame_views_f32_kernel`] for `linear`.
pub fn frame_views_f32<F: CkFloat>(client: &Client, frames: &Handle, t: usize, meta: &FrameMeta, adjust: LinearAdjust, linear: bool) -> CkResult<FrameViews> {
    let ((h0, w0), (h, w)) = check_geometry(meta)?;
    let p = meta.pad;
    let total = t * h * w * 3;
    let (model, foundation) = (empty::<F>(client, total), empty::<F>(client, total));
    let (scale, offset) = adjust.affine();
    let (knee, denom) = asinh_params();
    let geom = LaunchGeometry::elementwise(client, total);
    unsafe {
        frame_views_f32_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            1,
            BufferArg::from_raw_parts(frames.clone(), t * h0 * w0 * 3),
            BufferArg::from_raw_parts(model.clone(), total),
            BufferArg::from_raw_parts(foundation.clone(), total),
            total as u32,
            h as u32,
            w as u32,
            h0 as u32,
            w0 as u32,
            p.top as u32,
            p.left as u32,
            cast::<F>(scale),
            cast::<F>(offset),
            cast::<F>(1.0 / knee),
            cast::<F>(1.0 / denom),
            linear,
            !adjust.is_identity(),
        );
    }
    Ok(FrameViews { model, foundation, t, hw: (h, w) })
}

/// [`FrameViews`] of `t` packed frames of `meta`'s native size (`model_scale` 1: the product's
/// render path supports only that), reflect-101 padded to `meta.shape_hw`.
pub fn frame_views<F: CkFloat>(client: &Client, frames: &Handle, t: usize, meta: &FrameMeta, adjust: LinearAdjust) -> CkResult<FrameViews> {
    let ((h0, w0), (h, w)) = check_geometry(meta)?;
    let p = meta.pad;
    let total = t * h * w * 3;
    let (model, foundation) = (empty::<F>(client, total), empty::<F>(client, total));
    let (scale, offset) = adjust.affine();
    let (knee, denom) = asinh_params();
    let geom = LaunchGeometry::elementwise(client, total);
    unsafe {
        frame_views_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            1,
            BufferArg::from_raw_parts(frames.clone(), (t * h0 * w0 * 3).div_ceil(4)),
            BufferArg::from_raw_parts(model.clone(), total),
            BufferArg::from_raw_parts(foundation.clone(), total),
            total as u32,
            h as u32,
            w as u32,
            h0 as u32,
            w0 as u32,
            p.top as u32,
            p.left as u32,
            cast::<F>(scale),
            cast::<F>(offset),
            cast::<F>(1.0 / knee),
            cast::<F>(1.0 / denom),
            !adjust.is_identity(),
        );
    }
    Ok(FrameViews { model, foundation, t, hw: (h, w) })
}

/// `interpolate(mode="area")` of `[t, h, w, ch]` to `[t, oh, ow, ch]`.
pub fn area_resize<F: CkFloat>(client: &Client, input: &Handle, (t, h, w, ch): (usize, usize, usize, usize), (oh, ow): (usize, usize)) -> Handle {
    let total = t * oh * ow * ch;
    let out = empty::<F>(client, total);
    let geom = LaunchGeometry::elementwise(client, total);
    unsafe {
        area_resize_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), t * h * w * ch),
            BufferArg::from_raw_parts(out.clone(), total),
            total as u32,
            h as u32,
            w as u32,
            ch as u32,
            oh as u32,
            ow as u32,
        );
    }
    out
}

/// `interpolate(mode="bilinear", align_corners=False)` of `[t, h, w, ch]` to `[t, oh, ow, ch]`,
/// clamped to [0, 1] when `clamp`.
pub fn bilinear_resize<F: CkFloat>(client: &Client, input: &Handle, (t, h, w, ch): (usize, usize, usize, usize), (oh, ow): (usize, usize), clamp: bool) -> Handle {
    let total = t * oh * ow * ch;
    let out = empty::<F>(client, total);
    let geom = LaunchGeometry::elementwise(client, total);
    unsafe {
        bilinear_resize_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), t * h * w * ch),
            BufferArg::from_raw_parts(out.clone(), total),
            total as u32,
            h as u32,
            w as u32,
            ch as u32,
            oh as u32,
            ow as u32,
            clamp,
        );
    }
    out
}

/// The three backbones' inputs from the foundation view, at the product's sizes.
#[derive(Debug, Clone)]
pub struct BackboneInputs {
    /// `[t, H/4, W/4, 3]`, area resize.
    pub cradio: Handle,
    /// `[t, 7H/32, 7W/32, 3]`, area resize.
    pub moge: Handle,
    /// `[t, H/4, W/4, 3]`, bilinear resize clamped to [0, 1] (`precompute_rvm_to_memmap`).
    pub rvm: Handle,
    pub quarter_hw: (usize, usize),
    pub moge_hw: (usize, usize),
}

impl BackboneInputs {
    pub fn from_views<F: CkFloat>(client: &Client, views: &FrameViews) -> Self {
        let (t, (h, w)) = (views.t, views.hw);
        let dims = (t, h, w, 3);
        let quarter_hw = (h / 4, w / 4);
        let moge_hw = (h * 7 / 32, w * 7 / 32);
        Self {
            cradio: area_resize::<F>(client, &views.foundation, dims, quarter_hw),
            moge: area_resize::<F>(client, &views.foundation, dims, moge_hw),
            rvm: bilinear_resize::<F>(client, &views.foundation, dims, quarter_hw, true),
            quarter_hw,
            moge_hw,
        }
    }
}

/// An output stream of `infer.py` (`--outputs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderOutput {
    /// The matte, clamped to [0, 1] (video: three equal channels; EXR: one).
    Alpha,
    /// The foreground (video: sRGB; EXR: scene-linear, unclamped).
    Fg,
    /// Despilled foreground plus alpha, BGRA.
    Cutout,
    /// Despilled foreground composited over a 64-pixel checkerboard in linear light.
    Checker,
}

impl RenderOutput {
    fn code(self) -> u32 {
        match self {
            Self::Alpha => OUT_ALPHA,
            Self::Fg => OUT_FG,
            Self::Cutout => OUT_CUTOUT,
            Self::Checker => OUT_CHECKER,
        }
    }

    /// Channels per pixel of the written frames.
    pub fn channels(self, exr: bool) -> usize {
        match self {
            Self::Alpha if exr => 1,
            Self::Cutout => 4,
            _ => 3,
        }
    }
}

/// How logits become output frames (`meta` fields of `_render_predictions_*`).
#[derive(Debug, Clone, Copy)]
pub struct RenderSettings {
    /// Native size written; the logits are cropped from the padded model size at `pad`.
    pub native_hw: (usize, usize),
    pub padded_hw: (usize, usize),
    /// `(top, left)` padding.
    pub pad: (usize, usize),
    /// The clip's linear adjustment (undone before rendering).
    pub adjust: LinearAdjust,
    pub despill: f64,
    /// `--cutout_linear`: despill the linear foreground for the cutout instead of the sRGB one.
    pub cutout_linear: bool,
}

impl RenderSettings {
    pub fn new(meta: &FrameMeta, despill: f64, cutout_linear: bool) -> Self {
        Self {
            native_hw: meta.native_hw,
            padded_hw: meta.shape_hw,
            pad: (meta.pad.top, meta.pad.left),
            adjust: meta.linear_adjust,
            despill,
            cutout_linear,
        }
    }
}

/// Launch arguments shared by both render kernels.
fn render_args(settings: &RenderSettings) -> (f32, f32, f32, f32, f32) {
    let (inv_scale, inv_offset) = settings.adjust.inverse_affine();
    let (knee, denom) = asinh_params();
    (inv_scale as f32, inv_offset as f32, knee as f32, denom as f32, settings.despill as f32)
}

/// 16-bit frames `[t, h, w, channels]` (BGR / BGRA, values `floor(x · 65535)`) of `output` from
/// `[t, H, W, 4]` logits: one download.
pub fn render_video<F: CkFloat>(client: &Client, logits: &Handle, t: usize, settings: &RenderSettings, output: RenderOutput) -> CkResult<Vec<u16>> {
    let (h, w) = settings.padded_hw;
    let (oh, ow) = settings.native_hw;
    let channels = output.channels(false);
    let total = t * oh * ow * channels;
    let words_n = total.div_ceil(2);
    let words = empty::<u32>(client, words_n);
    let (inv_scale, inv_offset, knee, denom, strength) = render_args(settings);
    let geom = LaunchGeometry::elementwise(client, words_n);
    unsafe {
        render_u16_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(logits.clone(), t * h * w * 4),
            BufferArg::from_raw_parts(words.clone(), words_n),
            total as u32,
            h as u32,
            w as u32,
            oh as u32,
            ow as u32,
            settings.pad.0 as u32,
            settings.pad.1 as u32,
            inv_scale,
            inv_offset,
            knee,
            denom,
            strength,
            channels as u32,
            output.code(),
            settings.cutout_linear,
        );
    }
    let words: Vec<u32> = crate::core::buffer::try_download(client, words)?;
    Ok(words.into_iter().flat_map(|w| [w as u16, (w >> 16) as u16]).take(total).collect())
}

/// f32 EXR frames `[t, h, w, channels]` of `output` from `[t, H, W, 4]` logits: one download.
pub fn render_exr<F: CkFloat>(client: &Client, logits: &Handle, t: usize, settings: &RenderSettings, output: RenderOutput) -> CkResult<Vec<f32>> {
    let (h, w) = settings.padded_hw;
    let (oh, ow) = settings.native_hw;
    let channels = output.channels(true);
    let total = t * oh * ow * channels;
    let out = empty::<f32>(client, total);
    let (inv_scale, inv_offset, knee, denom, strength) = render_args(settings);
    let geom = LaunchGeometry::elementwise(client, total);
    unsafe {
        render_f32_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(logits.clone(), t * h * w * 4),
            BufferArg::from_raw_parts(out.clone(), total),
            total as u32,
            h as u32,
            w as u32,
            oh as u32,
            ow as u32,
            settings.pad.0 as u32,
            settings.pad.1 as u32,
            inv_scale,
            inv_offset,
            knee,
            denom,
            strength,
            channels as u32,
            output.code(),
        );
    }
    crate::core::buffer::try_download(client, out)
}
