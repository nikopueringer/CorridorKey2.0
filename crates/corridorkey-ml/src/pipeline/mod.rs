//! Keying a clip on the device (`runtime.run_inference`, `hann_stage1_stream`,
//! `precompute_rvm_to_memmap`).
//!
//! Frames go up once per pass as decoded bytes; everything between (frame views, the three
//! backbones, Stage 1 over overlapping windows, the Hann blend of the batons, carried hints,
//! Stage 2, rendering) stays on the device; one download per output and frame batch comes back.
//! Where the Python driver sends batons, hints and backbone features through the host (and to
//! memory-mapped files), this keeps them in device buffers.
//!
//! Decoding and encoding sit behind [`FrameReader`] and [`FrameSink`], so the same driver serves the
//! command line (ffmpeg, image sequences) and the browser (WebCodecs).

pub mod kernels;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use burn_tensor::Tensor;
use corridorkey_base::core::CkFloat;
use corridorkey_base::core::buffer::{empty, zeros};
use corridorkey_base::frame::{BackboneInputs, RenderOutput, RenderSettings, bilinear_resize, render_exr, render_video, upload_frame_views};
pub use corridorkey_base::frame::FrameData;
use corridorkey_core::compute::LaunchGeometry;
use corridorkey_core::{CkError, CkResult, FrameMeta, WindowSchedule};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::hub::{FoundationStats, SafetensorsFile};
use crate::models::foundations::{CRadio, MogeEncoder, RvmSmall};
use crate::models::stage1::Stage1;
use crate::models::stage2::Stage2;
use crate::runtime::{MlContext, elem_dtype, into_contiguous_cube};
use kernels::{accumulate_kernel, assemble_rgb_hint_kernel, carry_hint_kernel, copy_kernel, finalize_baton_kernel, pad_hint_kernel};

/// Frames per RVM chunk in the continuous pre-pass (`min(hann_chunk, 8)` in the product).
pub const RVM_CHUNK: usize = 8;
/// Lower bound of a frame's Hann weight sum before dividing (`max(live_w, 1e-6)`).
const MIN_WEIGHT_SUM: f64 = 1e-6;

/// Where the runtime finds its weights.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub checkpoint: PathBuf,
    pub moge: PathBuf,
    pub rvm: PathBuf,
    pub cradio: PathBuf,
    pub stats: PathBuf,
}

impl ModelPaths {
    /// The package layout: `weights/` (checkpoint, stats, converted backbones) and `vfms/`.
    pub fn from_root(root: &Path) -> Self {
        Self {
            checkpoint: root.join("weights/corridorkey_v2.safetensors"),
            moge: root.join("weights/foundations/moge2_vitb_encoder.safetensors"),
            rvm: root.join("weights/foundations/rvm_small.safetensors"),
            cradio: root.join("vfms/cradio/C-RADIOv4-SO400M/model.safetensors"),
            stats: root.join("weights/foundation_stats.safetensors"),
        }
    }
}

/// Decoded frames of a clip, RGB `[count, h, w, 3]` at the native size.
pub trait FrameReader {
    /// Frames `start .. start + count` as 8-bit RGB (the driver reads overlapping ranges in order).
    fn read_rgb8(&mut self, start: usize, count: usize) -> CkResult<Vec<u8>>;

    /// Frames `start .. start + count` in the reader's own form: 8-bit unless overridden (float
    /// sources such as EXR frames hand over [`FrameData::RgbF32`]). What the driver calls.
    fn read(&mut self, start: usize, count: usize) -> CkResult<FrameData> {
        Ok(FrameData::Rgb8(self.read_rgb8(start, count)?))
    }
}

/// Hints of a clip's frames at the native size, values in [0, 1].
pub trait HintReader {
    /// Hints of frames `start .. start + count`: `[count, h0, w0]` (read in order, like frames).
    fn read_hints(&mut self, start: usize, count: usize) -> CkResult<Vec<f32>>;
}

/// External hints of a run (`--hint-first-frame`, `--hint-video`) and the quality (trust) Stage 1
/// is told for each. Carried hints override the video's; the first-frame hint overrides both on
/// frame 0 of the first window (`hann_stage1_stream`).
#[derive(Default)]
pub struct Hints<'a> {
    /// Hint of frame 0, `[h0, w0]`.
    pub first_frame: Option<Vec<f32>>,
    pub first_frame_quality: f64,
    /// Hints of every frame.
    pub video: Option<&'a mut dyn HintReader>,
    pub video_quality: f64,
}

/// Pads `count` native-size hints (`values`, `[count, h0, w0]`) into `dst` from frame `dst_frame`.
fn pad_hints<F: CkFloat>(client: &Client, values: &[f32], count: usize, meta: &FrameMeta, dst: &Handle, dst_frame: usize, round: bool) -> CkResult<()> {
    let (h0, w0) = meta.native_hw;
    let (h, w) = meta.shape_hw;
    if values.len() != count * h0 * w0 {
        return Err(CkError::Shape(format!("{} hint values for {count} frames of {h0}x{w0}", values.len())));
    }
    let src = corridorkey_base::core::buffer::upload(client, values);
    let total = count * h * w;
    let geom = LaunchGeometry::elementwise(client, total);
    unsafe {
        pad_hint_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(src, values.len()),
            BufferArg::from_raw_parts(dst.clone(), (dst_frame + count) * h * w),
            total as u32,
            h as u32,
            w as u32,
            h0 as u32,
            w0 as u32,
            meta.pad.top as u32,
            meta.pad.left as u32,
            (dst_frame * h * w) as u32,
            round,
        );
    }
    Ok(())
}

/// Rendered frames of one output: `[frames, h, w, channels]`.
pub enum RenderedFrames {
    /// 16-bit BGR / BGRA (video).
    Video(Vec<u16>),
    /// f32 (EXR).
    Exr(Vec<f32>),
}

/// Receives rendered frames, in order per output.
pub trait FrameSink {
    fn write(&mut self, output: RenderOutput, first_frame: usize, frames: usize, data: RenderedFrames) -> CkResult<()>;
}

/// Options of a run (`InferenceConfig`).
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub chunk: usize,
    pub stride: usize,
    pub carry_hint: bool,
    pub carry_hint_quality: f64,
    pub stage2_batch: usize,
    pub outputs: Vec<RenderOutput>,
    pub exr: bool,
    pub despill: f64,
    pub cutout_linear: bool,
    /// Reproduce the product's host-side roundings: blended batons to f16 before Stage 2, carried
    /// hints and hint qualities to bf16.
    pub product_rounding: bool,
    /// Frames per backbone pass (C-RADIO, MoGe) within a window; `None` runs the whole window at
    /// once. The backbones see each frame on its own, so the result is the same, with only this
    /// many frames' backbone activations held at once (the product's `low_vram_foundation_chunk`).
    pub foundation_chunk: Option<usize>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            chunk: 80,
            stride: 40,
            carry_hint: true,
            carry_hint_quality: 0.95,
            stage2_batch: 1,
            outputs: vec![RenderOutput::Alpha, RenderOutput::Fg],
            exr: false,
            despill: 0.5,
            cutout_linear: false,
            product_rounding: true,
            foundation_chunk: None,
        }
    }
}

/// All networks of the keyer, loaded as `F`.
pub struct Keyer {
    stage1: Stage1,
    stage2: Stage2,
    moge: MogeEncoder,
    rvm: RvmSmall,
    cradio: CRadio,
}

impl Keyer {
    pub fn load<F: CkFloat>(ctx: &MlContext, paths: &ModelPaths) -> CkResult<Self> {
        let open = |p: &Path| SafetensorsFile::open(p);
        let ckpt = open(&paths.checkpoint)?;
        let header = ckpt.checkpoint_header()?;
        let stats = FoundationStats::from_file(&paths.stats)?;
        Ok(Self {
            stage1: Stage1::load::<F>(ctx, &ckpt, &header.model_config, "stage1")?,
            stage2: Stage2::load::<F>(ctx, &ckpt, &header.model_config, "stage2")?,
            moge: MogeEncoder::load::<F>(ctx, &open(&paths.moge)?, &stats)?,
            rvm: RvmSmall::load::<F>(ctx, &open(&paths.rvm)?, &stats)?,
            cradio: CRadio::load::<F>(ctx, &open(&paths.cradio)?, &stats)?,
        })
    }

    /// Keys a clip without external hints: see [`Keyer::run_with_hints`].
    pub fn run<F: CkFloat>(
        &self,
        ctx: &MlContext,
        meta: &FrameMeta,
        reader: &mut dyn FrameReader,
        sink: &mut dyn FrameSink,
        cfg: &PipelineConfig,
    ) -> CkResult<()> {
        self.run_with_hints::<F>(ctx, meta, reader, &mut Hints::default(), sink, cfg)
    }

    /// Keys a clip: RVM over all frames, then Stage 1 windows with Hann-blended batons, external
    /// and carried hints, Stage 2 and rendering as frames complete.
    pub fn run_with_hints<F: CkFloat>(
        &self,
        ctx: &MlContext,
        meta: &FrameMeta,
        reader: &mut dyn FrameReader,
        hints: &mut Hints<'_>,
        sink: &mut dyn FrameSink,
        cfg: &PipelineConfig,
    ) -> CkResult<()> {
        let client = ctx.client();
        let dt = elem_dtype::<F>();
        let total = meta.num_frames;
        let (h, w) = meta.shape_hw;
        let (h4, w4) = (h / 4, w / 4);
        let (h16, w16, d16) = (h / 16, w / 16, self.stage1.d_16);
        let (h64, w64) = (h / 64, w / 64);
        let p64 = h64 * w64;
        let rvm_dim = self.rvm.dim();
        let frame_px = h * w;
        let baton_len = h16 * w16 * d16;
        let adjust = meta.linear_adjust;
        let render = RenderSettings::new(meta, cfg.despill, cfg.cutout_linear);
        if cfg.chunk == 0 || cfg.stride == 0 || cfg.stride > cfg.chunk || cfg.stage2_batch == 0 {
            return Err(CkError::InvalidConfig(format!("bad window settings {cfg:?}")));
        }

        // 1. RVM, continuously over the clip (state carried across chunks).
        let rvm_all = empty::<F>(client, total * p64 * rvm_dim);
        let mut state = self.rvm.zero_state::<F>(ctx, h64, w64);
        let rvm_chunk = cfg.chunk.clamp(1, RVM_CHUNK);
        for t0 in (0..total).step_by(rvm_chunk) {
            let n = rvm_chunk.min(total - t0);
            let views = upload_frame_views::<F>(client, &reader.read(t0, n)?, n, meta, adjust)?;
            let rvm_in = bilinear_resize::<F>(client, &views.foundation, (n, h, w, 3), (h4, w4), true);
            let (feats, next) = self.rvm.forward::<F>(ctx, ctx.tensor_from_handle(rvm_in, [n, h4, w4, 3], dt), state)?;
            state = next;
            let len = n * p64 * rvm_dim;
            copy::<F>(client, &into_contiguous_cube(feats)?.handle, &rvm_all, len, 0, t0 * p64 * rvm_dim);
        }

        // 2. Stage 1 windows.
        let schedule = WindowSchedule::new(total, cfg.chunk, cfg.stride, cfg.chunk - cfg.stride)?;
        let quality = |q: f64| if cfg.product_rounding { half::bf16::from_f64(q).to_f64() } else { q };
        let mut live: BTreeMap<usize, (Handle, f64)> = BTreeMap::new();
        let mut carry: HashMap<usize, Handle> = HashMap::new();
        let mut model_frames: HashMap<usize, (Handle, usize)> = HashMap::new();
        let windows = schedule.windows().to_vec();
        for (wi, window) in windows.iter().enumerate() {
            let (t0, t1) = (window.frames.start, window.frames.end);
            let tc = t1 - t0;
            let views = upload_frame_views::<F>(client, &reader.read(t0, tc)?, tc, meta, adjust)?;
            let inputs = BackboneInputs::from_views::<F>(client, &views);
            let (mh, mw) = inputs.moge_hw;
            let cradio_in = ctx.tensor_from_handle::<4>(inputs.cradio, [tc, h4, w4, 3], dt);
            let moge_in = ctx.tensor_from_handle::<4>(inputs.moge, [tc, mh, mw, 3], dt);
            let fc = cfg.foundation_chunk.unwrap_or(tc).clamp(1, tc);
            let (mut cradio_parts, mut moge_parts) = (Vec::new(), Vec::new());
            for s in (0..tc).step_by(fc) {
                let n = fc.min(tc - s);
                cradio_parts.push(self.cradio.forward::<F>(ctx, cradio_in.clone().narrow(0, s, n))?);
                moge_parts.push(self.moge.forward::<F>(ctx, moge_in.clone().narrow(0, s, n))?);
            }
            // One part (the usual case): used as is, no concatenated copy
            let join = |mut parts: Vec<Tensor<4>>| if parts.len() == 1 { parts.pop().expect("one part") } else { Tensor::cat(parts, 0) };
            let cradio = join(cradio_parts);
            let moge = join(moge_parts);
            let rvm_w = empty::<F>(client, tc * p64 * rvm_dim);
            copy::<F>(client, &rvm_all, &rvm_w, tc * p64 * rvm_dim, t0 * p64 * rvm_dim, 0);
            let rvm = ctx.tensor_from_handle::<4>(rvm_w, [tc, h64, w64, rvm_dim], dt);
            let vfm = Tensor::cat(vec![cradio, moge, rvm], 3);
            let vfm_c = vfm.dims()[3];
            let vfm = vfm.reshape([1, tc, h64, w64, vfm_c]);

            // Hints of this window: the hint video's, overridden by those carried from the previous
            // window's aux, then by the first-frame hint on frame 0.
            let hint = zeros::<F>(client, tc * frame_px);
            let mut hq = vec![0.0f64; tc];
            let mut present = false;
            if let Some(video) = hints.video.as_deref_mut() {
                pad_hints::<F>(client, &video.read_hints(t0, tc)?, tc, meta, &hint, 0, cfg.product_rounding)?;
                hq.fill(quality(hints.video_quality));
                present = true;
            }
            for j in 0..tc {
                if let Some(c) = carry.get(&(t0 + j)) {
                    copy::<F>(client, c, &hint, frame_px, 0, j * frame_px);
                    hq[j] = quality(cfg.carry_hint_quality);
                    present = true;
                }
            }
            if let (Some(first), 0) = (&hints.first_frame, t0) {
                pad_hints::<F>(client, first, 1, meta, &hint, 0, cfg.product_rounding)?;
                hq[0] = quality(hints.first_frame_quality);
                present = true;
            }
            if !present {
                hq.fill(-1.0);
            }
            let rgb_hint = empty::<F>(client, tc * frame_px * 4);
            let n4 = tc * frame_px * 4;
            let geom = LaunchGeometry::elementwise(client, n4);
            unsafe {
                assemble_rgb_hint_kernel::launch::<F>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    BufferArg::from_raw_parts(views.model.clone(), tc * frame_px * 3),
                    BufferArg::from_raw_parts(hint, tc * frame_px),
                    BufferArg::from_raw_parts(rgb_hint.clone(), n4),
                    n4 as u32,
                );
            }
            let hq_t = ctx.upload_tensor(&corridorkey_base::core::cast_all::<F>(&hq), [tc, 1])?;
            let out = self.stage1.forward::<F>(ctx, ctx.tensor_from_handle(rgb_hint, [1, tc, h, w, 4], dt), vfm, Some(hq_t), 1.0)?;
            let baton = into_contiguous_cube(out.baton)?.handle;

            let next_start = windows.get(wi + 1).map_or(total, |nw| nw.frames.start);
            if cfg.carry_hint {
                let aux = into_contiguous_cube(out.aux)?.handle;
                let resized = bilinear_resize::<F>(client, &aux, (tc, h / 2, w / 2, 4), (h, w), true);
                for j in 0..tc {
                    let c = empty::<F>(client, frame_px);
                    let geom = LaunchGeometry::elementwise(client, frame_px);
                    unsafe {
                        carry_hint_kernel::launch::<F>(
                            client,
                            geom.cube_count,
                            geom.cube_dim,
                            BufferArg::from_raw_parts(resized.clone(), tc * frame_px * 4),
                            BufferArg::from_raw_parts(c.clone(), frame_px),
                            frame_px as u32,
                            j as u32,
                            cfg.product_rounding,
                        );
                    }
                    carry.insert(t0 + j, c);
                }
                carry.retain(|&t, _| t >= next_start);
            }

            // Hann blend into the per-frame accumulators.
            for (j, &wgt) in schedule.weights(window).iter().enumerate() {
                let t = t0 + j;
                let entry = live.entry(t).or_insert_with(|| (zeros::<f32>(client, baton_len), 0.0));
                let geom = LaunchGeometry::elementwise(client, baton_len);
                unsafe {
                    accumulate_kernel::launch::<F>(
                        client,
                        geom.cube_count,
                        geom.cube_dim,
                        BufferArg::from_raw_parts(entry.0.clone(), baton_len),
                        BufferArg::from_raw_parts(baton.clone(), tc * baton_len),
                        wgt as f32,
                        baton_len as u32,
                        (j * baton_len) as u32,
                    );
                }
                entry.1 += wgt as f32 as f64;
                model_frames.insert(t, (views.model.clone(), j));
            }

            // Frames no later window touches: Stage 2, render, hand to the sink.
            let ready: Vec<usize> = live.range(..next_start).map(|(&t, _)| t).collect();
            for batch in ready.chunks(cfg.stage2_batch) {
                self.stage2_batch::<F>(ctx, batch, &mut live, &mut model_frames, (h, w, h16, w16, d16), &render, sink, cfg)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn stage2_batch<F: CkFloat>(
        &self,
        ctx: &MlContext,
        frames: &[usize],
        live: &mut BTreeMap<usize, (Handle, f64)>,
        model_frames: &mut HashMap<usize, (Handle, usize)>,
        (h, w, h16, w16, d16): (usize, usize, usize, usize, usize),
        render: &RenderSettings,
        sink: &mut dyn FrameSink,
        cfg: &PipelineConfig,
    ) -> CkResult<()> {
        let client = ctx.client();
        let dt = elem_dtype::<F>();
        let n = frames.len();
        let (baton_len, frame_len) = (h16 * w16 * d16, h * w * 3);
        let batons = empty::<F>(client, n * baton_len);
        let rgb = empty::<F>(client, n * frame_len);
        for (k, t) in frames.iter().enumerate() {
            let (acc, wsum) = live.remove(t).expect("frame accumulated");
            let geom = LaunchGeometry::elementwise(client, baton_len);
            unsafe {
                finalize_baton_kernel::launch::<F>(
                    client,
                    geom.cube_count,
                    geom.cube_dim,
                    BufferArg::from_raw_parts(acc, baton_len),
                    BufferArg::from_raw_parts(batons.clone(), n * baton_len),
                    (1.0 / wsum.max(MIN_WEIGHT_SUM)) as f32,
                    baton_len as u32,
                    (k * baton_len) as u32,
                    cfg.product_rounding,
                );
            }
            let (view, j) = model_frames.remove(t).expect("frame view cached");
            copy::<F>(client, &view, &rgb, frame_len, j * frame_len, k * frame_len);
        }
        let logits = self.stage2.forward::<F>(
            ctx,
            ctx.tensor_from_handle(rgb, [n, h, w, 3], dt),
            ctx.tensor_from_handle(batons, [n, h16, w16, d16], dt),
        )?;
        let logits = into_contiguous_cube(logits)?.handle;
        for &output in &cfg.outputs {
            let data = if cfg.exr {
                RenderedFrames::Exr(render_exr::<F>(client, &logits, n, render, output)?)
            } else {
                RenderedFrames::Video(render_video::<F>(client, &logits, n, render, output)?)
            };
            sink.write(output, frames[0], n, data)?;
        }
        Ok(())
    }
}

/// `dst[dst_offset ..] = src[src_offset .. src_offset + len]` on the device.
fn copy<F: CkFloat>(client: &Client, src: &Handle, dst: &Handle, len: usize, src_offset: usize, dst_offset: usize) {
    let geom = LaunchGeometry::elementwise(client, len);
    unsafe {
        copy_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(src.clone(), src_offset + len),
            BufferArg::from_raw_parts(dst.clone(), dst_offset + len),
            len as u32,
            src_offset as u32,
            dst_offset as u32,
        );
    }
}
