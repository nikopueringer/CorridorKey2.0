//! MoGe-2 ViT-B encoder (`moge.model.modules.DINOv2Encoder`, Microsoft, MIT; DINOv2 ViT-B/14,
//! Meta, Apache-2.0) as `FrozenFoundations` runs it: features of layers 5 and 11 after the final
//! norm, each through a 1 × 1 projection, summed, then the `moge_mean` / `moge_std` normalisation.
//! MoGe's neck and heads are not used.
//!
//! Two load-time folds replace whole passes:
//! - the ImageNet `(x - mean) / std` of the input folds into the patch embedding:
//!   `W · ((x - m) / s) + b = (W / s) · x + (b - Σ W · m / s)`;
//! - the projections of all read-out layers, their summed biases and the feature normalisation
//!   become one GEMM: `[ln(b5) | ln(b11)] · [W5; W11] / std + (b5 + b11 - mean) / std`.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use burn_tensor::Tensor;
use corridorkey_base::core::{CkFloat, buffer::{empty, upload}, cast_all};
use corridorkey_core::{CkError, CkResult, compute::LaunchGeometry};
use cubecl::{device::DeviceId, prelude::*, server::Handle};
use serde::Deserialize;

use super::vit::{LayerNorm, VIT_LAYER_NORM_EPS, VitBlock, resize_pos_embed_bicubic};
use crate::{
    hub::{FoundationStats, SafetensorsFile},
    models::blocks::{
        PatchifyConv,
        kernels::{add_row_bias_kernel, assemble_tokens_kernel},
    },
    runtime::{MlContext, elem_dtype, into_contiguous_cube},
};

/// DINOv2 ViT-B/14: patch size, heads, the `interpolate_offset` of its position-embedding resize.
pub const MOGE_PATCH: usize = 14;
const MOGE_HEADS: usize = 12;
const DINOV2_INTERPOLATE_OFFSET: f64 = 0.1;
/// `FrozenFoundations` clamps the feature std from below before dividing.
const STATS_MIN_STD: f64 = 1e-6;

#[derive(Deserialize)]
struct EncoderConfig {
    intermediate_layers: Vec<usize>,
    dim_out: usize,
}

/// Device position-embedding tables per `(device, rows, cols)`.
type PosCache = Arc<Mutex<HashMap<(DeviceId, usize, usize, usize), Handle>>>;

/// The MoGe-2 encoder, ready to run as `F`.
#[derive(Clone, Debug)]
pub struct MogeEncoder {
    patch: PatchifyConv,
    cls: Handle,
    /// Square `[grid, grid, dim]` patch position table and the class position, f64 on the host.
    pos_grid: Vec<f64>,
    pos_cls: Vec<f64>,
    grid: usize,
    pos_cache: PosCache,
    blocks: Vec<VitBlock>,
    norm: LayerNorm,
    layers: Vec<usize>,
    readout_w: Tensor<2>,
    readout_b: Handle,
    dim: usize,
    dim_out: usize,
}

impl MogeEncoder {
    /// Loads `moge2_vitb_encoder.safetensors` (`scripts/convert_foundations.py`) and folds the
    /// MoGe feature statistics into the read-out.
    pub fn load<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, stats: &FoundationStats) -> CkResult<Self> {
        let cfg: EncoderConfig = st
            .metadata()
            .get("config")
            .ok_or_else(|| CkError::Weights("MoGe encoder safetensors has no 'config' metadata".into()))
            .and_then(|c| serde_json::from_str(c).map_err(|e| CkError::Weights(format!("MoGe config: {e}"))))?;
        let f64s = |name: &str| -> CkResult<(Vec<usize>, Vec<f64>)> {
            let (shape, v) = st.read_f32(name)?;
            Ok((shape, v.into_iter().map(f64::from).collect()))
        };

        // Patch embedding with the ImageNet normalisation folded in.
        let (_, mean) = f64s("image_mean")?;
        let (_, std) = f64s("image_std")?;
        let (pshape, mut pw) = f64s("backbone.patch_embed.proj.weight")?;
        let (_, mut pb) = f64s("backbone.patch_embed.proj.bias")?;
        let (dim, cin, k) = (pshape[0], pshape[1], pshape[2]);
        if k != MOGE_PATCH || cin != 3 || mean.len() != 3 || std.len() != 3 {
            return Err(CkError::Weights(format!("unexpected MoGe patch embedding {pshape:?}")));
        }
        for o in 0..dim {
            for c in 0..cin {
                for t in 0..k * k {
                    let w = &mut pw[(o * cin + c) * k * k + t];
                    pb[o] -= *w * mean[c] / std[c];
                    *w /= std[c];
                }
            }
        }
        let patch = PatchifyConv::from_slices::<F>(ctx, cin, dim, k, 1, &cast_all::<F>(&pw), &cast_all::<F>(&pb))?;

        let (_, cls) = st.read_as::<F>("backbone.cls_token")?;
        let (pos_shape, pos) = f64s("backbone.pos_embed")?;
        let grid = ((pos_shape[1] - 1) as f64).sqrt() as usize;
        if grid * grid + 1 != pos_shape[1] || pos_shape[2] != dim {
            return Err(CkError::Weights(format!("unexpected MoGe pos_embed {pos_shape:?}")));
        }

        let last = *cfg.intermediate_layers.iter().max().ok_or_else(|| CkError::Weights("no read-out layers".into()))?;
        let blocks = (0..=last)
            .map(|i| VitBlock::load::<F>(ctx, st, &format!("backbone.blocks.{i}"), MOGE_HEADS))
            .collect::<CkResult<Vec<_>>>()?;
        let norm = LayerNorm::load::<F>(ctx, st, "backbone.norm", VIT_LAYER_NORM_EPS)?;

        // One read-out GEMM: [L·dim, dim_out] in GEMM (input-major) layout, scaled by 1 / std.
        let n_layers = cfg.intermediate_layers.len();
        let (dim_out, inv_std): (usize, Vec<f64>) = (
            cfg.dim_out,
            stats.moge_std.iter().map(|&s| 1.0 / (s as f64).max(STATS_MIN_STD)).collect(),
        );
        if stats.moge_dim() != dim_out {
            return Err(CkError::Weights(format!("moge stats have {} channels, encoder {dim_out}", stats.moge_dim())));
        }
        let mut w_cat = vec![0.0f64; n_layers * dim * dim_out];
        let mut b_cat: Vec<f64> = stats.moge_mean.iter().map(|&m| -(m as f64)).collect();
        for l in 0..n_layers {
            let (_, w) = f64s(&format!("output_projections.{l}.weight"))?; // [dim_out, dim, 1, 1]
            let (_, b) = f64s(&format!("output_projections.{l}.bias"))?;
            for o in 0..dim_out {
                b_cat[o] += b[o];
                for i in 0..dim {
                    w_cat[(l * dim + i) * dim_out + o] = w[o * dim + i] * inv_std[o];
                }
            }
        }
        for (b, s) in b_cat.iter_mut().zip(&inv_std) {
            *b *= s;
        }

        Ok(Self {
            patch,
            cls: upload(ctx.client(), &cls),
            pos_cls: pos[..dim].to_vec(),
            pos_grid: pos[dim..].to_vec(),
            grid,
            pos_cache: Arc::new(Mutex::new(HashMap::new())),
            blocks,
            norm,
            layers: cfg.intermediate_layers,
            readout_w: ctx.upload_tensor(&cast_all::<F>(&w_cat), [n_layers * dim, dim_out])?,
            readout_b: upload(ctx.client(), &cast_all::<F>(&b_cat)),
            dim,
            dim_out,
        })
    }

    pub fn dim_out(&self) -> usize {
        self.dim_out
    }

    /// `[1 + rows · cols, dim]` position embedding for a `rows × cols` token grid, as `F`, built once
    /// per device and grid.
    pub fn pos_embed<F: CkFloat>(&self, ctx: &MlContext, rows: usize, cols: usize) -> Handle {
        let key = (ctx.device_id(), rows, cols, size_of::<F>());
        let mut cache = self.pos_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = cache.get(&key) {
            return h.clone();
        }
        let grid = resize_pos_embed_bicubic(&self.pos_grid, self.grid, self.dim, (rows, cols), DINOV2_INTERPOLATE_OFFSET);
        let table: Vec<f64> = self.pos_cls.iter().chain(&grid).copied().collect();
        let h = upload(ctx.client(), &cast_all::<F>(&table));
        cache.insert(key, h.clone());
        h
    }

    /// Normalised MoGe features `[n, rows, cols, dim_out]` of `image` (`[n, Hm, Wm, 3]`, display sRGB in
    /// [0, 1], `Hm` and `Wm` multiples of 14: the encoder's own resize is then the identity).
    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, image: Tensor<4>) -> CkResult<Tensor<4>> {
        let [n, hm, wm, _] = image.dims();
        if !hm.is_multiple_of(MOGE_PATCH) || !wm.is_multiple_of(MOGE_PATCH) {
            return Err(CkError::Shape(format!("MoGe input {hm}x{wm} is not a multiple of {MOGE_PATCH}")));
        }
        let (rows, cols, dim) = (hm / MOGE_PATCH, wm / MOGE_PATCH, self.dim);
        let patches_n = rows * cols;
        let tokens_n = patches_n + 1;

        let patches = into_contiguous_cube(self.patch.forward::<F>(ctx, image)?)?.handle;
        let pos = self.pos_embed::<F>(ctx, rows, cols);
        let total = n * tokens_n * dim;
        let mut x = empty::<F>(ctx.client(), total);
        let geom = LaunchGeometry::elementwise(ctx.client(), total);
        unsafe {
            assemble_tokens_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(patches, n * patches_n * dim),
                BufferArg::from_raw_parts(self.cls.clone(), dim),
                BufferArg::from_raw_parts(pos, tokens_n * dim),
                BufferArg::from_raw_parts(x.clone(), total),
                total as u32,
                1,
                patches_n as u32,
                dim as u32,
            );
        }

        // Normed patch tokens of every read-out layer, side by side: [n · patches, L · dim].
        let width = self.layers.len() * dim;
        let readout_in = empty::<F>(ctx.client(), n * patches_n * width);
        for (i, blk) in self.blocks.iter().enumerate() {
            x = blk.forward::<F>(ctx, x, n, tokens_n)?;
            if let Some(slot) = self.layers.iter().position(|&l| l == i) {
                self.norm.forward_into::<F>(
                    ctx,
                    &x,
                    total,
                    &readout_in,
                    n * patches_n * width,
                    n * patches_n,
                    (tokens_n, patches_n),
                    (width, slot * dim),
                );
            }
        }

        let dt = elem_dtype::<F>();
        let features = ctx.matmul(
            ctx.tensor_from_handle(readout_in, [n * patches_n, width], dt),
            self.readout_w.clone(),
        );
        let out = into_contiguous_cube(features)?.handle;
        let len = n * patches_n * self.dim_out;
        let geom = LaunchGeometry::elementwise(ctx.client(), len);
        unsafe {
            add_row_bias_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(out.clone(), len),
                BufferArg::from_raw_parts(self.readout_b.clone(), self.dim_out),
                len as u32,
                self.dim_out as u32,
            );
        }
        Ok(ctx.tensor_from_handle(out, [n, rows, cols, self.dim_out], dt))
    }
}
