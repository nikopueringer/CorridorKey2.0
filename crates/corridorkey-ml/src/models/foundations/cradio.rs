//! C-RADIOv4-SO400M (NVIDIA RADIO, `vfms/cradio/C-RADIOv4-SO400M`) as `FrozenFoundations` runs it:
//! input conditioner, RADIO's CPE patch generator (patch linear, position table resized to the
//! token grid, CLS + register tokens), the 27-block timm `vit_so400m_patch16_224`, then `.features`
//! (the patch tokens; RADIO replaces the backbone's final norm and feature normaliser with
//! identities for this checkpoint) and the `cradio_mean` / `cradio_std` normalisation.
//!
//! Reads the bundled `model.safetensors` directly (fp32, names under `radio_model.`).
//! The input conditioner's `(x - mean) / std` is folded into the bias-free patch linear.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use burn_tensor::Tensor;
use corridorkey_base::core::{
    CkFloat, cast_all,
    buffer::{empty, upload},
};
use corridorkey_core::{CkError, CkResult, compute::LaunchGeometry};
use cubecl::{device::DeviceId, prelude::*, server::Handle};

use super::vit::{VitBlock, resize_table_bilinear};
use crate::{
    hub::{FoundationStats, SafetensorsFile},
    models::blocks::{
        PatchifyConv,
        kernels::{assemble_tokens_kernel, drop_prefix_normalize_kernel},
    },
    runtime::{MlContext, elem_dtype, into_contiguous_cube},
};

/// SO400M patch size and heads (`vit_so400m_patch16_224`: 1152 wide, 16 heads of 72).
pub const CRADIO_PATCH: usize = 16;
const CRADIO_HEADS: usize = 16;
const PREFIX: &str = "radio_model";
const STATS_MIN_STD: f64 = 1e-6;

type PosCache = Arc<Mutex<HashMap<(DeviceId, usize, usize, usize), Handle>>>;

/// C-RADIOv4-SO400M, ready to run as `F`.
#[derive(Clone, Debug)]
pub struct CRadio {
    patch: PatchifyConv,
    prefix_tokens: Handle,
    num_prefix: usize,
    pos_grid: Vec<f64>,
    grid: usize,
    pos_cache: PosCache,
    blocks: Vec<VitBlock>,
    mean: Handle,
    inv_std: Handle,
    dim: usize,
}

impl CRadio {
    /// Loads the bundled snapshot's `model.safetensors` and the C-RADIO feature stats.
    pub fn load<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, stats: &FoundationStats) -> CkResult<Self> {
        let f64s = |name: &str| -> CkResult<(Vec<usize>, Vec<f64>)> {
            let (shape, v) = st.read_f32(&format!("{PREFIX}.{name}"))?;
            Ok((shape, v.into_iter().map(f64::from).collect()))
        };
        // Patch linear [dim, 3·16·16] in (c, y, x) order = OIHW, with the conditioner folded in.
        let (_, mean) = f64s("input_conditioner.norm_mean")?;
        let (_, std) = f64s("input_conditioner.norm_std")?;
        let (wshape, mut w) = f64s("model.patch_generator.embedder.weight")?;
        let (dim, k2c) = (wshape[0], wshape[1]);
        let k2 = CRADIO_PATCH * CRADIO_PATCH;
        if k2c != 3 * k2 || mean.len() != 3 || std.len() != 3 {
            return Err(CkError::Weights(format!("unexpected C-RADIO patch embedder {wshape:?}")));
        }
        let mut bias = vec![0.0f64; dim];
        for o in 0..dim {
            for c in 0..3 {
                for t in 0..k2 {
                    let v = &mut w[o * k2c + c * k2 + t];
                    bias[o] -= *v * mean[c] / std[c];
                    *v /= std[c];
                }
            }
        }
        let patch = PatchifyConv::from_slices::<F>(ctx, 3, dim, CRADIO_PATCH, 1, &cast_all::<F>(&w), &cast_all::<F>(&bias))?;

        let (tshape, tokens) = st.read_as::<F>(&format!("{PREFIX}.model.patch_generator.cls_token.token"))?;
        let (pshape, pos) = f64s("model.patch_generator.pos_embed")?;
        let grid = (pshape[1] as f64).sqrt() as usize;
        if grid * grid != pshape[1] || pshape[2] != dim || tshape[1] != dim {
            return Err(CkError::Weights(format!("unexpected C-RADIO pos_embed {pshape:?} / tokens {tshape:?}")));
        }
        let depth = (0..)
            .take_while(|i| st.entry(&format!("{PREFIX}.model.blocks.{i}.attn.qkv.weight")).is_ok())
            .count();
        let blocks = (0..depth)
            .map(|i| VitBlock::load::<F>(ctx, st, &format!("{PREFIX}.model.blocks.{i}"), CRADIO_HEADS))
            .collect::<CkResult<Vec<_>>>()?;

        if stats.cradio_dim() != dim {
            return Err(CkError::Weights(format!("cradio stats have {} channels, model {dim}", stats.cradio_dim())));
        }
        let inv_std: Vec<f64> = stats.cradio_std.iter().map(|&s| 1.0 / (s as f64).max(STATS_MIN_STD)).collect();
        let mean_f: Vec<f64> = stats.cradio_mean.iter().map(|&m| m as f64).collect();
        Ok(Self {
            patch,
            prefix_tokens: upload(ctx.client(), &tokens),
            num_prefix: tshape[0],
            pos_grid: pos,
            grid,
            pos_cache: Arc::new(Mutex::new(HashMap::new())),
            blocks,
            mean: upload(ctx.client(), &cast_all::<F>(&mean_f)),
            inv_std: upload(ctx.client(), &cast_all::<F>(&inv_std)),
            dim,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// `[prefix + rows · cols, dim]` position table: zero rows for the CLS / register tokens (RADIO
    /// adds positions before prepending them), then the CPE eval path of `_get_pos_embeddings`:
    /// the stored grid resized bilinearly to `max(rows, cols)` square, top-left `rows × cols` kept.
    pub fn pos_table<F: CkFloat>(&self, ctx: &MlContext, rows: usize, cols: usize) -> Handle {
        let key = (ctx.device_id(), rows, cols, size_of::<F>());
        let mut cache = self.pos_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = cache.get(&key) {
            return h.clone();
        }
        let side = rows.max(cols);
        let square = resize_table_bilinear(&self.pos_grid, (self.grid, self.grid), self.dim, (side, side));
        let mut table = vec![0.0f64; self.num_prefix * self.dim];
        for r in 0..rows {
            table.extend_from_slice(&square[r * side * self.dim..(r * side + cols) * self.dim]);
        }
        let h = upload(ctx.client(), &cast_all::<F>(&table));
        cache.insert(key, h.clone());
        h
    }

    /// Normalised C-RADIO features `[n, He/16, We/16, dim]` of `image` (`[n, He, We, 3]`, display
    /// sRGB in [0, 1], He and We multiples of 16).
    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, image: Tensor<4>) -> CkResult<Tensor<4>> {
        let [n, he, we, _] = image.dims();
        if !he.is_multiple_of(CRADIO_PATCH) || !we.is_multiple_of(CRADIO_PATCH) {
            return Err(CkError::Shape(format!("C-RADIO input {he}x{we} is not a multiple of {CRADIO_PATCH}")));
        }
        let (rows, cols, dim) = (he / CRADIO_PATCH, we / CRADIO_PATCH, self.dim);
        let patches_n = rows * cols;
        let tokens = self.num_prefix + patches_n;

        let patches = into_contiguous_cube(self.patch.forward::<F>(ctx, image)?)?.handle;
        let total = n * tokens * dim;
        let mut x = empty::<F>(ctx.client(), total);
        let geom = LaunchGeometry::elementwise(ctx.client(), total);
        unsafe {
            assemble_tokens_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(patches, n * patches_n * dim),
                BufferArg::from_raw_parts(self.prefix_tokens.clone(), self.num_prefix * dim),
                BufferArg::from_raw_parts(self.pos_table::<F>(ctx, rows, cols), tokens * dim),
                BufferArg::from_raw_parts(x.clone(), total),
                total as u32,
                self.num_prefix as u32,
                patches_n as u32,
                dim as u32,
            );
        }
        for blk in &self.blocks {
            x = blk.forward::<F>(ctx, x, n, tokens)?;
        }

        let len = n * patches_n * dim;
        let out = empty::<F>(ctx.client(), len);
        let geom = LaunchGeometry::elementwise(ctx.client(), len);
        unsafe {
            drop_prefix_normalize_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(x, total),
                BufferArg::from_raw_parts(self.mean.clone(), dim),
                BufferArg::from_raw_parts(self.inv_std.clone(), dim),
                BufferArg::from_raw_parts(out.clone(), len),
                len as u32,
                self.num_prefix as u32,
                patches_n as u32,
                dim as u32,
            );
        }
        Ok(ctx.tensor_from_handle(out, [n, rows, cols, dim], elem_dtype::<F>()))
    }
}
