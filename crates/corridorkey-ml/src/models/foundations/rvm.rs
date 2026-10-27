//! RVM-small (Recurrent Video MAE, Google DeepMind `representations4d`, Apache-2.0) as
//! `corridorkey/rvm.py` runs it: a ViT-S encoder over every frame (batched), then the gated
//! cross-attention core frame after frame, carrying a `[1 + tokens, 384]` state. The feature of a
//! frame is the new state without its CLS slot, normalised by `rvm_mean` / `rvm_std` (fused into
//! the state update). The MAE decoder is not part of the converted weights.
//!
//! The product runs RVM over the whole clip once, continuously (state carried across its chunks
//! of frames), on the display frame resized bilinearly to H/4 × W/4 (`precompute_rvm_to_memmap`).

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use burn_tensor::Tensor;
use corridorkey_base::core::{
    CkFloat, cast, cast_all,
    buffer::{empty, upload},
};
use corridorkey_core::{CkError, CkResult, compute::LaunchGeometry};
use cubecl::{device::DeviceId, prelude::*, server::Handle};
use serde::Deserialize;

use super::vit::{Attention, LayerNorm, Linear, Mlp, TORCH_LAYER_NORM_EPS, VitBlock, ones, residual_add, resize_pos_embed_bicubic};
use crate::{
    hub::{FoundationStats, SafetensorsFile},
    models::blocks::{
        PatchifyConv,
        kernels::{assemble_tokens_kernel, copy_range_kernel, gru_gates_kernel, state_update_kernel},
    },
    runtime::{MlContext, elem_dtype, into_contiguous_cube},
};

/// RVM patch size and the token grid its sinusoidal position table is defined on.
pub const RVM_PATCH: usize = 16;
const RVM_BASE_GRID: usize = 16;
/// `state_layer_norm = nn.LayerNorm(d_model, eps=1e-4)` (scale only; its bias is zero).
const STATE_LN_EPS: f64 = 1e-4;
const STATS_MIN_STD: f64 = 1e-6;

#[derive(Deserialize)]
struct RvmConfig {
    embed_dim: usize,
    enc_num_layers: usize,
    enc_num_heads: usize,
    rnn_num_xa_blocks: usize,
    rnn_num_heads: usize,
}

/// `x + ca(ca_norm(x), kv)`, then `x + mlp(mlp_norm(x))`, then `x + attn(attention_norm(x))`.
#[derive(Clone, Debug)]
struct XaBlock {
    ca_norm: LayerNorm,
    ca: Attention,
    mlp_norm: LayerNorm,
    mlp: Mlp,
    attn_norm: LayerNorm,
    attn: Attention,
}

/// Device position tables per `(device, rows, cols, element size)`.
type PosCache = Arc<Mutex<HashMap<(DeviceId, usize, usize, usize), Handle>>>;

/// RVM-small encoder and recurrent core, ready to run as `F`.
#[derive(Clone, Debug)]
pub struct RvmSmall {
    patch: PatchifyConv,
    cls: Handle,
    pos_base: Vec<f64>,
    pos_cache: PosCache,
    encoder: Vec<VitBlock>,
    encoder_norm: LayerNorm,
    gate_x: Linear,
    gate_s: Linear,
    state_norm: LayerNorm,
    blocks: Vec<XaBlock>,
    output_norm: LayerNorm,
    ones: Handle,
    mean: Handle,
    inv_std: Handle,
    dim: usize,
}

/// RVM's `SincosPosEmb` base table `[16, 16, dim]` (MAE-style 1-D table over the 256 positions,
/// even channels sin, odd cos), in f64.
fn sincos_base(dim: usize) -> Vec<f64> {
    let n = RVM_BASE_GRID * RVM_BASE_GRID;
    let mut t = vec![0.0; n * dim];
    for i in 0..n {
        for j in 0..dim {
            let angle = i as f64 / 10000f64.powf(2.0 * (j / 2) as f64 / dim as f64);
            t[i * dim + j] = if j % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    t
}

impl RvmSmall {
    /// Loads `rvm_small.safetensors` (`scripts/convert_foundations.py`) and the RVM feature stats.
    pub fn load<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, stats: &FoundationStats) -> CkResult<Self> {
        let cfg: RvmConfig = st
            .metadata()
            .get("config")
            .ok_or_else(|| CkError::Weights("RVM safetensors has no 'config' metadata".into()))
            .and_then(|c| serde_json::from_str(c).map_err(|e| CkError::Weights(format!("RVM config: {e}"))))?;
        let dim = cfg.embed_dim;
        if stats.rvm_dim() != dim {
            return Err(CkError::Weights(format!("rvm stats have {} channels, model {dim}", stats.rvm_dim())));
        }
        let (_, pw) = st.read_as::<F>("patch_embed.weight")?;
        let (_, pb) = st.read_as::<F>("patch_embed.bias")?;
        let patch = PatchifyConv::from_slices::<F>(ctx, 3, dim, RVM_PATCH, 1, &pw, &pb)?;
        let lin = |name: &str, bias: bool| Linear::load::<F>(ctx, st, name, bias);
        let stacked = |names: &[&str], bias: bool| {
            Linear::load_stacked::<F>(ctx, st, &names.iter().map(|s| s.to_string()).collect::<Vec<_>>(), bias)
        };
        let norm = |name: &str, eps: f64| LayerNorm::load::<F>(ctx, st, name, eps);

        let encoder = (0..cfg.enc_num_layers)
            .map(|i| VitBlock::load_rvm::<F>(ctx, st, &format!("encoder_layers.{i}"), cfg.enc_num_heads))
            .collect::<CkResult<Vec<_>>>()?;
        let blocks = (0..cfg.rnn_num_xa_blocks)
            .map(|i| -> CkResult<XaBlock> {
                let p = format!("rnn_core.transformer.xa_blocks.{i}");
                Ok(XaBlock {
                    ca_norm: norm(&format!("{p}.ca_attention_norm"), TORCH_LAYER_NORM_EPS)?,
                    ca: Attention::cross_attention(
                        lin(&format!("{p}.ca_attention.query"), true)?,
                        stacked(&[&format!("{p}.ca_attention.key"), &format!("{p}.ca_attention.value")], true)?,
                        lin(&format!("{p}.ca_attention.out"), true)?,
                        cfg.rnn_num_heads,
                    )?,
                    mlp_norm: norm(&format!("{p}.mlp_norm"), TORCH_LAYER_NORM_EPS)?,
                    mlp: Mlp::new(lin(&format!("{p}.mlp.dense_in"), true)?, lin(&format!("{p}.mlp.dense_out"), true)?),
                    attn_norm: norm(&format!("{p}.attention_norm"), TORCH_LAYER_NORM_EPS)?,
                    attn: Attention::self_attention(
                        stacked(
                            &[&format!("{p}.attention.query"), &format!("{p}.attention.key"), &format!("{p}.attention.value")],
                            true,
                        )?,
                        lin(&format!("{p}.attention.out"), true)?,
                        cfg.rnn_num_heads,
                    )?,
                })
            })
            .collect::<CkResult<Vec<_>>>()?;

        let inv_std: Vec<f64> = stats.rvm_std.iter().map(|&s| 1.0 / (s as f64).max(STATS_MIN_STD)).collect();
        let mean: Vec<f64> = stats.rvm_mean.iter().map(|&m| m as f64).collect();
        Ok(Self {
            patch,
            cls: upload(ctx.client(), &st.read_as::<F>("cls_token")?.1),
            pos_base: sincos_base(dim),
            pos_cache: Arc::new(Mutex::new(HashMap::new())),
            encoder,
            encoder_norm: norm("encoder_final_norm", TORCH_LAYER_NORM_EPS)?,
            gate_x: stacked(&["rnn_core.input_update", "rnn_core.input_reset"], false)?,
            gate_s: stacked(&["rnn_core.state_update", "rnn_core.state_reset"], false)?,
            state_norm: norm("rnn_core.state_layer_norm", STATE_LN_EPS)?,
            blocks,
            output_norm: norm("rnn_core.transformer.output_norm", TORCH_LAYER_NORM_EPS)?,
            ones: ones::<F>(ctx, dim),
            mean: upload(ctx.client(), &cast_all::<F>(&mean)),
            inv_std: upload(ctx.client(), &cast_all::<F>(&inv_std)),
            dim,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// `[1 + rows · cols, dim]` position table: a zero row for the CLS token (RVM does not encode
    /// its position), then the base sincos table resized to the grid.
    fn pos_table<F: CkFloat>(&self, ctx: &MlContext, rows: usize, cols: usize) -> Handle {
        let key = (ctx.device_id(), rows, cols, size_of::<F>());
        let mut cache = self.pos_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = cache.get(&key) {
            return h.clone();
        }
        let grid = resize_pos_embed_bicubic(&self.pos_base, RVM_BASE_GRID, self.dim, (rows, cols), 0.0);
        let table: Vec<F> = std::iter::repeat_n(0.0, self.dim).chain(grid).map(cast::<F>).collect();
        let h = upload(ctx.client(), &table);
        cache.insert(key, h.clone());
        h
    }

    /// A zero state for a `rows × cols` grid (`[1 + rows · cols, dim]`): the start of a clip.
    pub fn zero_state<F: CkFloat>(&self, ctx: &MlContext, rows: usize, cols: usize) -> Handle {
        upload(ctx.client(), &vec![cast::<F>(0.0); (1 + rows * cols) * self.dim])
    }

    /// Features of `frames` (`[T, He, We, 3]`, display sRGB in [0, 1], He and We multiples of 16), in
    /// order, starting from `state` (`[1 + tokens, dim]`; [`Self::zero_state`] at the start of a clip).
    /// Returns the normalised features `[T, He/16, We/16, dim]` and the state after the last frame.
    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, frames: Tensor<4>, state: Handle) -> CkResult<(Tensor<4>, Handle)> {
        let [t, he, we, _] = frames.dims();
        if !he.is_multiple_of(RVM_PATCH) || !we.is_multiple_of(RVM_PATCH) {
            return Err(CkError::Shape(format!("RVM input {he}x{we} is not a multiple of {RVM_PATCH}")));
        }
        let (rows, cols, dim) = (he / RVM_PATCH, we / RVM_PATCH, self.dim);
        let (patches_n, tokens) = (rows * cols, rows * cols + 1);
        let frame_len = tokens * dim;

        // Encoder, batched over the frames.
        let patches = into_contiguous_cube(self.patch.forward::<F>(ctx, frames)?)?.handle;
        let total = t * frame_len;
        let mut x = empty::<F>(ctx.client(), total);
        let geom = LaunchGeometry::elementwise(ctx.client(), total);
        unsafe {
            assemble_tokens_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(patches, t * patches_n * dim),
                BufferArg::from_raw_parts(self.cls.clone(), dim),
                BufferArg::from_raw_parts(self.pos_table::<F>(ctx, rows, cols), frame_len),
                BufferArg::from_raw_parts(x.clone(), total),
                total as u32,
                1,
                patches_n as u32,
                dim as u32,
            );
        }
        for blk in &self.encoder {
            x = blk.forward::<F>(ctx, x, t, tokens)?;
        }
        let encoded = self.encoder_norm.forward::<F>(ctx, &x, t * tokens);

        // Recurrent core, frame after frame.
        let features = empty::<F>(ctx.client(), t * patches_n * dim);
        let geom_f = LaunchGeometry::elementwise(ctx.client(), frame_len);
        let mut state = state;
        for f in 0..t {
            let xf = empty::<F>(ctx.client(), frame_len);
            unsafe {
                copy_range_kernel::launch::<F>(
                    ctx.client(),
                    geom_f.cube_count.clone(),
                    geom_f.cube_dim,
                    BufferArg::from_raw_parts(encoded.clone(), total),
                    BufferArg::from_raw_parts(xf.clone(), frame_len),
                    frame_len as u32,
                    (f * frame_len) as u32,
                );
            }
            state = self.core_step::<F>(ctx, xf, state, tokens, &features, f * patches_n * dim)?;
        }
        let dt = elem_dtype::<F>();
        Ok((ctx.tensor_from_handle(features, [t, rows, cols, dim], dt), state))
    }

    /// One step of `_GatedTransformerCore`: returns the new state and writes the frame's
    /// normalised features at `feature_offset`.
    fn core_step<F: CkFloat>(
        &self,
        ctx: &MlContext,
        x: Handle,
        state: Handle,
        tokens: usize,
        features: &Handle,
        feature_offset: usize,
    ) -> CkResult<Handle> {
        let dim = self.dim;
        let len = tokens * dim;
        let gx = self.gate_x.gemm::<F>(ctx, x.clone(), tokens)?;
        let gs = self.gate_s.gemm::<F>(ctx, state.clone(), tokens)?;
        let ln_s = self.state_norm.forward::<F>(ctx, &state, tokens);
        let (update, gated) = (empty::<F>(ctx.client(), len), empty::<F>(ctx.client(), len));
        let geom = LaunchGeometry::elementwise(ctx.client(), len);
        unsafe {
            gru_gates_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count.clone(),
                geom.cube_dim,
                BufferArg::from_raw_parts(gx, 2 * len),
                BufferArg::from_raw_parts(gs, 2 * len),
                BufferArg::from_raw_parts(ln_s, len),
                BufferArg::from_raw_parts(update.clone(), len),
                BufferArg::from_raw_parts(gated.clone(), len),
                len as u32,
                dim as u32,
            );
        }

        // Cross-attention transformer over (queries = x, kv = gated state).
        let mut h = x;
        for blk in &self.blocks {
            let q = blk.ca_norm.forward::<F>(ctx, &h, tokens);
            let a = blk.ca.forward::<F>(ctx, q, Some(gated.clone()), 1, tokens)?;
            h = residual_add::<F>(ctx, &h, &a, &blk.ca.out.bias, &self.ones, len, dim);
            let m_in = blk.mlp_norm.forward::<F>(ctx, &h, tokens);
            let m = blk.mlp.forward::<F>(ctx, m_in, tokens)?;
            h = residual_add::<F>(ctx, &h, &m, &blk.mlp.fc2.bias, &self.ones, len, dim);
            let s_in = blk.attn_norm.forward::<F>(ctx, &h, tokens);
            let s = blk.attn.forward::<F>(ctx, s_in, None, 1, tokens)?;
            h = residual_add::<F>(ctx, &h, &s, &blk.attn.out.bias, &self.ones, len, dim);
        }
        let h = self.output_norm.forward::<F>(ctx, &h, tokens);

        let new_state = empty::<F>(ctx.client(), len);
        let features_len = feature_offset + (tokens - 1) * dim;
        unsafe {
            state_update_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(state, len),
                BufferArg::from_raw_parts(update, len),
                BufferArg::from_raw_parts(h, len),
                BufferArg::from_raw_parts(self.mean.clone(), dim),
                BufferArg::from_raw_parts(self.inv_std.clone(), dim),
                BufferArg::from_raw_parts(new_state.clone(), len),
                BufferArg::from_raw_parts(features.clone(), features_len),
                len as u32,
                dim as u32,
                feature_offset as u32,
            );
        }
        Ok(new_state)
    }
}
