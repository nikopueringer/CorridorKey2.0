//! Plain pre-norm ViT pieces shared by the frozen backbones (DINOv2 in MoGe-2, RVM, C-RADIO):
//! LayerNorm with bias, linear projections, self- and cross-attention, the GELU MLP, the ViT
//! block with optional LayerScale, and the bicubic position-table resize of
//! `interpolate_pos_encoding`.
//!
//! Linear layers return the bare GEMM output; their bias is applied by the kernel that consumes
//! it (head split, GELU, residual), so no pass exists only to add a bias.

use burn_tensor::Tensor;
use corridorkey_base::core::{
    CkFloat, cast,
    buffer::{empty, upload},
};
use corridorkey_core::{CkError, CkResult, compute::LaunchGeometry};
use cubecl::{prelude::*, server::Handle};

use crate::{
    hub::SafetensorsFile,
    models::blocks::{
        attn_core::run_unmasked_sdpa_handles,
        kernels::{
            bias_gelu_kernel, bias_layerscale_residual_kernel, layer_norm_kernel, merge_heads_kernel,
            split_heads_kernel,
        },
    },
    runtime::{MlContext, elem_dtype, into_contiguous_cube},
};

/// `nn.LayerNorm` epsilon of DINOv2 / timm ViTs.
pub const VIT_LAYER_NORM_EPS: f64 = 1e-6;
/// PyTorch's default `nn.LayerNorm` epsilon (RVM).
pub const TORCH_LAYER_NORM_EPS: f64 = 1e-5;

/// LayerNorm with bias over the last dimension.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Handle,
    bias: Handle,
    dim: usize,
    eps: f64,
}

impl LayerNorm {
    pub fn load<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, prefix: &str, eps: f64) -> CkResult<Self> {
        let (_, w) = st.read_as::<F>(&format!("{prefix}.weight"))?;
        let (_, b) = st.read_as::<F>(&format!("{prefix}.bias"))?;
        if w.len() != b.len() {
            return Err(CkError::Weights(format!("LayerNorm '{prefix}': weight / bias length mismatch")));
        }
        Ok(Self { dim: w.len(), weight: upload(ctx.client(), &w), bias: upload(ctx.client(), &b), eps })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Normalizes rows of `input` into `output` (see [`layer_norm_kernel`] for the row mapping).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_into<F: CkFloat>(
        &self,
        ctx: &MlContext,
        input: &Handle,
        input_len: usize,
        output: &Handle,
        output_len: usize,
        out_rows: usize,
        (in_group, out_group): (usize, usize),
        (out_stride, out_offset): (usize, usize),
    ) {
        let geom = LaunchGeometry::per_row(ctx.client(), out_rows, self.dim);
        let block_size = geom.cube_dim.x;
        unsafe {
            layer_norm_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(input.clone(), input_len),
                BufferArg::from_raw_parts(self.weight.clone(), self.dim),
                BufferArg::from_raw_parts(self.bias.clone(), self.dim),
                BufferArg::from_raw_parts(output.clone(), output_len),
                out_rows as u32,
                self.dim as u32,
                in_group as u32,
                out_group as u32,
                out_stride as u32,
                out_offset as u32,
                self.eps as f32,
                block_size,
            );
        }
    }

    /// LayerNorm of every row of a contiguous `[rows, dim]` buffer into a new buffer.
    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, input: &Handle, rows: usize) -> Handle {
        let len = rows * self.dim;
        let out = empty::<F>(ctx.client(), len);
        self.forward_into::<F>(ctx, input, len, &out, len, rows, (1, 1), (self.dim, 0));
        out
    }
}

/// `y = x · Wᵀ` with `W` stored transposed for the GEMM; the bias (if any) is kept for the
/// consuming kernel.
#[derive(Clone, Debug)]
pub struct Linear {
    w_t: Tensor<2>,
    pub bias: Handle,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl Linear {
    /// Rows `[out, in]` of the stacked PyTorch weights `names` (e.g. `query`, `key`, `value` into one
    /// `qkv`), and their stacked biases (zeros where a layer has none).
    pub fn load_stacked<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, names: &[String], bias: bool) -> CkResult<Self> {
        let (mut w_all, mut b_all, mut in_dim) = (Vec::new(), Vec::new(), 0);
        for name in names {
            let (shape, w) = st.read_as::<F>(&format!("{name}.weight"))?;
            if shape.len() != 2 || (in_dim != 0 && shape[1] != in_dim) {
                return Err(CkError::Weights(format!("linear '{name}': shape {shape:?}")));
            }
            in_dim = shape[1];
            w_all.extend(w);
            if bias {
                b_all.extend(st.read_as::<F>(&format!("{name}.bias"))?.1);
            } else {
                b_all.extend(std::iter::repeat_n(cast::<F>(0.0), shape[0]));
            }
        }
        let out_dim = b_all.len();
        Ok(Self {
            w_t: ctx.upload_transposed(&w_all, out_dim, in_dim)?,
            bias: upload(ctx.client(), &b_all),
            in_dim,
            out_dim,
        })
    }

    pub fn load<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, name: &str, bias: bool) -> CkResult<Self> {
        Self::load_stacked::<F>(ctx, st, &[name.to_string()], bias)
    }

    /// The bare product of contiguous `[rows, in_dim]` `x`: a new `[rows, out_dim]` buffer.
    pub fn gemm<F: CkFloat>(&self, ctx: &MlContext, x: Handle, rows: usize) -> CkResult<Handle> {
        let x = ctx.tensor_from_handle(x, [rows, self.in_dim], elem_dtype::<F>());
        Ok(into_contiguous_cube(ctx.matmul(x, self.w_t.clone()))?.handle)
    }
}

/// Multi-head attention: self-attention with a fused `qkv` projection, or cross-attention with `q`
/// from the queries and a fused `kv` projection of another sequence of the same length.
#[derive(Clone, Debug)]
pub struct Attention {
    q_or_qkv: Linear,
    kv: Option<Linear>,
    pub out: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn self_attention(qkv: Linear, out: Linear, heads: usize) -> CkResult<Self> {
        let inner = qkv.out_dim / 3;
        if qkv.out_dim != 3 * inner || !inner.is_multiple_of(heads) || out.in_dim != inner {
            return Err(CkError::Weights(format!("attention: qkv {} / out {} / heads {heads}", qkv.out_dim, out.in_dim)));
        }
        Ok(Self { q_or_qkv: qkv, kv: None, out, heads, head_dim: inner / heads })
    }

    pub fn cross_attention(q: Linear, kv: Linear, out: Linear, heads: usize) -> CkResult<Self> {
        let inner = q.out_dim;
        if kv.out_dim != 2 * inner || !inner.is_multiple_of(heads) || out.in_dim != inner {
            return Err(CkError::Weights(format!("cross-attention: q {inner} / kv {} / heads {heads}", kv.out_dim)));
        }
        Ok(Self { q_or_qkv: q, kv: Some(kv), out, heads, head_dim: inner / heads })
    }

    fn split<F: CkFloat>(&self, ctx: &MlContext, src: &Handle, src_cols: usize, bias: &Handle, offset: usize, (n, tokens): (usize, usize)) -> Handle {
        let len = n * tokens * self.heads * self.head_dim;
        let dst = empty::<F>(ctx.client(), len);
        let geom = LaunchGeometry::elementwise(ctx.client(), len);
        unsafe {
            split_heads_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(src.clone(), n * tokens * src_cols),
                BufferArg::from_raw_parts(bias.clone(), src_cols),
                BufferArg::from_raw_parts(dst.clone(), len),
                len as u32,
                self.heads as u32,
                tokens as u32,
                self.head_dim as u32,
                src_cols as u32,
                offset as u32,
            );
        }
        dst
    }

    /// Attention of `x` (`[n, tokens, dim]`, already normed) over itself or over `kv_src` (same
    /// shape); returns the output projection's bare product (`self.out.bias` not yet added).
    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x: Handle, kv_src: Option<Handle>, n: usize, tokens: usize) -> CkResult<Handle> {
        let rows = n * tokens;
        let inner = self.heads * self.head_dim;
        let shape = (n, tokens);
        let (q, k, v) = match (&self.kv, kv_src) {
            (None, None) => {
                let qkv = self.q_or_qkv.gemm::<F>(ctx, x, rows)?;
                let b = &self.q_or_qkv.bias;
                (
                    self.split::<F>(ctx, &qkv, 3 * inner, b, 0, shape),
                    self.split::<F>(ctx, &qkv, 3 * inner, b, inner, shape),
                    self.split::<F>(ctx, &qkv, 3 * inner, b, 2 * inner, shape),
                )
            }
            (Some(kv_lin), Some(src)) => {
                let qp = self.q_or_qkv.gemm::<F>(ctx, x, rows)?;
                let kv = kv_lin.gemm::<F>(ctx, src, rows)?;
                (
                    self.split::<F>(ctx, &qp, inner, &self.q_or_qkv.bias, 0, shape),
                    self.split::<F>(ctx, &kv, 2 * inner, &kv_lin.bias, 0, shape),
                    self.split::<F>(ctx, &kv, 2 * inner, &kv_lin.bias, inner, shape),
                )
            }
            _ => return Err(CkError::InvalidConfig("cross-attention needs its key / value sequence".into())),
        };
        let attn = run_unmasked_sdpa_handles::<F>(ctx, q, k, v, n, self.heads, tokens, self.head_dim)?;
        let len = rows * inner;
        let merged = empty::<F>(ctx.client(), len);
        let geom = LaunchGeometry::elementwise(ctx.client(), len);
        unsafe {
            merge_heads_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(attn, len),
                BufferArg::from_raw_parts(merged.clone(), len),
                len as u32,
                self.heads as u32,
                tokens as u32,
                self.head_dim as u32,
            );
        }
        self.out.gemm::<F>(ctx, merged, rows)
    }
}

/// `fc2(gelu(fc1(x)))`; returns `fc2`'s bare product.
#[derive(Clone, Debug)]
pub struct Mlp {
    fc1: Linear,
    pub fc2: Linear,
}

impl Mlp {
    pub fn new(fc1: Linear, fc2: Linear) -> Self {
        Self { fc1, fc2 }
    }

    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x: Handle, rows: usize) -> CkResult<Handle> {
        let h = self.fc1.gemm::<F>(ctx, x, rows)?;
        let (len, hidden) = (rows * self.fc1.out_dim, self.fc1.out_dim);
        let act = empty::<F>(ctx.client(), len);
        let geom = LaunchGeometry::elementwise(ctx.client(), len);
        unsafe {
            bias_gelu_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(h, len),
                BufferArg::from_raw_parts(self.fc1.bias.clone(), hidden),
                BufferArg::from_raw_parts(act.clone(), len),
                len as u32,
                hidden as u32,
            );
        }
        self.fc2.gemm::<F>(ctx, act, rows)
    }
}

/// `residual + gamma · (branch + bias)` into a new buffer (`gamma` = ones without LayerScale).
pub fn residual_add<F: CkFloat>(ctx: &MlContext, residual: &Handle, branch: &Handle, bias: &Handle, gamma: &Handle, len: usize, dim: usize) -> Handle {
    let out = empty::<F>(ctx.client(), len);
    let geom = LaunchGeometry::elementwise(ctx.client(), len);
    unsafe {
        bias_layerscale_residual_kernel::launch::<F>(
            ctx.client(),
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(residual.clone(), len),
            BufferArg::from_raw_parts(branch.clone(), len),
            BufferArg::from_raw_parts(bias.clone(), dim),
            BufferArg::from_raw_parts(gamma.clone(), dim),
            BufferArg::from_raw_parts(out.clone(), len),
            len as u32,
            dim as u32,
        );
    }
    out
}

/// A `dim`-long vector of ones (the LayerScale of blocks without one).
pub fn ones<F: CkFloat>(ctx: &MlContext, dim: usize) -> Handle {
    upload(ctx.client(), &vec![cast::<F>(1.0); dim])
}

/// Pre-norm ViT block: `x + ls1 · attn(norm1(x))`, then `x + ls2 · mlp(norm2(x))`.
#[derive(Clone, Debug)]
pub struct VitBlock {
    norm1: LayerNorm,
    attn: Attention,
    ls1: Handle,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: Handle,
    dim: usize,
}

impl VitBlock {
    /// timm / DINOv2 naming: `norm1`, `attn.qkv`, `attn.proj`, `ls1.gamma`, `norm2`, `mlp.fc1`,
    /// `mlp.fc2`, `ls2.gamma` (LayerScale optional: absent in timm ViTs built without
    /// `init_values`, e.g. C-RADIO's SO400M); LayerNorm eps 1e-6.
    pub fn load<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, prefix: &str, heads: usize) -> CkResult<Self> {
        let lin = |n: &str| Linear::load::<F>(ctx, st, &format!("{prefix}.{n}"), true);
        let attn = Attention::self_attention(lin("attn.qkv")?, lin("attn.proj")?, heads)?;
        let dim = attn.out.out_dim;
        let gamma = |n: &str| -> CkResult<Handle> {
            let name = format!("{prefix}.{n}.gamma");
            if st.entry(&name).is_err() {
                return Ok(ones::<F>(ctx, dim));
            }
            Ok(upload(ctx.client(), &st.read_as::<F>(&name)?.1))
        };
        Ok(Self {
            dim,
            norm1: LayerNorm::load::<F>(ctx, st, &format!("{prefix}.norm1"), VIT_LAYER_NORM_EPS)?,
            attn,
            ls1: gamma("ls1")?,
            norm2: LayerNorm::load::<F>(ctx, st, &format!("{prefix}.norm2"), VIT_LAYER_NORM_EPS)?,
            mlp: Mlp::new(lin("mlp.fc1")?, lin("mlp.fc2")?),
            ls2: gamma("ls2")?,
        })
    }

    /// RVM naming (`attention_norm`, `attention.{query,key,value,out}`, `mlp_norm`,
    /// `mlp.dense_in`, `mlp.dense_out`), no LayerScale, PyTorch's default LayerNorm eps.
    pub fn load_rvm<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, prefix: &str, heads: usize) -> CkResult<Self> {
        let p = |n: &str| format!("{prefix}.{n}");
        let qkv = Linear::load_stacked::<F>(ctx, st, &[p("attention.query"), p("attention.key"), p("attention.value")], true)?;
        let attn = Attention::self_attention(qkv, Linear::load::<F>(ctx, st, &p("attention.out"), true)?, heads)?;
        let dim = attn.out.out_dim;
        Ok(Self {
            norm1: LayerNorm::load::<F>(ctx, st, &p("attention_norm"), TORCH_LAYER_NORM_EPS)?,
            attn,
            ls1: ones::<F>(ctx, dim),
            norm2: LayerNorm::load::<F>(ctx, st, &p("mlp_norm"), TORCH_LAYER_NORM_EPS)?,
            mlp: Mlp::new(
                Linear::load::<F>(ctx, st, &p("mlp.dense_in"), true)?,
                Linear::load::<F>(ctx, st, &p("mlp.dense_out"), true)?,
            ),
            ls2: ones::<F>(ctx, dim),
            dim,
        })
    }

    /// `x`: contiguous `[n, tokens, dim]`.
    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x: Handle, n: usize, tokens: usize) -> CkResult<Handle> {
        let rows = n * tokens;
        let len = rows * self.dim;
        let h1 = self.norm1.forward::<F>(ctx, &x, rows);
        let a = self.attn.forward::<F>(ctx, h1, None, n, tokens)?;
        let x1 = residual_add::<F>(ctx, &x, &a, &self.attn.out.bias, &self.ls1, len, self.dim);
        let h2 = self.norm2.forward::<F>(ctx, &x1, rows);
        let m = self.mlp.forward::<F>(ctx, h2, rows)?;
        Ok(residual_add::<F>(ctx, &x1, &m, &self.mlp.fc2.bias, &self.ls2, len, self.dim))
    }
}

/// Resizes a `[grid, grid, dim]` position table to `[rows, cols, dim]` with PyTorch's bicubic
/// (`A = -0.75`, `align_corners = False`, edge clamped, no antialias), driven by the scale factors
/// `(rows + offset) / grid`, `(cols + offset) / grid`, computed in f64 on the host. DINOv2's
/// `interpolate_pos_encoding` uses `offset = 0.1`; `offset = 0` is `interpolate(size=…)` (RVM).
pub fn resize_pos_embed_bicubic(
    table: &[f64],
    grid: usize,
    dim: usize,
    (rows, cols): (usize, usize),
    offset: f64,
) -> Vec<f64> {
    if offset == 0.0 && rows == grid && cols == grid {
        return table.to_vec(); // PyTorch returns the table itself at its own size
    }
    const A: f64 = -0.75;
    let near = |x: f64| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let far = |x: f64| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    // Source taps and weights along one axis of `out` samples.
    let axis = |out: usize| -> Vec<([usize; 4], [f64; 4])> {
        let scale = grid as f64 / (out as f64 + offset); // 1 / scale_factor
        (0..out)
            .map(|o| {
                let src = scale * (o as f64 + 0.5) - 0.5;
                let i0 = src.floor();
                let t = src - i0;
                let w = [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)];
                let idx = std::array::from_fn(|k| (i0 as i64 - 1 + k as i64).clamp(0, grid as i64 - 1) as usize);
                (idx, w)
            })
            .collect()
    };
    let (ys, xs) = (axis(rows), axis(cols));
    let mut out = vec![0.0; rows * cols * dim];
    for (r, (yi, yw)) in ys.iter().enumerate() {
        for (c, (xi, xw)) in xs.iter().enumerate() {
            let dst = &mut out[(r * cols + c) * dim..(r * cols + c + 1) * dim];
            for ky in 0..4 {
                for kx in 0..4 {
                    let w = yw[ky] * xw[kx];
                    let src = &table[(yi[ky] * grid + xi[kx]) * dim..(yi[ky] * grid + xi[kx] + 1) * dim];
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d += w * s;
                    }
                }
            }
        }
    }
    out
}

/// Resizes a `[in_h, in_w, dim]` table to `[out_h, out_w, dim]` like PyTorch's
/// `interpolate(mode="bilinear", align_corners=False)` without antialias (each output samples its
/// two nearest inputs per axis, source index clamped at 0), in f64 on the host.
pub fn resize_table_bilinear(table: &[f64], (in_h, in_w): (usize, usize), dim: usize, (out_h, out_w): (usize, usize)) -> Vec<f64> {
    let axis = |input: usize, out: usize| -> Vec<(usize, usize, f64)> {
        let scale = input as f64 / out as f64;
        (0..out)
            .map(|o| {
                let src = (scale * (o as f64 + 0.5) - 0.5).max(0.0);
                let i0 = (src.floor() as usize).min(input - 1);
                let i1 = (i0 + 1).min(input - 1);
                (i0, i1, src - i0 as f64)
            })
            .collect()
    };
    let (ys, xs) = (axis(in_h, out_h), axis(in_w, out_w));
    let mut out = vec![0.0; out_h * out_w * dim];
    for (r, &(y0, y1, ly)) in ys.iter().enumerate() {
        for (c, &(x0, x1, lx)) in xs.iter().enumerate() {
            let taps = [(y0, x0, (1.0 - ly) * (1.0 - lx)), (y0, x1, (1.0 - ly) * lx), (y1, x0, ly * (1.0 - lx)), (y1, x1, ly * lx)];
            let dst = &mut out[(r * out_w + c) * dim..(r * out_w + c + 1) * dim];
            for (y, x, w) in taps {
                let src = &table[(y * in_w + x) * dim..(y * in_w + x + 1) * dim];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += w * s;
                }
            }
        }
    }
    out
}
