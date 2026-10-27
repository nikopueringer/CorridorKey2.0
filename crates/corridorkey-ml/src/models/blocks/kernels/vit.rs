//! CubeCL kernels for the plain ViT blocks of the frozen backbones (DINOv2 in MoGe-2, C-RADIO):
//! LayerNorm with bias, GELU and LayerScale epilogues, token assembly, head split / merge.

use cubecl::prelude::*;

use corridorkey_base::core::CkFloat;
use corridorkey_core::compute::row_position;

/// LayerNorm over rows of width `dim` (f32 mean and variance whatever `F` is):
/// `(x - mean) / sqrt(var + eps) * weight + bias`.
///
/// Output row `r` reads input row `(r / out_group) * in_group + (in_group - out_group) + r % out_group`
/// (groups of `in_group` input rows keep their last `out_group`: with `1 + patches` / `patches` the
/// class token is skipped; `1` / `1` is the identity) and is written to
/// `output[r * out_stride + out_offset ..]`, so two norms can fill the halves of one concatenated row.
///
/// One cube per row (`block_size` units, a power of two), units striding over the channels.
#[cube(launch)]
pub fn layer_norm_kernel<F: CkFloat>(
    input: &[F],
    weight: &[F],
    bias: &[F],
    output: &mut [F],
    num_rows: u32,
    dim: u32,
    in_group: u32,
    out_group: u32,
    out_stride: u32,
    out_offset: u32,
    eps: f32,
    #[comptime] block_size: u32,
) {
    // Declared outside any branch (CUDA LLVM lowering).
    let mut partial = Shared::<[f32]>::new_slice(block_size as usize);
    let row = row_position();
    if row < num_rows {
        let tid = UNIT_POS_X;
        let in_row = (row / out_group) * in_group + (in_group - out_group) + row % out_group;
        let base = in_row * dim;
        let inv_dim = 1.0f32 / f32::cast_from(dim);

        let mut sum = 0.0f32;
        let mut c = tid;
        while c < dim {
            sum += f32::cast_from(input[(base + c) as usize]);
            c += block_size;
        }
        partial[tid as usize] = sum;
        sync_cube();
        // From the runtime `CUBE_DIM_X`, not the comptime `block_size` (dsp-kitchen §6b).
        let mut stride = CUBE_DIM_X / 2u32;
        while stride > 0u32 {
            if tid < stride {
                let s = partial[tid as usize] + partial[(tid + stride) as usize];
                partial[tid as usize] = s;
            }
            sync_cube();
            stride /= 2u32;
        }
        let mean = partial[0] * inv_dim;
        sync_cube();

        // Second pass: variance about the mean (no cancellation from E[x²] - E[x]²).
        let mut sq = 0.0f32;
        let mut c2 = tid;
        while c2 < dim {
            let d = f32::cast_from(input[(base + c2) as usize]) - mean;
            sq += d * d;
            c2 += block_size;
        }
        partial[tid as usize] = sq;
        sync_cube();
        let mut stride2 = CUBE_DIM_X / 2u32;
        while stride2 > 0u32 {
            if tid < stride2 {
                let s = partial[tid as usize] + partial[(tid + stride2) as usize];
                partial[tid as usize] = s;
            }
            sync_cube();
            stride2 /= 2u32;
        }
        let inv_std = 1.0f32 / f32::sqrt(partial[0] * inv_dim + eps);

        let out_base = row * out_stride + out_offset;
        let mut c3 = tid;
        while c3 < dim {
            let x = f32::cast_from(input[(base + c3) as usize]);
            let y = (x - mean) * inv_std * f32::cast_from(weight[c3 as usize])
                + f32::cast_from(bias[c3 as usize]);
            output[(out_base + c3) as usize] = F::cast_from(y);
            c3 += block_size;
        }
    }
}

/// `output[r, c] = gelu(input[r, c] + bias[c])`, exact GELU `x/2 · (1 + erf(x / √2))` (PyTorch's
/// `nn.GELU()` default), evaluated in f32.
#[cube(launch)]
pub fn bias_gelu_kernel<F: CkFloat>(input: &[F], bias: &[F], output: &mut [F], total: u32, cols: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let x = f32::cast_from(input[i as usize]) + f32::cast_from(bias[(i % cols) as usize]);
        output[i as usize] = F::cast_from(0.5f32 * x * (1.0f32 + Erf::erf(x * 0.70710678118654752f32)));
    }
}

/// LayerScale residual: `output[r, c] = residual[r, c] + gamma[c] · (branch[r, c] + bias[c])`.
#[cube(launch)]
pub fn bias_layerscale_residual_kernel<F: CkFloat>(
    residual: &[F],
    branch: &[F],
    bias: &[F],
    gamma: &[F],
    output: &mut [F],
    total: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let c = (i % cols) as usize;
        let y = f32::cast_from(branch[i as usize]) + f32::cast_from(bias[c]);
        let r = f32::cast_from(residual[i as usize]);
        output[i as usize] = F::cast_from(r + f32::cast_from(gamma[c]) * y);
    }
}

/// Builds `[n, prefix + patches, dim]` tokens: the `prefix` learned tokens (`cls`: `[prefix, dim]`,
/// e.g. a CLS token, or CLS + registers) then the patch embeddings, each plus its row of `pos`
/// (`[prefix + patches, dim]`; zero rows where a model does not encode a prefix token's position).
#[cube(launch)]
pub fn assemble_tokens_kernel<F: CkFloat>(
    patches: &[F],
    cls: &[F],
    pos: &[F],
    output: &mut [F],
    total: u32,
    prefix: u32,
    num_patches: u32,
    dim: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let c = i % dim;
        let tokens = prefix + num_patches;
        let t = (i / dim) % tokens;
        let n = i / (dim * tokens);
        let is_patch = t >= prefix;
        // In-bounds indices for both arms (an `if` expression may evaluate both).
        let p_idx = if is_patch { (n * num_patches + t - prefix) * dim + c } else { 0u32 };
        let c_idx = if is_patch { 0u32 } else { t * dim + c };
        let patch = patches[p_idx as usize];
        let token = cls[c_idx as usize];
        let base = if is_patch { patch } else { token };
        output[i as usize] = base + pos[(t * dim + c) as usize];
    }
}

/// Patch tokens of `[n, prefix + patches, dim]` as `[n, patches, dim]`, normalised per channel:
/// `(x - mean[c]) · inv_std[c]` (the feature statistics CorridorKey applies to a backbone).
#[cube(launch)]
pub fn drop_prefix_normalize_kernel<F: CkFloat>(
    tokens: &[F],
    mean: &[F],
    inv_std: &[F],
    output: &mut [F],
    total: u32,
    prefix: u32,
    num_patches: u32,
    dim: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let c = i % dim;
        let p = (i / dim) % num_patches;
        let n = i / (dim * num_patches);
        let x = f32::cast_from(tokens[((n * (prefix + num_patches) + prefix + p) * dim + c) as usize]);
        let y = (x - f32::cast_from(mean[c as usize])) * f32::cast_from(inv_std[c as usize]);
        output[i as usize] = F::cast_from(y);
    }
}

/// Gathers one head-split operand from a projection output: `dst[n, h, t, d] =
/// src[(n · tokens + t) · src_cols + col_offset + h · head_dim + d] + bias[col_offset + h · head_dim + d]`
/// (`dst`: `[n, heads, tokens, head_dim]`). A fused `qkv` projection is split by three launches with
/// offsets `0`, `dim`, `2 · dim`; a fused `kv` by two.
#[cube(launch)]
pub fn split_heads_kernel<F: CkFloat>(
    src: &[F],
    bias: &[F],
    dst: &mut [F],
    total: u32,
    heads: u32,
    tokens: u32,
    head_dim: u32,
    src_cols: u32,
    col_offset: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let d = i % head_dim;
        let t = (i / head_dim) % tokens;
        let h = (i / (head_dim * tokens)) % heads;
        let n = i / (head_dim * tokens * heads);
        let col = col_offset + h * head_dim + d;
        dst[i as usize] = src[((n * tokens + t) * src_cols + col) as usize] + bias[col as usize];
    }
}

/// Merges `[n, heads, tokens, head_dim]` back into `[n, tokens, heads · head_dim]`.
#[cube(launch)]
pub fn merge_heads_kernel<F: CkFloat>(
    attn: &[F],
    output: &mut [F],
    total: u32,
    heads: u32,
    tokens: u32,
    head_dim: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        // i indexes [n, tokens, heads, head_dim]
        let d = i % head_dim;
        let h = (i / head_dim) % heads;
        let t = (i / (head_dim * heads)) % tokens;
        let n = i / (head_dim * heads * tokens);
        output[i as usize] = attn[(((n * heads + h) * tokens + t) * head_dim + d) as usize];
    }
}
