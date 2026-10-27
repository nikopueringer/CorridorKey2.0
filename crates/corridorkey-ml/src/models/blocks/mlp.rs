//! Gated MLP (`SwiGLU` in `corridorkey/blocks.py`).
//!
//! Computes `h = silu(x @ W_gate^T + b_gate) * (x @ W_up^T + b_up); out = h @ W_down^T + b_down`.
//! Following Rule 1 of the port plan, linear weights `[out_features, in_features]` are transposed
//! once at load time into `[in_features, out_features]` on the device so every forward pass calls
//! `matmul` directly on contiguous row-major buffers without per-step transpose views.

use burn_tensor::Tensor;
use corridorkey_base::core::CkFloat;
use corridorkey_base::core::buffer::{empty, upload};
use corridorkey_core::compute::LaunchGeometry;
use corridorkey_core::{CkError, CkResult};
use cubecl::prelude::BufferArg;
use cubecl::server::Handle;

use crate::hub::SafetensorsFile;
use crate::runtime::{MlContext, elem_dtype, into_contiguous_cube};

use super::kernels::{add_row_bias_kernel, add_row_bias_residual_kernel, swiglu_gate_fused_kernel};

/// Transposes a row-major `[rows, cols]` slice on the host into `[cols, rows]`.
pub fn transpose_2d<T: Copy>(data: &[T], rows: usize, cols: usize) -> Vec<T> {
    assert_eq!(data.len(), rows * cols);
    let mut out = Vec::with_capacity(data.len());
    for c in 0..cols {
        for r in 0..rows {
            out.push(data[r * cols + c]);
        }
    }
    out
}

/// Device-resident SwiGLU MLP (`dim -> hidden -> dim`) with horizontally-fused `[w_gate | w_up]`
/// projection (`[dim, 2 * hidden]`).
#[derive(Clone, Debug)]
pub struct SwiGlu {
    /// Pre-transposed horizontally-concatenated `[dim, 2 * hidden]` gate+up projection.
    w_gate_up_t: Tensor<2>,
    b_gate: Handle,
    b_up: Handle,
    /// Pre-transposed `[hidden, dim]` down projection.
    w_down_t: Tensor<2>,
    b_down: Handle,
    dim: usize,
    hidden: usize,
}

impl SwiGlu {
    /// Creates a `SwiGlu` from PyTorch-layout weights (`w_gate` and `w_up` of shape
    /// `[hidden, dim]`, `w_down` of shape `[dim, hidden]`), concatenating `[w_gate; w_up]` into
    /// `[2 * hidden, dim]` and transposing into `[dim, 2 * hidden]` once before uploading.
    #[allow(clippy::too_many_arguments)]
    pub fn from_slices<F: CkFloat>(
        ctx: &MlContext,
        dim: usize,
        hidden: usize,
        w_gate: &[F],
        b_gate: &[F],
        w_up: &[F],
        b_up: &[F],
        w_down: &[F],
        b_down: &[F],
    ) -> CkResult<Self> {
        if w_gate.len() != hidden * dim
            || w_up.len() != hidden * dim
            || w_down.len() != dim * hidden
        {
            return Err(CkError::Shape(format!(
                "SwiGlu weight length mismatch for dim={dim}, hidden={hidden}"
            )));
        }
        if b_gate.len() != hidden || b_up.len() != hidden || b_down.len() != dim {
            return Err(CkError::Shape(format!(
                "SwiGlu bias length mismatch for dim={dim}, hidden={hidden}"
            )));
        }
        let mut w_gate_up = Vec::with_capacity(2 * hidden * dim);
        w_gate_up.extend_from_slice(w_gate);
        w_gate_up.extend_from_slice(w_up);

        Ok(Self {
            w_gate_up_t: ctx.upload_transposed(&w_gate_up, 2 * hidden, dim)?,
            b_gate: upload(ctx.client(), b_gate),
            b_up: upload(ctx.client(), b_up),
            w_down_t: ctx.upload_transposed(w_down, dim, hidden)?,
            b_down: upload(ctx.client(), b_down),
            dim,
            hidden,
        })
    }

    /// Loads `<prefix>.w_{gate,up,down}.{weight,bias}` from a [`SafetensorsFile`] onto `ctx`.
    pub fn load<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, prefix: &str) -> CkResult<Self> {
        let (gate_shape, w_gate) = st.read_as::<F>(&format!("{prefix}.w_gate.weight"))?;
        let (_, b_gate) = st.read_as::<F>(&format!("{prefix}.w_gate.bias"))?;
        let (_, w_up) = st.read_as::<F>(&format!("{prefix}.w_up.weight"))?;
        let (_, b_up) = st.read_as::<F>(&format!("{prefix}.w_up.bias"))?;
        let (_, w_down) = st.read_as::<F>(&format!("{prefix}.w_down.weight"))?;
        let (_, b_down) = st.read_as::<F>(&format!("{prefix}.w_down.bias"))?;
        if gate_shape.len() != 2 {
            return Err(CkError::Weights(format!(
                "expected 2D w_gate.weight at '{prefix}', got {gate_shape:?}"
            )));
        }
        let (hidden, dim) = (gate_shape[0], gate_shape[1]);
        Self::from_slices(
            ctx, dim, hidden, &w_gate, &b_gate, &w_up, &b_up, &w_down, &b_down,
        )
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    fn forward_down_no_bias<F: CkFloat, const D: usize>(
        &self,
        ctx: &MlContext,
        x: Tensor<D>,
    ) -> CkResult<([usize; D], usize, Handle)> {
        let shape = x.dims();
        let last = shape[D - 1];
        if last != self.dim {
            return Err(CkError::Shape(format!(
                "SwiGlu expected last dim {}, got {last} in shape {shape:?}",
                self.dim
            )));
        }
        let total_in: usize = shape.iter().product();
        let num_rows = total_in / self.dim;
        let x2d: Tensor<2> = x.reshape([num_rows, self.dim]);

        let gate_up_2d = ctx.matmul(x2d, self.w_gate_up_t.clone());
        let total_hidden = num_rows * self.hidden;
        let c_gate_up = into_contiguous_cube(gate_up_2d)?;
        let h_handle = empty::<F>(ctx.client(), total_hidden);

        let geom_h = LaunchGeometry::elementwise(ctx.client(), total_hidden);
        unsafe {
            swiglu_gate_fused_kernel::launch::<F>(
                ctx.client(),
                geom_h.cube_count,
                geom_h.cube_dim,
                BufferArg::from_raw_parts(c_gate_up.handle, 2 * total_hidden),
                BufferArg::from_raw_parts(self.b_gate.clone(), self.hidden),
                BufferArg::from_raw_parts(self.b_up.clone(), self.hidden),
                BufferArg::from_raw_parts(h_handle.clone(), total_hidden),
                total_hidden as u32,
                self.hidden as u32,
            );
        }

        let h2d = ctx.tensor_from_handle(h_handle, [num_rows, self.hidden], elem_dtype::<F>());
        let out2d = ctx.matmul(h2d, self.w_down_t.clone());
        let c_out = into_contiguous_cube(out2d)?;
        Ok((shape, total_in, c_out.handle))
    }

    /// Executes SwiGLU over a `Tensor<D>` whose last axis is `self.dim`.
    pub fn forward<F: CkFloat, const D: usize>(
        &self,
        ctx: &MlContext,
        x: Tensor<D>,
    ) -> CkResult<Tensor<D>> {
        let (shape, total_in, out_handle) = self.forward_down_no_bias::<F, D>(ctx, x)?;
        let geom_out = LaunchGeometry::elementwise(ctx.client(), total_in);
        unsafe {
            add_row_bias_kernel::launch::<F>(
                ctx.client(),
                geom_out.cube_count,
                geom_out.cube_dim,
                BufferArg::from_raw_parts(out_handle.clone(), total_in),
                BufferArg::from_raw_parts(self.b_down.clone(), self.dim),
                total_in as u32,
                self.dim as u32,
            );
        }
        Ok(ctx.tensor_from_handle(out_handle, shape, elem_dtype::<F>()))
    }

    /// Executes `residual + SwiGLU(x)` with a fused `b_down + residual` epilogue.
    pub fn forward_residual<F: CkFloat, const D: usize>(
        &self,
        ctx: &MlContext,
        x: Tensor<D>,
        residual: Tensor<D>,
    ) -> CkResult<Tensor<D>> {
        let (shape, total_in, out_handle) = self.forward_down_no_bias::<F, D>(ctx, x)?;
        let c_res = into_contiguous_cube(residual)?;
        let geom_out = LaunchGeometry::elementwise(ctx.client(), total_in);
        unsafe {
            add_row_bias_residual_kernel::launch::<F>(
                ctx.client(),
                geom_out.cube_count,
                geom_out.cube_dim,
                BufferArg::from_raw_parts(out_handle.clone(), total_in),
                BufferArg::from_raw_parts(self.b_down.clone(), self.dim),
                BufferArg::from_raw_parts(c_res.handle, total_in),
                total_in as u32,
                self.dim as u32,
            );
        }
        Ok(ctx.tensor_from_handle(out_handle, shape, elem_dtype::<F>()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use corridorkey_core::ComputeTarget;

    #[allow(clippy::too_many_arguments)]
    fn host_swiglu(
        x: &[f64],
        num_rows: usize,
        dim: usize,
        hidden: usize,
        w_gate: &[f64],
        b_gate: &[f64],
        w_up: &[f64],
        b_up: &[f64],
        w_down: &[f64],
        b_down: &[f64],
    ) -> Vec<f64> {
        let mut out = vec![0.0; num_rows * dim];
        for r in 0..num_rows {
            let xr = &x[r * dim..(r + 1) * dim];
            let mut h = vec![0.0; hidden];
            for k in 0..hidden {
                let mut g = b_gate[k];
                let mut u = b_up[k];
                for c in 0..dim {
                    g += xr[c] * w_gate[k * dim + c];
                    u += xr[c] * w_up[k * dim + c];
                }
                let silu_g = g / (1.0 + (-g).exp());
                h[k] = silu_g * u;
            }
            for c in 0..dim {
                let mut d = b_down[c];
                for k in 0..hidden {
                    d += h[k] * w_down[c * hidden + k];
                }
                out[r * dim + c] = d;
            }
        }
        out
    }

    #[test]
    fn swiglu_matches_f64_reference() {
        let (num_rows, dim, hidden) = (7, 16, 40);
        let make_vec = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32) * seed + 0.1).sin() * 0.3)
                .collect()
        };
        let x32 = make_vec(num_rows * dim, 0.17);
        let wg32 = make_vec(hidden * dim, 0.11);
        let bg32 = make_vec(hidden, 0.29);
        let wu32 = make_vec(hidden * dim, 0.13);
        let bu32 = make_vec(hidden, 0.31);
        let wd32 = make_vec(dim * hidden, 0.07);
        let bd32 = make_vec(dim, 0.19);

        let to64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };
        let want = host_swiglu(
            &to64(&x32),
            num_rows,
            dim,
            hidden,
            &to64(&wg32),
            &to64(&bg32),
            &to64(&wu32),
            &to64(&bu32),
            &to64(&wd32),
            &to64(&bd32),
        );

        for target in ComputeTarget::available() {
            let ctx = MlContext::new(target).unwrap();
            let mlp =
                SwiGlu::from_slices(&ctx, dim, hidden, &wg32, &bg32, &wu32, &bu32, &wd32, &bd32)
                    .unwrap();
            let t_in = ctx.upload_tensor(&x32, [num_rows, dim]).unwrap();
            let t_out = mlp.forward::<f32, 2>(&ctx, t_in).unwrap();
            let got = ctx.download_tensor::<f32, 2>(t_out).unwrap();
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!((g as f64 - w).abs() < 1e-4, "{} [{i}]: {g} vs {w}", target);
            }
        }
    }
}
