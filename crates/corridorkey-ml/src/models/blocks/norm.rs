//! Channels-last RMSNorm (`RMSNorm` and `LayerNorm2d` in `corridorkey/blocks.py`).

use burn_tensor::Tensor;
use corridorkey_base::core::CkFloat;
use corridorkey_base::core::buffer::{empty, upload};
use corridorkey_core::compute::LaunchGeometry;
use corridorkey_core::{CkError, CkResult};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::hub::SafetensorsFile;
use crate::runtime::{MlContext, elem_dtype, into_contiguous_cube};

use super::kernels::rms_norm_kernel;

/// Default epsilon for `RMSNorm` and `LayerNorm2d` in `corridorkey/blocks.py`.
pub const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;

/// Dispatches `rms_norm_kernel` over `[num_rows, dim]` on the device.
pub fn rms_norm<F: CkFloat>(
    client: &Client,
    input: &Handle,
    weight: &Handle,
    output: &Handle,
    num_rows: usize,
    dim: usize,
    eps: f64,
) {
    if num_rows == 0 || dim == 0 {
        return;
    }
    let total = num_rows * dim;
    // The runtime sizes the cube for a row of `dim` channels (power of two for the reduction).
    let geom = LaunchGeometry::per_row(client, num_rows, dim);
    let block_size = geom.cube_dim.x;
    unsafe {
        rms_norm_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), total),
            BufferArg::from_raw_parts(weight.clone(), dim),
            BufferArg::from_raw_parts(output.clone(), total),
            num_rows as u32,
            dim as u32,
            eps as f32,
            block_size,
        );
    }
}

/// Device-resident RMSNorm layer over the last dimension `dim`.
#[derive(Clone, Debug)]
pub struct RmsNorm {
    weight: Handle,
    dim: usize,
    eps: f64,
}

impl RmsNorm {
    /// Creates an `RmsNorm` from a host weight slice of length `dim`.
    pub fn from_slice<F: CkFloat>(ctx: &MlContext, weight: &[F], eps: f64) -> CkResult<Self> {
        if weight.is_empty() {
            return Err(CkError::InvalidConfig(
                "RmsNorm weight cannot be empty".into(),
            ));
        }
        Ok(Self {
            weight: upload(ctx.client(), weight),
            dim: weight.len(),
            eps,
        })
    }

    /// Loads `<prefix>.weight` from a [`SafetensorsFile`] onto `ctx`.
    pub fn load<F: CkFloat>(
        ctx: &MlContext,
        st: &SafetensorsFile,
        prefix: &str,
        eps: f64,
    ) -> CkResult<Self> {
        let key = format!("{prefix}.weight");
        let (shape, data) = st.read_as::<F>(&key)?;
        if shape.len() != 1 {
            return Err(CkError::Weights(format!(
                "expected 1D weight for '{key}', got shape {shape:?}"
            )));
        }
        Self::from_slice(ctx, &data, eps)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    pub fn weight_handle(&self) -> &Handle {
        &self.weight
    }

    /// Normalizes a Burn `Tensor<D>` along its last dimension (`dims[D - 1] == self.dim`),
    /// returning a new device tensor of the same shape without leaving the GPU.
    pub fn forward<F: CkFloat, const D: usize>(
        &self,
        ctx: &MlContext,
        x: Tensor<D>,
    ) -> CkResult<Tensor<D>> {
        let shape = x.dims();
        let last = shape[D - 1];
        if last != self.dim {
            return Err(CkError::Shape(format!(
                "RmsNorm expected last dim {}, got {last} in shape {shape:?}",
                self.dim
            )));
        }
        let total: usize = shape.iter().product();
        let num_rows = total / self.dim;
        let cube = into_contiguous_cube(x)?;
        let out_handle = empty::<F>(ctx.client(), total);
        rms_norm::<F>(
            ctx.client(),
            &cube.handle,
            &self.weight,
            &out_handle,
            num_rows,
            self.dim,
            self.eps,
        );
        Ok(ctx.tensor_from_handle(out_handle, shape, elem_dtype::<F>()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use corridorkey_core::ComputeTarget;

    fn host_rms_norm(x: &[f64], w: &[f64], num_rows: usize, dim: usize, eps: f64) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        for r in 0..num_rows {
            let row = &x[r * dim..(r + 1) * dim];
            let mean_sq = row.iter().map(|&v| v * v).sum::<f64>() / dim as f64;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            for c in 0..dim {
                out[r * dim + c] = row[c] * inv_rms * w[c];
            }
        }
        out
    }

    #[test]
    fn rms_norm_matches_f64_reference_on_odd_shapes() {
        let (b, h, w, dim) = (2, 3, 5, 19); // odd spatial dims and odd channel count
        let num_rows = b * h * w;
        let total = num_rows * dim;
        let x32: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.23).sin() * 2.5)
            .collect();
        let w32: Vec<f32> = (0..dim).map(|c| 0.5 + (c as f32) * 0.07).collect();
        let x64: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
        let w64: Vec<f64> = w32.iter().map(|&v| v as f64).collect();
        let want = host_rms_norm(&x64, &w64, num_rows, dim, DEFAULT_RMS_NORM_EPS);

        for target in ComputeTarget::available() {
            let ctx = MlContext::new(target).unwrap();
            let norm = RmsNorm::from_slice(&ctx, &w32, DEFAULT_RMS_NORM_EPS).unwrap();
            let t_in = ctx.upload_tensor(&x32, [b, h, w, dim]).unwrap();
            let t_out = norm.forward::<f32, 4>(&ctx, t_in).unwrap();
            let got = ctx.download_tensor::<f32, 4>(t_out).unwrap();
            for (i, (&g, &ref_v)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g as f64 - ref_v).abs() < 1e-5,
                    "{} [{i}]: {g} vs {ref_v}",
                    target
                );
            }
        }
    }
}
