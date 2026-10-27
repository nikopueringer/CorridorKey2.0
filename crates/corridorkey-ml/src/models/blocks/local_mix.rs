//! Depthwise spatial modules on NHWC/NTHWC tokens (`QKVDepthwiseSpatial` and
//! `DepthwiseSwiGLU2D` in `corridorkey/blocks.py`).

use burn_tensor::Tensor;
use corridorkey_base::core::CkFloat;
use corridorkey_base::core::buffer::{empty, upload};
use corridorkey_core::compute::LaunchGeometry;
use corridorkey_core::{CkError, CkResult};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::hub::SafetensorsFile;
use crate::runtime::{MlContext, elem_dtype, into_contiguous_cube};

use super::kernels::{depthwise_conv2d_nhwc_kernel, depthwise_swiglu_2d_kernel};
use super::norm::{DEFAULT_RMS_NORM_EPS, RmsNorm};

/// Transposes a PyTorch depthwise weight `[C, 1, K, K]` into tap-major `[K * K, C]` order so
/// threads for adjacent channels `c` perform coalesced reads.
pub fn depthwise_weight_to_tap_major<T: Copy>(
    weight: &[T],
    channels: usize,
    kernel_size: usize,
) -> Vec<T> {
    let taps = kernel_size * kernel_size;
    assert_eq!(weight.len(), channels * taps);
    let mut out = Vec::with_capacity(weight.len());
    for t in 0..taps {
        for c in 0..channels {
            out.push(weight[c * taps + t]);
        }
    }
    out
}

/// Device-resident `QKVDepthwiseSpatial` filter over `[B, H, W, 3 * dim]` (or `[B, T, H, W, 3 * dim]`).
#[derive(Clone, Debug)]
pub struct QkvDepthwiseSpatial {
    weight_tap_major: Handle,
    bias: Handle,
    channels: usize,
    kernel_size: usize,
}

impl QkvDepthwiseSpatial {
    pub fn from_slices<F: CkFloat>(
        ctx: &MlContext,
        dim: usize,
        kernel_size: usize,
        weight: &[F],
        bias: &[F],
    ) -> CkResult<Self> {
        let channels = 3 * dim;
        let taps = kernel_size * kernel_size;
        if kernel_size == 0 || kernel_size.is_multiple_of(2) {
            return Err(CkError::InvalidConfig(format!(
                "QkvDepthwiseSpatial kernel_size ({kernel_size}) must be odd"
            )));
        }
        if weight.len() != channels * taps || bias.len() != channels {
            return Err(CkError::Shape(format!(
                "QkvDepthwiseSpatial weight/bias mismatch for channels={channels}, kernel={kernel_size}"
            )));
        }
        let tap_major = depthwise_weight_to_tap_major(weight, channels, kernel_size);
        Ok(Self {
            weight_tap_major: upload(ctx.client(), &tap_major),
            bias: upload(ctx.client(), bias),
            channels,
            kernel_size,
        })
    }

    pub fn load<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, prefix: &str) -> CkResult<Self> {
        let (shape, w) = st.read_as::<F>(&format!("{prefix}.dw.weight"))?;
        let (_, b) = st.read_as::<F>(&format!("{prefix}.dw.bias"))?;
        if shape.len() != 4 || shape[1] != 1 || shape[2] != shape[3] || !shape[0].is_multiple_of(3)
        {
            return Err(CkError::Weights(format!(
                "unexpected QKVDepthwiseSpatial shape {shape:?} at '{prefix}'"
            )));
        }
        Self::from_slices(ctx, shape[0] / 3, shape[2], &w, &b)
    }

    /// Applies the depthwise spatial filter to a `[B, H, W, 3*dim]` or `[B, T, H, W, 3*dim]` tensor,
    /// optionally adding `in_bias` (`[3 * dim]`) to each input tap before filtering.
    pub fn forward_with_in_bias<F: CkFloat, const D: usize>(
        &self,
        ctx: &MlContext,
        qkv: Tensor<D>,
        in_bias: Option<&Handle>,
    ) -> CkResult<Tensor<D>> {
        let shape = qkv.dims();
        let (batches, h, w, c) = match D {
            4 => (shape[0], shape[1], shape[2], shape[3]),
            5 => (shape[0] * shape[1], shape[2], shape[3], shape[4]),
            _ => {
                return Err(CkError::Shape(format!(
                    "QkvDepthwiseSpatial expects 4D or 5D tensor, got {shape:?}"
                )));
            }
        };
        if c != self.channels {
            return Err(CkError::Shape(format!(
                "QkvDepthwiseSpatial expected {} channels, got {c}",
                self.channels
            )));
        }
        let total = batches * h * w * c;
        let pad = self.kernel_size / 2;
        let reflect = u32::from(h > pad && w > pad);
        let cube = into_contiguous_cube(qkv)?;
        let out = empty::<F>(ctx.client(), total);
        let geom = LaunchGeometry::elementwise(ctx.client(), total);
        let taps = self.kernel_size * self.kernel_size;
        let (in_b_handle, apply_in_b) = match in_bias {
            Some(bh) => (bh.clone(), 1u32),
            None => (self.bias.clone(), 0u32),
        };
        unsafe {
            depthwise_conv2d_nhwc_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(cube.handle, total),
                BufferArg::from_raw_parts(in_b_handle, c),
                BufferArg::from_raw_parts(self.weight_tap_major.clone(), taps * c),
                BufferArg::from_raw_parts(self.bias.clone(), c),
                BufferArg::from_raw_parts(out.clone(), total),
                total as u32,
                h as u32,
                w as u32,
                c as u32,
                apply_in_b,
                self.kernel_size as u32,
                reflect,
            );
        }
        Ok(ctx.tensor_from_handle(out, shape, elem_dtype::<F>()))
    }

    /// Applies the depthwise spatial filter to a `[B, H, W, 3*dim]` or `[B, T, H, W, 3*dim]` tensor.
    pub fn forward<F: CkFloat, const D: usize>(
        &self,
        ctx: &MlContext,
        qkv: Tensor<D>,
    ) -> CkResult<Tensor<D>> {
        self.forward_with_in_bias::<F, D>(ctx, qkv, None)
    }
}

/// Device-resident `DepthwiseSwiGLU2D` (`local_mix` in Stage 1 and Stage 2 blocks).
#[derive(Clone, Debug)]
pub struct DepthwiseSwiGlu2d {
    norm: RmsNorm,
    value_w_tap_major: Handle,
    value_b: Handle,
    gate_w_tap_major: Handle,
    gate_b: Handle,
    gamma: Handle,
    dim: usize,
    kernel_size: usize,
}

impl DepthwiseSwiGlu2d {
    #[allow(clippy::too_many_arguments)]
    pub fn from_slices<F: CkFloat>(
        ctx: &MlContext,
        dim: usize,
        kernel_size: usize,
        norm_w: &[F],
        value_w: &[F],
        value_b: &[F],
        gate_w: &[F],
        gate_b: &[F],
        gamma: &[F],
    ) -> CkResult<Self> {
        let taps = kernel_size * kernel_size;
        if value_w.len() != dim * taps || gate_w.len() != dim * taps {
            return Err(CkError::Shape(
                "DepthwiseSwiGlu2d kernel weight length mismatch".into(),
            ));
        }
        if value_b.len() != dim || gate_b.len() != dim || gamma.len() != dim {
            return Err(CkError::Shape(
                "DepthwiseSwiGlu2d bias/gamma length mismatch".into(),
            ));
        }
        let vw_tap = depthwise_weight_to_tap_major(value_w, dim, kernel_size);
        let gw_tap = depthwise_weight_to_tap_major(gate_w, dim, kernel_size);
        Ok(Self {
            norm: RmsNorm::from_slice(ctx, norm_w, DEFAULT_RMS_NORM_EPS)?,
            value_w_tap_major: upload(ctx.client(), &vw_tap),
            value_b: upload(ctx.client(), value_b),
            gate_w_tap_major: upload(ctx.client(), &gw_tap),
            gate_b: upload(ctx.client(), gate_b),
            gamma: upload(ctx.client(), gamma),
            dim,
            kernel_size,
        })
    }

    pub fn load<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, prefix: &str) -> CkResult<Self> {
        let (_, norm_w) = st.read_as::<F>(&format!("{prefix}.norm.weight"))?;
        let (v_shape, value_w) = st.read_as::<F>(&format!("{prefix}.value.weight"))?;
        let (_, value_b) = st.read_as::<F>(&format!("{prefix}.value.bias"))?;
        let (_, gate_w) = st.read_as::<F>(&format!("{prefix}.gate.weight"))?;
        let (_, gate_b) = st.read_as::<F>(&format!("{prefix}.gate.bias"))?;
        let (_, gamma) = st.read_as::<F>(&format!("{prefix}.gamma"))?;
        if v_shape.len() != 4 || v_shape[1] != 1 || v_shape[2] != v_shape[3] {
            return Err(CkError::Weights(format!(
                "unexpected DepthwiseSwiGlu2d shape {v_shape:?} at '{prefix}'"
            )));
        }
        Self::from_slices(
            ctx, v_shape[0], v_shape[2], &norm_w, &value_w, &value_b, &gate_w, &gate_b, &gamma,
        )
    }

    /// Computes `x + local_mix(x)` in a single norm + fused depthwise SwiGLU pass.
    pub fn forward_residual<F: CkFloat, const D: usize>(
        &self,
        ctx: &MlContext,
        x: Tensor<D>,
    ) -> CkResult<Tensor<D>> {
        let shape = x.dims();
        let (batches, h, w, c) = match D {
            4 => (shape[0], shape[1], shape[2], shape[3]),
            5 => (shape[0] * shape[1], shape[2], shape[3], shape[4]),
            _ => {
                return Err(CkError::Shape(format!(
                    "DepthwiseSwiGlu2d expects 4D or 5D tensor, got {shape:?}"
                )));
            }
        };
        if c != self.dim {
            return Err(CkError::Shape(format!(
                "DepthwiseSwiGlu2d expected dim {}, got {c}",
                self.dim
            )));
        }
        let total = batches * h * w * c;
        let normed = self.norm.forward::<F, D>(ctx, x.clone())?;
        let c_normed = into_contiguous_cube(normed)?;
        let c_res = into_contiguous_cube(x)?;
        let out = empty::<F>(ctx.client(), total);
        let pad = self.kernel_size / 2;
        let reflect = u32::from(h > pad && w > pad);
        let geom = LaunchGeometry::elementwise(ctx.client(), total);
        let taps = self.kernel_size * self.kernel_size;
        unsafe {
            depthwise_swiglu_2d_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(c_normed.handle, total),
                BufferArg::from_raw_parts(c_res.handle, total),
                BufferArg::from_raw_parts(self.value_w_tap_major.clone(), taps * c),
                BufferArg::from_raw_parts(self.value_b.clone(), c),
                BufferArg::from_raw_parts(self.gate_w_tap_major.clone(), taps * c),
                BufferArg::from_raw_parts(self.gate_b.clone(), c),
                BufferArg::from_raw_parts(self.gamma.clone(), c),
                BufferArg::from_raw_parts(out.clone(), total),
                total as u32,
                h as u32,
                w as u32,
                c as u32,
                self.kernel_size as u32,
                reflect,
            );
        }
        Ok(ctx.tensor_from_handle(out, shape, elem_dtype::<F>()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use corridorkey_core::ComputeTarget;

    fn host_boundary(pos: isize, tap: isize, pad: isize, len: isize, reflect: bool) -> usize {
        let raw = pos + tap - pad;
        let max_idx = len - 1;
        let mapped = if reflect {
            let low = if raw < 0 { -raw } else { raw };
            if low > max_idx {
                2 * max_idx - low
            } else {
                low
            }
        } else {
            raw.clamp(0, max_idx)
        };
        mapped as usize
    }

    fn host_dw_conv2d(
        x: &[f64],
        w: &[f64],
        b: &[f64],
        batches: usize,
        h: usize,
        width: usize,
        c: usize,
        k: usize,
    ) -> Vec<f64> {
        let pad = k / 2;
        let reflect = h > pad && width > pad;
        let taps = k * k;
        let mut out = vec![0.0; batches * h * width * c];
        for batch in 0..batches {
            for y in 0..h {
                for col in 0..width {
                    for ch in 0..c {
                        let mut acc = b[ch];
                        for ky in 0..k {
                            let ry = host_boundary(
                                y as isize,
                                ky as isize,
                                pad as isize,
                                h as isize,
                                reflect,
                            );
                            for kx in 0..k {
                                let rx = host_boundary(
                                    col as isize,
                                    kx as isize,
                                    pad as isize,
                                    width as isize,
                                    reflect,
                                );
                                let in_idx = ((batch * h + ry) * width + rx) * c + ch;
                                let w_idx = ch * taps + ky * k + kx;
                                acc += x[in_idx] * w[w_idx];
                            }
                        }
                        out[((batch * h + y) * width + col) * c + ch] = acc;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn qkv_depthwise_and_local_mix_match_f64_reference_on_reflect_and_replicate() {
        // Test both reflect (H=5, W=6 > pad=1 for k=3) and replicate (H=2, W=3 <= pad=3 for k=7)
        for (b, h, w, dim, k) in [(2, 5, 6, 4, 3), (1, 2, 3, 5, 7)] {
            let c3 = 3 * dim;
            let taps = k * k;
            let make_vec = |n: usize, s: f32| -> Vec<f32> {
                (0..n).map(|i| ((i as f32) * s + 0.2).cos() * 0.4).collect()
            };
            let to64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };

            let qkv_in = make_vec(b * h * w * c3, 0.13);
            let qkv_w = make_vec(c3 * taps, 0.09);
            let qkv_b = make_vec(c3, 0.17);
            let want_qkv =
                host_dw_conv2d(&to64(&qkv_in), &to64(&qkv_w), &to64(&qkv_b), b, h, w, c3, k);

            for target in ComputeTarget::available() {
                let ctx = MlContext::new(target).unwrap();
                let dw = QkvDepthwiseSpatial::from_slices(&ctx, dim, k, &qkv_w, &qkv_b).unwrap();
                let t_in = ctx.upload_tensor(&qkv_in, [b, h, w, c3]).unwrap();
                let t_out = dw.forward::<f32, 4>(&ctx, t_in).unwrap();
                let got = ctx.download_tensor::<f32, 4>(t_out).unwrap();
                for (i, (&g, &ref_v)) in got.iter().zip(&want_qkv).enumerate() {
                    assert!(
                        (g as f64 - ref_v).abs() < 2e-5,
                        "{target} k={k} [{i}]: {g} vs {ref_v}"
                    );
                }
            }

            // Now verify DepthwiseSwiGlu2d::forward_residual against f64 host reference
            let x_in = make_vec(b * h * w * dim, 0.11);
            let norm_w = make_vec(dim, 0.23);
            let vw = make_vec(dim * taps, 0.07);
            let vb = make_vec(dim, 0.19);
            let gw = make_vec(dim * taps, 0.05);
            let gb = make_vec(dim, 0.29);
            let gamma = make_vec(dim, 0.31);

            let x64 = to64(&x_in);
            let nw64 = to64(&norm_w);
            let mut normed64 = vec![0.0; b * h * w * dim];
            for row in 0..(b * h * w) {
                let slice = &x64[row * dim..(row + 1) * dim];
                let mean_sq = slice.iter().map(|v| v * v).sum::<f64>() / (dim as f64);
                let rrms = 1.0 / (mean_sq + (DEFAULT_RMS_NORM_EPS as f64)).sqrt();
                for ch in 0..dim {
                    normed64[row * dim + ch] = slice[ch] * rrms * nw64[ch];
                }
            }
            let val64 = host_dw_conv2d(&normed64, &to64(&vw), &to64(&vb), b, h, w, dim, k);
            let gate64 = host_dw_conv2d(&normed64, &to64(&gw), &to64(&gb), b, h, w, dim, k);
            let gam64 = to64(&gamma);
            let mut want_res = vec![0.0; b * h * w * dim];
            for idx in 0..(b * h * w * dim) {
                let ch = idx % dim;
                let g = gate64[idx];
                let silu_g = g / (1.0 + (-g).exp());
                want_res[idx] = x64[idx] + gam64[ch] * val64[idx] * silu_g;
            }

            for target in ComputeTarget::available() {
                let ctx = MlContext::new(target).unwrap();
                let mix = DepthwiseSwiGlu2d::from_slices(
                    &ctx, dim, k, &norm_w, &vw, &vb, &gw, &gb, &gamma,
                )
                .unwrap();
                let t_in = ctx.upload_tensor(&x_in, [b, h, w, dim]).unwrap();
                let t_out = mix.forward_residual::<f32, 4>(&ctx, t_in).unwrap();
                let got = ctx.download_tensor::<f32, 4>(t_out).unwrap();
                for (i, (&g, &ref_v)) in got.iter().zip(&want_res).enumerate() {
                    assert!(
                        (g as f64 - ref_v).abs() < 2e-5,
                        "{target} swiglu k={k} [{i}]: {g} vs {ref_v}"
                    );
                }
            }
        }
    }
}
