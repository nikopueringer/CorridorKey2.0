//! `AdaLNZero` conditioning (`enc_16`) and `FusionLinSwiglu` (`/64` VFM + `down_aux` fusion)
//! blocks (`corridorkey/blocks.py`).

use burn_tensor::Tensor;
use corridorkey_base::core::{
    CkFloat,
    buffer::{empty, upload},
};
use corridorkey_core::{CkError, CkResult, compute::LaunchGeometry};
use cubecl::{prelude::BufferArg, server::Handle};

use crate::{
    hub::SafetensorsFile,
    models::blocks::{
        kernels::{
            adaln_gate_residual_kernel, adaln_modulate_kernel, add_row_bias_kernel,
            silu_bias_kernel, swiglu_gate_kernel,
        },
        norm::{DEFAULT_RMS_NORM_EPS, RmsNorm},
    },
    runtime::{MlContext, elem_dtype, into_contiguous_cube},
};

/// Device-resident `AdaLNZero` conditioning module (`corridorkey/blocks.py`).
///
/// Maps `cond` (`[BT, cond_dim]`) through `in_proj -> SiLU -> proj` to produce
/// `pairs * 3 * dim` modulation parameters (`shift, scale, gate` per pair).
#[derive(Clone, Debug)]
pub struct AdaLnZero {
    in_proj_w_t: Tensor<2>,
    in_proj_b: Handle,
    proj_w_t: Tensor<2>,
    proj_b: Handle,
    cond_dim: usize,
    hidden: usize,
    dim: usize,
    pairs: usize,
}

impl AdaLnZero {
    #[allow(clippy::too_many_arguments)]
    pub fn from_slices<F: CkFloat>(
        ctx: &MlContext,
        dim: usize,
        cond_dim: usize,
        pairs: usize,
        hidden: usize,
        in_w: &[F],
        in_b: &[F],
        proj_w: &[F],
        proj_b: &[F],
    ) -> CkResult<Self> {
        let out_dim = pairs * 3 * dim;
        if in_w.len() != hidden * cond_dim
            || in_b.len() != hidden
            || proj_w.len() != out_dim * hidden
            || proj_b.len() != out_dim
        {
            return Err(CkError::Shape(
                "AdaLnZero weight/bias shape mismatch".into(),
            ));
        }
        Ok(Self {
            in_proj_w_t: ctx.upload_transposed(in_w, hidden, cond_dim)?,
            in_proj_b: upload(ctx.client(), in_b),
            proj_w_t: ctx.upload_transposed(proj_w, out_dim, hidden)?,
            proj_b: upload(ctx.client(), proj_b),
            cond_dim,
            hidden,
            dim,
            pairs,
        })
    }

    pub fn load<F: CkFloat>(
        ctx: &MlContext,
        st: &SafetensorsFile,
        prefix: &str,
        dim: usize,
        pairs: usize,
    ) -> CkResult<Self> {
        let (in_shape, in_w) = st.read_as::<F>(&format!("{prefix}.in_proj.weight"))?;
        let (_, in_b) = st.read_as::<F>(&format!("{prefix}.in_proj.bias"))?;
        let (p_shape, proj_w) = st.read_as::<F>(&format!("{prefix}.proj.weight"))?;
        let (_, proj_b) = st.read_as::<F>(&format!("{prefix}.proj.bias"))?;
        if in_shape.len() != 2 || p_shape.len() != 2 {
            return Err(CkError::Weights(format!(
                "unexpected AdaLnZero shapes in={in_shape:?} proj={p_shape:?} at '{prefix}'"
            )));
        }
        let hidden = in_shape[0];
        let cond_dim = in_shape[1];
        Self::from_slices(
            ctx, dim, cond_dim, pairs, hidden, &in_w, &in_b, &proj_w, &proj_b,
        )
    }

    /// Computes raw (pre-`proj_b` bias) modulation parameters `[BT, pairs * 3 * dim]` as a
    /// contiguous `Handle` on device (`proj_b` is fused into [`Self::modulate`] and
    /// [`Self::gate_residual`]).
    pub fn compute_params<F: CkFloat>(
        &self,
        ctx: &MlContext,
        cond_bt: Tensor<2>,
    ) -> CkResult<Handle> {
        let [bt, cd] = cond_bt.dims();
        if cd != self.cond_dim {
            return Err(CkError::Shape(format!(
                "AdaLnZero expected cond_dim={}, got {cd}",
                self.cond_dim
            )));
        }
        let h_pre = ctx.matmul(cond_bt, self.in_proj_w_t.clone());
        let c_h = into_contiguous_cube(h_pre)?;
        let total_h = bt * self.hidden;
        let act_h = empty::<F>(ctx.client(), total_h);
        let geom_h = LaunchGeometry::elementwise(ctx.client(), total_h);
        unsafe {
            silu_bias_kernel::launch::<F>(
                ctx.client(),
                geom_h.cube_count,
                geom_h.cube_dim,
                BufferArg::from_raw_parts(c_h.handle, total_h),
                BufferArg::from_raw_parts(self.in_proj_b.clone(), self.hidden),
                BufferArg::from_raw_parts(act_h.clone(), total_h),
                total_h as u32,
                self.hidden as u32,
            );
        }
        let act_2d = ctx.tensor_from_handle(act_h, [bt, self.hidden], elem_dtype::<F>());
        let params_2d = ctx.matmul(act_2d, self.proj_w_t.clone());
        Ok(into_contiguous_cube(params_2d)?.handle)
    }

    /// Modulates `x` (`[B, T, H, W, dim]` or `[BT, H, W, dim]`) using pair `pair_idx`:
    /// `x * (1 + scale) + shift`, into a new buffer: `x` may share its buffer with the caller's
    /// clones (Burn tensors are reference-counted), so writing it in place would change them.
    pub fn modulate<F: CkFloat, const D: usize>(
        &self,
        ctx: &MlContext,
        x: Tensor<D>,
        params: &Handle,
        bt: usize,
        hw: usize,
        pair_idx: usize,
    ) -> CkResult<Tensor<D>> {
        let shape = x.dims();
        let total = bt * hw * self.dim;
        let params_stride = self.pairs * 3 * self.dim;
        let pair_offset = pair_idx * 3 * self.dim;
        let c_x = into_contiguous_cube(x)?;
        let out = empty::<F>(ctx.client(), total);
        let geom = LaunchGeometry::elementwise(ctx.client(), total);
        unsafe {
            adaln_modulate_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(c_x.handle, total),
                BufferArg::from_raw_parts(params.clone(), bt * params_stride),
                BufferArg::from_raw_parts(self.proj_b.clone(), params_stride),
                BufferArg::from_raw_parts(out.clone(), total),
                total as u32,
                hw as u32,
                self.dim as u32,
                params_stride as u32,
                pair_offset as u32,
            );
        }
        Ok(ctx.tensor_from_handle(out, shape, elem_dtype::<F>()))
    }

    /// Computes `shortcut + branch * gate` using pair `pair_idx`.
    pub fn gate_residual<F: CkFloat, const D: usize>(
        &self,
        ctx: &MlContext,
        shortcut: Tensor<D>,
        branch: Tensor<D>,
        params: &Handle,
        bt: usize,
        hw: usize,
        pair_idx: usize,
    ) -> CkResult<Tensor<D>> {
        let shape = shortcut.dims();
        let total = bt * hw * self.dim;
        let params_stride = self.pairs * 3 * self.dim;
        let pair_offset = pair_idx * 3 * self.dim;
        let c_short = into_contiguous_cube(shortcut)?;
        let c_branch = into_contiguous_cube(branch)?;
        let out = empty::<F>(ctx.client(), total);
        let geom = LaunchGeometry::elementwise(ctx.client(), total);
        unsafe {
            adaln_gate_residual_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(c_short.handle, total),
                BufferArg::from_raw_parts(c_branch.handle, total),
                BufferArg::from_raw_parts(params.clone(), bt * params_stride),
                BufferArg::from_raw_parts(self.proj_b.clone(), params_stride),
                BufferArg::from_raw_parts(out.clone(), total),
                total as u32,
                hw as u32,
                self.dim as u32,
                params_stride as u32,
                pair_offset as u32,
            );
        }
        Ok(ctx.tensor_from_handle(out, shape, elem_dtype::<F>()))
    }
}

/// Device-resident `FusionLinSwiglu` (`/64` VFM + `down_aux` fusion, `corridorkey/blocks.py`).
///
/// Computes `linear_shortcut(x) + w_down(silu(w_gate(norm(x))) * w_up(norm(x)))`.
#[derive(Clone, Debug)]
pub struct FusionLinSwiglu {
    shortcut_w_t: Tensor<2>,
    shortcut_b: Handle,
    norm: RmsNorm,
    gate_w_t: Tensor<2>,
    gate_b: Handle,
    up_w_t: Tensor<2>,
    up_b: Handle,
    down_w_t: Tensor<2>,
    down_b: Handle,
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
}

impl FusionLinSwiglu {
    #[allow(clippy::too_many_arguments)]
    pub fn from_slices<F: CkFloat>(
        ctx: &MlContext,
        in_dim: usize,
        hidden: usize,
        out_dim: usize,
        shortcut_w: &[F],
        shortcut_b: &[F],
        norm_w: &[F],
        gate_w: &[F],
        gate_b: &[F],
        up_w: &[F],
        up_b: &[F],
        down_w: &[F],
        down_b: &[F],
    ) -> CkResult<Self> {
        if shortcut_w.len() != out_dim * in_dim
            || shortcut_b.len() != out_dim
            || norm_w.len() != in_dim
            || gate_w.len() != hidden * in_dim
            || gate_b.len() != hidden
            || up_w.len() != hidden * in_dim
            || up_b.len() != hidden
            || down_w.len() != out_dim * hidden
            || down_b.len() != out_dim
        {
            return Err(CkError::Shape(
                "FusionLinSwiglu weight/bias shape mismatch".into(),
            ));
        }
        Ok(Self {
            shortcut_w_t: ctx.upload_transposed(shortcut_w, out_dim, in_dim)?,
            shortcut_b: upload(ctx.client(), shortcut_b),
            norm: RmsNorm::from_slice(ctx, norm_w, DEFAULT_RMS_NORM_EPS)?,
            gate_w_t: ctx.upload_transposed(gate_w, hidden, in_dim)?,
            gate_b: upload(ctx.client(), gate_b),
            up_w_t: ctx.upload_transposed(up_w, hidden, in_dim)?,
            up_b: upload(ctx.client(), up_b),
            down_w_t: ctx.upload_transposed(down_w, out_dim, hidden)?,
            down_b: upload(ctx.client(), down_b),
            in_dim,
            hidden,
            out_dim,
        })
    }

    pub fn load<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, prefix: &str) -> CkResult<Self> {
        let (s_shape, shortcut_w) = st.read_as::<F>(&format!("{prefix}.linear_shortcut.weight"))?;
        let (_, shortcut_b) = st.read_as::<F>(&format!("{prefix}.linear_shortcut.bias"))?;
        let (_, norm_w) = st.read_as::<F>(&format!("{prefix}.norm.weight"))?;
        let (g_shape, gate_w) = st.read_as::<F>(&format!("{prefix}.w_gate.weight"))?;
        let (_, gate_b) = st.read_as::<F>(&format!("{prefix}.w_gate.bias"))?;
        let (_, up_w) = st.read_as::<F>(&format!("{prefix}.w_up.weight"))?;
        let (_, up_b) = st.read_as::<F>(&format!("{prefix}.w_up.bias"))?;
        let (_, down_w) = st.read_as::<F>(&format!("{prefix}.w_down.weight"))?;
        let (_, down_b) = st.read_as::<F>(&format!("{prefix}.w_down.bias"))?;
        if s_shape.len() != 2 || g_shape.len() != 2 {
            return Err(CkError::Weights(format!(
                "unexpected FusionLinSwiglu shapes s={s_shape:?} g={g_shape:?} at '{prefix}'"
            )));
        }
        let out_dim = s_shape[0];
        let in_dim = s_shape[1];
        let hidden = g_shape[0];
        Self::from_slices(
            ctx,
            in_dim,
            hidden,
            out_dim,
            &shortcut_w,
            &shortcut_b,
            &norm_w,
            &gate_w,
            &gate_b,
            &up_w,
            &up_b,
            &down_w,
            &down_b,
        )
    }

    pub const fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Runs `FusionLinSwiglu` on a rank-`D` channels-last tensor `[..., in_dim]`, returning
    /// `[..., out_dim]`.
    pub fn forward<F: CkFloat, const D: usize>(
        &self,
        ctx: &MlContext,
        x: Tensor<D>,
    ) -> CkResult<Tensor<D>> {
        let mut out_shape = x.dims();
        if out_shape[D - 1] != self.in_dim {
            return Err(CkError::Shape(format!(
                "FusionLinSwiglu expected in_dim={}, got {}",
                self.in_dim,
                out_shape[D - 1]
            )));
        }
        out_shape[D - 1] = self.out_dim;
        let tokens: usize = x.dims()[..D - 1].iter().product();

        let x_2d = x.clone().reshape([tokens, self.in_dim]);
        let shortcut_2d = ctx.matmul(x_2d, self.shortcut_w_t.clone());
        let c_short = into_contiguous_cube(shortcut_2d)?;
        let total_out = tokens * self.out_dim;
        let geom_out = LaunchGeometry::elementwise(ctx.client(), total_out);
        unsafe {
            add_row_bias_kernel::launch::<F>(
                ctx.client(),
                geom_out.cube_count.clone(),
                geom_out.cube_dim,
                BufferArg::from_raw_parts(c_short.handle.clone(), total_out),
                BufferArg::from_raw_parts(self.shortcut_b.clone(), self.out_dim),
                total_out as u32,
                self.out_dim as u32,
            );
        }

        let h_norm = self
            .norm
            .forward::<F, D>(ctx, x)?
            .reshape([tokens, self.in_dim]);
        let gate_2d = ctx.matmul(h_norm.clone(), self.gate_w_t.clone());
        let up_2d = ctx.matmul(h_norm, self.up_w_t.clone());
        let c_gate = into_contiguous_cube(gate_2d)?;
        let c_up = into_contiguous_cube(up_2d)?;

        let total_hidden = tokens * self.hidden;
        let gated_h = empty::<F>(ctx.client(), total_hidden);
        let geom_h = LaunchGeometry::elementwise(ctx.client(), total_hidden);
        unsafe {
            swiglu_gate_kernel::launch::<F>(
                ctx.client(),
                geom_h.cube_count,
                geom_h.cube_dim,
                BufferArg::from_raw_parts(c_gate.handle, total_hidden),
                BufferArg::from_raw_parts(c_up.handle, total_hidden),
                BufferArg::from_raw_parts(self.gate_b.clone(), self.hidden),
                BufferArg::from_raw_parts(self.up_b.clone(), self.hidden),
                BufferArg::from_raw_parts(gated_h.clone(), total_hidden),
                total_hidden as u32,
                self.hidden as u32,
            );
        }

        let gated_2d = ctx.tensor_from_handle(gated_h, [tokens, self.hidden], elem_dtype::<F>());
        let nonlin_2d = ctx.matmul(gated_2d, self.down_w_t.clone());
        let c_nonlin = into_contiguous_cube(nonlin_2d)?;
        unsafe {
            add_row_bias_kernel::launch::<F>(
                ctx.client(),
                geom_out.cube_count,
                geom_out.cube_dim,
                BufferArg::from_raw_parts(c_nonlin.handle.clone(), total_out),
                BufferArg::from_raw_parts(self.down_b.clone(), self.out_dim),
                total_out as u32,
                self.out_dim as u32,
            );
        }

        let short_t = ctx.tensor_from_handle(c_short.handle, out_shape, elem_dtype::<F>());
        let nonlin_t = ctx.tensor_from_handle(c_nonlin.handle, out_shape, elem_dtype::<F>());
        Ok(short_t + nonlin_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use corridorkey_core::ComputeTarget;

    #[test]
    fn adaln_zero_and_fusion_match_f64_reference() {
        let (bt, hw, dim, cond_dim, hidden) = (3, 5, 7, 1, 9);
        let make_vec = |n: usize, s: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32) * s + 0.1).sin() * 0.3).collect()
        };
        let cond = make_vec(bt * cond_dim, 0.21);
        let in_w = make_vec(hidden * cond_dim, 0.13);
        let in_b = make_vec(hidden, 0.07);
        let proj_w = make_vec(6 * dim * hidden, 0.05);
        let proj_b = make_vec(6 * dim, 0.11);
        let x = make_vec(bt * hw * dim, 0.17);
        let branch = make_vec(bt * hw * dim, 0.19);

        // Host f64 reference for AdaLnZero pair 0 modulate + gate_residual
        let mut params = vec![0.0f64; bt * 6 * dim];
        for b in 0..bt {
            let mut h = vec![0.0f64; hidden];
            for hi in 0..hidden {
                let v = (cond[b] as f64) * (in_w[hi] as f64) + (in_b[hi] as f64);
                h[hi] = v / (1.0 + (-v).exp());
            }
            for o in 0..6 * dim {
                let mut acc = proj_b[o] as f64;
                for hi in 0..hidden {
                    acc += h[hi] * (proj_w[o * hidden + hi] as f64);
                }
                params[b * 6 * dim + o] = acc;
            }
        }
        let mut want_mod = vec![0.0f64; bt * hw * dim];
        let mut want_res = vec![0.0f64; bt * hw * dim];
        for b in 0..bt {
            for p in 0..hw {
                for c in 0..dim {
                    let i = (b * hw + p) * dim + c;
                    let shift = params[b * 6 * dim + c];
                    let scale = params[b * 6 * dim + dim + c];
                    let gate = params[b * 6 * dim + 2 * dim + c];
                    want_mod[i] = (x[i] as f64) * (1.0 + scale) + shift;
                    want_res[i] = (x[i] as f64) + (branch[i] as f64) * gate;
                }
            }
        }

        for target in ComputeTarget::available() {
            let ctx = MlContext::new(target).unwrap();
            let adaln = AdaLnZero::from_slices(
                &ctx, dim, cond_dim, 2, hidden, &in_w, &in_b, &proj_w, &proj_b,
            )
            .unwrap();
            let t_cond = ctx.upload_tensor(&cond, [bt, cond_dim]).unwrap();
            let p_handle = adaln.compute_params::<f32>(&ctx, t_cond).unwrap();

            let t_x = ctx.upload_tensor(&x, [bt, hw, dim]).unwrap();
            let t_mod = adaln
                .modulate::<f32, 3>(&ctx, t_x.clone(), &p_handle, bt, hw, 0)
                .unwrap();
            let got_mod = ctx.download_tensor::<f32, 3>(t_mod).unwrap();

            let t_br = ctx.upload_tensor(&branch, [bt, hw, dim]).unwrap();
            let t_res = adaln
                .gate_residual::<f32, 3>(&ctx, t_x, t_br, &p_handle, bt, hw, 0)
                .unwrap();
            let got_res = ctx.download_tensor::<f32, 3>(t_res).unwrap();

            for (i, (&gm, &wm)) in got_mod.iter().zip(&want_mod).enumerate() {
                assert!(
                    (gm as f64 - wm).abs() < 1e-5,
                    "{target} mod[{i}]: {gm} vs {wm}"
                );
            }
            for (i, (&gr, &wr)) in got_res.iter().zip(&want_res).enumerate() {
                assert!(
                    (gr as f64 - wr).abs() < 1e-5,
                    "{target} res[{i}]: {gr} vs {wr}"
                );
            }
        }
    }
}
