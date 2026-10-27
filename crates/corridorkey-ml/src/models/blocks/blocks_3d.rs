//! 3D spatiotemporal transformer blocks used in Stage 1 (`corridorkey/blocks.py`,
//! `corridorkey/stage1.py`): [`FullSpatialBlock3d`], [`FullSpatioTemporalBlock`],
//! [`CSwinBlock3d`], [`SwinBlock3d`], and [`Stage1DecBlock3d`].
//!
//! All blocks operate on channels-last `[B, T, H, W, C]` (`Tensor<5>`) directly on the device.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use burn_tensor::Tensor;
use corridorkey_base::core::{CkFloat, buffer::empty};
use corridorkey_core::{CkResult, compute::LaunchGeometry};
use cubecl::{prelude::BufferArg, server::Handle};

use crate::{
    hub::SafetensorsFile,
    models::{
        blocks::{
            attn_core::{
                AttentionCore, ShiftedRegionGroups, pair_units, run_unmasked_sdpa,
                run_unmasked_sdpa_handles,
            },
            blocks_2d::FullSpatialAttention2d,
            fusion::AdaLnZero,
            kernels::{
                add_row_bias_kernel, merge_cswin3d_gate_kernel,
                pad_roll_2d_kernel, prep_cswin3d_qkv_kernel, window_partition_3d_kernel,
                window_unpartition_3d_kernel,
            },
            local_mix::DepthwiseSwiGlu2d,
            mlp::SwiGlu,
            norm::{DEFAULT_RMS_NORM_EPS, RmsNorm},
        },
        rope::{Rope2d, Rope3d},
    },
    runtime::{MlContext, elem_dtype, into_contiguous_cube},
};

fn load_optional_local_mix<F: CkFloat>(
    ctx: &MlContext,
    st: &SafetensorsFile,
    prefix: &str,
) -> CkResult<Option<DepthwiseSwiGlu2d>> {
    let lm_prefix = format!("{prefix}.local_mix");
    if st.entry(&format!("{lm_prefix}.norm.weight")).is_ok() {
        Ok(Some(DepthwiseSwiGlu2d::load::<F>(ctx, st, &lm_prefix)?))
    } else {
        Ok(None)
    }
}

fn load_optional_adaln<F: CkFloat>(
    ctx: &MlContext,
    st: &SafetensorsFile,
    prefix: &str,
    dim: usize,
) -> CkResult<Option<AdaLnZero>> {
    let adaln_prefix = format!("{prefix}.adaln");
    if st.entry(&format!("{adaln_prefix}.in_proj.weight")).is_ok() {
        Ok(Some(AdaLnZero::load::<F>(ctx, st, &adaln_prefix, dim, 2)?))
    } else {
        Ok(None)
    }
}

/// Device-resident `FullSpatialBlock3D` (`corridorkey/blocks.py`).
///
/// Runs per-frame full-spatial attention on `[B, T, H, W, C]` with optional `AdaLNZero`
/// conditioning (`enc_16` in `Stage1`).
#[derive(Clone, Debug)]
pub struct FullSpatialBlock3d {
    pub norm1: RmsNorm,
    pub attn: FullSpatialAttention2d,
    pub local_mix: Option<DepthwiseSwiGlu2d>,
    pub adaln: Option<AdaLnZero>,
    pub norm2: RmsNorm,
    pub mlp: SwiGlu,
}

impl FullSpatialBlock3d {
    pub fn load<F: CkFloat>(
        ctx: &MlContext,
        st: &SafetensorsFile,
        prefix: &str,
        num_heads: usize,
        rope: Rope2d,
    ) -> CkResult<Self> {
        let (_, norm1_w) = st.read_as::<F>(&format!("{prefix}.norm1.weight"))?;
        let (_, norm2_w) = st.read_as::<F>(&format!("{prefix}.norm2.weight"))?;
        let core = AttentionCore::load::<F>(ctx, st, &format!("{prefix}.attn"), num_heads)?;
        let dim = core.dim;
        let local_mix = load_optional_local_mix::<F>(ctx, st, prefix)?;
        let adaln = load_optional_adaln::<F>(ctx, st, prefix, dim)?;
        let mlp = SwiGlu::load::<F>(ctx, st, &format!("{prefix}.mlp"))?;
        Ok(Self {
            norm1: RmsNorm::from_slice(ctx, &norm1_w, DEFAULT_RMS_NORM_EPS)?,
            attn: FullSpatialAttention2d { core, rope },
            local_mix,
            adaln,
            norm2: RmsNorm::from_slice(ctx, &norm2_w, DEFAULT_RMS_NORM_EPS)?,
            mlp,
        })
    }

    pub fn forward<F: CkFloat>(
        &self,
        ctx: &MlContext,
        x: Tensor<5>,
        cond_bt: Option<Tensor<2>>,
    ) -> CkResult<Tensor<5>> {
        let [b, t, h, w, c] = x.dims();
        let bt = b * t;
        let hw = h * w;

        let adaln_params = match (&self.adaln, cond_bt) {
            (Some(adaln), Some(cond)) => Some((adaln, adaln.compute_params::<F>(ctx, cond)?)),
            (Some(adaln), None) => {
                let zeros = vec![corridorkey_base::core::cast::<F>(0.0); bt];
                let cond = ctx.upload_tensor(&zeros, [bt, 1])?;
                Some((adaln, adaln.compute_params::<F>(ctx, cond)?))
            }
            (None, _) => None,
        };

        let mut u1 = self.norm1.forward::<F, 5>(ctx, x.clone())?;
        if let Some((adaln, params)) = &adaln_params {
            u1 = adaln.modulate::<F, 5>(ctx, u1, params, bt, hw, 0)?;
        }
        let attn_4d = self.attn.forward::<F>(ctx, u1.reshape([bt, h, w, c]))?;
        let attn_5d = attn_4d.reshape([b, t, h, w, c]);

        let mut res = if let Some((adaln, params)) = &adaln_params {
            adaln.gate_residual::<F, 5>(ctx, x, attn_5d, params, bt, hw, 0)?
        } else {
            x + attn_5d
        };

        if let Some(lm) = &self.local_mix {
            res = lm
                .forward_residual::<F, 4>(ctx, res.reshape([bt, h, w, c]))?
                .reshape([b, t, h, w, c]);
        }

        let mut u2 = self.norm2.forward::<F, 5>(ctx, res.clone())?;
        if let Some((adaln, params)) = &adaln_params {
            u2 = adaln.modulate::<F, 5>(ctx, u2, params, bt, hw, 1)?;
        }
        if let Some((adaln, params)) = &adaln_params {
            let mlp_out = self.mlp.forward::<F, 5>(ctx, u2)?;
            adaln.gate_residual::<F, 5>(ctx, res, mlp_out, params, bt, hw, 1)
        } else {
            self.mlp.forward_residual::<F, 5>(ctx, u2, res)
        }
    }
}

/// Device-resident `FullSpatioTemporalBlock` (`/64` global body in `Stage1`).
#[derive(Clone, Debug)]
pub struct FullSpatioTemporalBlock {
    pub norm1: RmsNorm,
    pub core: AttentionCore,
    pub rope: Rope3d,
    pub local_mix: Option<DepthwiseSwiGlu2d>,
    pub norm2: RmsNorm,
    pub mlp: SwiGlu,
}

impl FullSpatioTemporalBlock {
    pub fn load<F: CkFloat>(
        ctx: &MlContext,
        st: &SafetensorsFile,
        prefix: &str,
        num_heads: usize,
        rope: Rope3d,
    ) -> CkResult<Self> {
        let (_, norm1_w) = st.read_as::<F>(&format!("{prefix}.norm1.weight"))?;
        let (_, norm2_w) = st.read_as::<F>(&format!("{prefix}.norm2.weight"))?;
        let core = AttentionCore::load::<F>(ctx, st, &format!("{prefix}.attn"), num_heads)?;
        let local_mix = load_optional_local_mix::<F>(ctx, st, prefix)?;
        let mlp = SwiGlu::load::<F>(ctx, st, &format!("{prefix}.mlp"))?;
        Ok(Self {
            norm1: RmsNorm::from_slice(ctx, &norm1_w, DEFAULT_RMS_NORM_EPS)?,
            core,
            rope,
            local_mix,
            norm2: RmsNorm::from_slice(ctx, &norm2_w, DEFAULT_RMS_NORM_EPS)?,
            mlp,
        })
    }

    pub fn forward<F: CkFloat>(
        &self,
        ctx: &MlContext,
        x: Tensor<5>,
        t_stride: f64,
    ) -> CkResult<Tensor<5>> {
        let [b, t, h, w, c] = x.dims();
        let bt = b * t;
        let n_tokens = t * h * w;

        let u1 = self.norm1.forward::<F, 5>(ctx, x.clone())?;
        let (qkv_4d, bias_applied) = self
            .core
            .project_qkv_spatial::<F>(ctx, u1.clone().reshape([bt, h, w, c]))?;
        let c_qkv = into_contiguous_cube(qkv_4d)?;

        let (cos_h, sin_h) = self.rope.cos_sin_handles::<F>(ctx, t, h, w, t_stride);
        let (q, k, v) = self.core.prep_qkv_norm_rope::<F>(
            ctx,
            c_qkv.handle,
            &cos_h,
            &sin_h,
            b,
            n_tokens,
            bias_applied,
        );
        let attn = run_unmasked_sdpa::<F>(ctx, q, k, v)?;
        let gate_tokens_2d = u1.reshape([b * n_tokens, c]);
        let proj_h = self
            .core
            .gate_and_project_out::<F>(ctx, gate_tokens_2d, attn, b, n_tokens)?;
        let attn_5d = ctx.tensor_from_handle(proj_h, [b, t, h, w, c], elem_dtype::<F>());

        let mut res = x + attn_5d;
        if let Some(lm) = &self.local_mix {
            res = lm
                .forward_residual::<F, 4>(ctx, res.reshape([bt, h, w, c]))?
                .reshape([b, t, h, w, c]);
        }
        let u2 = self.norm2.forward::<F, 5>(ctx, res.clone())?;
        self.mlp.forward_residual::<F, 5>(ctx, u2, res)
    }
}

/// Device-resident `CSwinBlock3D` (`corridorkey/blocks.py`).
#[derive(Clone, Debug)]
pub struct CSwinBlock3d {
    pub norm1: RmsNorm,
    pub core: AttentionCore,
    pub rope: Rope3d,
    pub local_mix: Option<DepthwiseSwiGlu2d>,
    pub norm2: RmsNorm,
    pub mlp: SwiGlu,
}

impl CSwinBlock3d {
    pub fn load<F: CkFloat>(
        ctx: &MlContext,
        st: &SafetensorsFile,
        prefix: &str,
        num_heads: usize,
        rope: Rope3d,
    ) -> CkResult<Self> {
        let (_, norm1_w) = st.read_as::<F>(&format!("{prefix}.norm1.weight"))?;
        let (_, norm2_w) = st.read_as::<F>(&format!("{prefix}.norm2.weight"))?;
        let core = AttentionCore::load::<F>(ctx, st, &format!("{prefix}.attn"), num_heads)?;
        let local_mix = load_optional_local_mix::<F>(ctx, st, prefix)?;
        let mlp = SwiGlu::load::<F>(ctx, st, &format!("{prefix}.mlp"))?;
        Ok(Self {
            norm1: RmsNorm::from_slice(ctx, &norm1_w, DEFAULT_RMS_NORM_EPS)?,
            core,
            rope,
            local_mix,
            norm2: RmsNorm::from_slice(ctx, &norm2_w, DEFAULT_RMS_NORM_EPS)?,
            mlp,
        })
    }

    pub fn forward<F: CkFloat>(
        &self,
        ctx: &MlContext,
        x: Tensor<5>,
        t_stride: f64,
    ) -> CkResult<Tensor<5>> {
        let [b, t, h, w, c] = x.dims();
        let bt = b * t;
        let n_tokens = t * h * w;
        let num_heads = self.core.num_heads;
        let head_dim = self.core.head_dim;
        let h_heads = num_heads / 2;
        let w_heads = num_heads - h_heads;

        let u1 = self.norm1.forward::<F, 5>(ctx, x.clone())?;
        let (qkv_4d, bias_applied) = self
            .core
            .project_qkv_spatial::<F>(ctx, u1.clone().reshape([bt, h, w, c]))?;
        let c_qkv = into_contiguous_cube(qkv_4d)?;

        let (cos_h, sin_h) = self.rope.cos_sin_handles::<F>(ctx, t, h, 1, t_stride);
        let (cos_w, sin_w) = self.rope.cos_sin_handles::<F>(ctx, t, 1, w, t_stride);

        let total_bthwh = b * n_tokens * num_heads;
        let elems_h = b * w * h_heads * (t * h) * head_dim;
        let elems_w = b * h * w_heads * (t * w) * head_dim;
        let qh = empty::<F>(ctx.client(), elems_h);
        let kh = empty::<F>(ctx.client(), elems_h);
        let vh = empty::<F>(ctx.client(), elems_h);
        let qw = empty::<F>(ctx.client(), elems_w);
        let kw = empty::<F>(ctx.client(), elems_w);
        let vw = empty::<F>(ctx.client(), elems_w);

        let geom_prep = LaunchGeometry::per_row(ctx.client(), total_bthwh, pair_units(head_dim));
        let prep_block = geom_prep.cube_dim.x;
        assert!(prep_block as usize * 2 >= head_dim, "prep cube of {prep_block} units < head_dim / 2");
        let c3 = 3 * c;
        let half_d = head_dim / 2;
        unsafe {
            prep_cswin3d_qkv_kernel::launch::<F>(
                ctx.client(),
                geom_prep.cube_count,
                geom_prep.cube_dim,
                BufferArg::from_raw_parts(c_qkv.handle, b * n_tokens * c3),
                BufferArg::from_raw_parts(self.core.qkv_b.clone(), c3),
                BufferArg::from_raw_parts(self.core.q_norm_w.clone(), head_dim),
                BufferArg::from_raw_parts(self.core.k_norm_w.clone(), head_dim),
                BufferArg::from_raw_parts(cos_h, (t * h) * half_d),
                BufferArg::from_raw_parts(sin_h, (t * h) * half_d),
                BufferArg::from_raw_parts(cos_w, (t * w) * half_d),
                BufferArg::from_raw_parts(sin_w, (t * w) * half_d),
                BufferArg::from_raw_parts(qh.clone(), elems_h),
                BufferArg::from_raw_parts(kh.clone(), elems_h),
                BufferArg::from_raw_parts(vh.clone(), elems_h),
                BufferArg::from_raw_parts(qw.clone(), elems_w),
                BufferArg::from_raw_parts(kw.clone(), elems_w),
                BufferArg::from_raw_parts(vw.clone(), elems_w),
                total_bthwh as u32,
                t as u32,
                h as u32,
                w as u32,
                num_heads as u32,
                head_dim as u32,
                u32::from(!bias_applied),
                DEFAULT_RMS_NORM_EPS as f32,
                prep_block,
            );
        }

        let dt = elem_dtype::<F>();
        let ah_handle = run_unmasked_sdpa_handles::<F>(ctx, qh, kh, vh,
            b * w,
            h_heads,
            t * h,
            head_dim,
        )?;
        let aw_handle = run_unmasked_sdpa_handles::<F>(ctx, qw, kw, vw,
            b * h,
            w_heads,
            t * w,
            head_dim,
        )?;

        let gate_logits = ctx.matmul(u1.reshape([b * n_tokens, c]), self.core.gate_w_t.clone());
        let c_gate = into_contiguous_cube(gate_logits)?;

        let total_elems = b * n_tokens * c;
        let merged = empty::<F>(ctx.client(), total_elems);
        let geom_merge = LaunchGeometry::elementwise(ctx.client(), total_elems);
        unsafe {
            merge_cswin3d_gate_kernel::launch::<F>(
                ctx.client(),
                geom_merge.cube_count.clone(),
                geom_merge.cube_dim,
                BufferArg::from_raw_parts(ah_handle, elems_h),
                BufferArg::from_raw_parts(aw_handle, elems_w),
                BufferArg::from_raw_parts(c_gate.handle, b * n_tokens * num_heads),
                BufferArg::from_raw_parts(merged.clone(), total_elems),
                total_elems as u32,
                t as u32,
                h as u32,
                w as u32,
                num_heads as u32,
                head_dim as u32,
            );
        }

        let merged_2d = ctx.tensor_from_handle(merged, [b * n_tokens, c], dt);
        let proj_2d = ctx.matmul(merged_2d, self.core.proj_w_t.clone());
        let c_proj = into_contiguous_cube(proj_2d)?;
        unsafe {
            add_row_bias_kernel::launch::<F>(
                ctx.client(),
                geom_merge.cube_count,
                geom_merge.cube_dim,
                BufferArg::from_raw_parts(c_proj.handle.clone(), total_elems),
                BufferArg::from_raw_parts(self.core.proj_b.clone(), c),
                total_elems as u32,
                c as u32,
            );
        }

        let mut res = x + ctx.tensor_from_handle(c_proj.handle, [b, t, h, w, c], dt);
        if let Some(lm) = &self.local_mix {
            res = lm
                .forward_residual::<F, 4>(ctx, res.reshape([bt, h, w, c]))?
                .reshape([b, t, h, w, c]);
        }
        let u2 = self.norm2.forward::<F, 5>(ctx, res.clone())?;
        self.mlp.forward_residual::<F, 5>(ctx, u2, res)
    }
}

/// Device-resident `SwinBlock3D` (`corridorkey/blocks.py`).
#[derive(Clone, Debug)]
pub struct SwinBlock3d {
    pub norm1: RmsNorm,
    pub core: AttentionCore,
    pub rope: Rope3d,
    pub window_size: usize,
    pub shift: (usize, usize),
    pub local_mix: Option<DepthwiseSwiGlu2d>,
    pub adaln: Option<AdaLnZero>,
    pub norm2: RmsNorm,
    pub mlp: SwiGlu,
    region_cache: Arc<Mutex<BTreeMap<(usize, usize, usize), ShiftedRegionGroups>>>,
}

impl SwinBlock3d {
    pub fn load<F: CkFloat>(
        ctx: &MlContext,
        st: &SafetensorsFile,
        prefix: &str,
        num_heads: usize,
        window_size: usize,
        shift: (usize, usize),
        rope: Rope3d,
    ) -> CkResult<Self> {
        let (_, norm1_w) = st.read_as::<F>(&format!("{prefix}.norm1.weight"))?;
        let (_, norm2_w) = st.read_as::<F>(&format!("{prefix}.norm2.weight"))?;
        let core = AttentionCore::load::<F>(ctx, st, &format!("{prefix}.attn"), num_heads)?;
        let dim = core.dim;
        let local_mix = load_optional_local_mix::<F>(ctx, st, prefix)?;
        let adaln = load_optional_adaln::<F>(ctx, st, prefix, dim)?;
        let mlp = SwiGlu::load::<F>(ctx, st, &format!("{prefix}.mlp"))?;
        Ok(Self {
            norm1: RmsNorm::from_slice(ctx, &norm1_w, DEFAULT_RMS_NORM_EPS)?,
            core,
            rope,
            window_size,
            shift,
            local_mix,
            adaln,
            norm2: RmsNorm::from_slice(ctx, &norm2_w, DEFAULT_RMS_NORM_EPS)?,
            mlp,
            region_cache: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    fn region_groups(
        &self,
        t: usize,
        h: usize,
        w: usize,
        hp: usize,
        wp: usize,
    ) -> ShiftedRegionGroups {
        let key = (t, h, w);
        let mut guard = self.region_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(groups) = guard.get(&key) {
            return groups.clone();
        }
        let (sh, sw) = self.shift;
        let groups = ShiftedRegionGroups::build_3d(t, h, w, hp, wp, self.window_size, sh, sw);
        guard.insert(key, groups.clone());
        groups
    }

    #[allow(clippy::too_many_arguments)]
    fn partition_3d<F: CkFloat>(
        ctx: &MlContext,
        src: &Handle,
        b: usize,
        t: usize,
        h: usize,
        w: usize,
        hp: usize,
        wp: usize,
        win: usize,
        ch: usize,
    ) -> Handle {
        let total = b * t * hp * wp * ch;
        let out = empty::<F>(ctx.client(), total);
        let geom = LaunchGeometry::elementwise(ctx.client(), total);
        unsafe {
            window_partition_3d_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(src.clone(), b * t * h * w * ch),
                BufferArg::from_raw_parts(out.clone(), total),
                total as u32,
                t as u32,
                h as u32,
                w as u32,
                hp as u32,
                wp as u32,
                win as u32,
                ch as u32,
            );
        }
        out
    }

    pub fn forward<F: CkFloat>(
        &self,
        ctx: &MlContext,
        x: Tensor<5>,
        t_stride: f64,
        cond_bt: Option<Tensor<2>>,
    ) -> CkResult<Tensor<5>> {
        let [b, t, h, w, c] = x.dims();
        let bt = b * t;
        let hw = h * w;
        let win = self.window_size;
        let (sh, sw) = self.shift;
        let hp = h.div_ceil(win) * win;
        let wp = w.div_ceil(win) * win;
        let has_shift = sh != 0 || sw != 0;
        let has_pad = hp != h || wp != w;
        let dt = elem_dtype::<F>();

        let adaln_params = match (&self.adaln, cond_bt) {
            (Some(adaln), Some(cond)) => Some((adaln, adaln.compute_params::<F>(ctx, cond)?)),
            (Some(adaln), None) => {
                let zeros = vec![corridorkey_base::core::cast::<F>(0.0); bt];
                let cond = ctx.upload_tensor(&zeros, [bt, 1])?;
                Some((adaln, adaln.compute_params::<F>(ctx, cond)?))
            }
            (None, _) => None,
        };

        let mut normed1 = self.norm1.forward::<F, 5>(ctx, x.clone())?;
        if let Some((adaln, params)) = &adaln_params {
            normed1 = adaln.modulate::<F, 5>(ctx, normed1, params, bt, hw, 0)?;
        }

        let (qkv_src_handle, gate_src_handle, src_h, src_w, bias_applied) = if has_shift {
            let c_norm1 = into_contiguous_cube(normed1)?;
            let total_p = bt * hp * wp * c;
            let rolled = empty::<F>(ctx.client(), total_p);
            let geom = LaunchGeometry::elementwise(ctx.client(), total_p);
            unsafe {
                pad_roll_2d_kernel::launch::<F>(
                    ctx.client(),
                    geom.cube_count,
                    geom.cube_dim,
                    BufferArg::from_raw_parts(c_norm1.handle, bt * h * w * c),
                    BufferArg::from_raw_parts(rolled.clone(), total_p),
                    total_p as u32,
                    h as u32,
                    w as u32,
                    hp as u32,
                    wp as u32,
                    c as u32,
                    sh as u32,
                    sw as u32,
                );
            }
            let rolled_4d = ctx.tensor_from_handle(rolled.clone(), [bt, hp, wp, c], dt);
            let (qkv_4d, bias_done) = self.core.project_qkv_spatial::<F>(ctx, rolled_4d)?;
            let c_qkv = into_contiguous_cube(qkv_4d)?;
            (c_qkv.handle, rolled, hp, wp, bias_done)
        } else {
            let (qkv_4d, bias_done) = self
                .core
                .project_qkv_spatial::<F>(ctx, normed1.clone().reshape([bt, h, w, c]))?;
            let c_qkv = into_contiguous_cube(qkv_4d)?;
            let c_norm1 = into_contiguous_cube(normed1)?;
            (c_qkv.handle, c_norm1.handle, h, w, bias_done)
        };

        let c3 = 3 * c;
        let qkv_win =
            Self::partition_3d::<F>(ctx, &qkv_src_handle, b, t, src_h, src_w, hp, wp, win, c3);
        let gate_win =
            Self::partition_3d::<F>(ctx, &gate_src_handle, b, t, src_h, src_w, hp, wp, win, c);

        let nw_total = b * (hp / win) * (wp / win);
        let n_win_tokens = t * win * win;
        let (cos_h, sin_h) = self.rope.cos_sin_handles::<F>(ctx, t, win, win, t_stride);

        let (q, k, v) = self.core.prep_qkv_norm_rope::<F>(
            ctx,
            qkv_win,
            &cos_h,
            &sin_h,
            nw_total,
            n_win_tokens,
            bias_applied,
        );

        let attn = if has_shift || has_pad {
            let groups = self.region_groups(t, h, w, hp, wp);
            groups.run_sdpa::<F>(ctx, q, k, v)?
        } else {
            run_unmasked_sdpa::<F>(ctx, q, k, v)?
        };

        let gate_tokens_2d = ctx.tensor_from_handle(gate_win, [nw_total * n_win_tokens, c], dt);
        let proj_win_handle = self.core.gate_and_project_out::<F>(
            ctx,
            gate_tokens_2d,
            attn,
            nw_total,
            n_win_tokens,
        )?;

        let c_shortcut = into_contiguous_cube(x.clone())?;
        let total_out = bt * h * w * c;
        let out_handle = empty::<F>(ctx.client(), total_out);
        let geom_out = LaunchGeometry::elementwise(ctx.client(), total_out);
        let add_shortcut = u32::from(adaln_params.is_none());
        unsafe {
            window_unpartition_3d_kernel::launch::<F>(
                ctx.client(),
                geom_out.cube_count,
                geom_out.cube_dim,
                BufferArg::from_raw_parts(proj_win_handle, bt * hp * wp * c),
                BufferArg::from_raw_parts(c_shortcut.handle, total_out),
                BufferArg::from_raw_parts(out_handle.clone(), total_out),
                total_out as u32,
                t as u32,
                h as u32,
                w as u32,
                hp as u32,
                wp as u32,
                win as u32,
                c as u32,
                sh as u32,
                sw as u32,
                add_shortcut,
            );
        }

        let unpart_5d = ctx.tensor_from_handle(out_handle, [b, t, h, w, c], dt);
        let mut res = if let Some((adaln, params)) = &adaln_params {
            adaln.gate_residual::<F, 5>(ctx, x, unpart_5d, params, bt, hw, 0)?
        } else {
            unpart_5d
        };

        if let Some(lm) = &self.local_mix {
            res = lm
                .forward_residual::<F, 4>(ctx, res.reshape([bt, h, w, c]))?
                .reshape([b, t, h, w, c]);
        }

        let mut u2 = self.norm2.forward::<F, 5>(ctx, res.clone())?;
        if let Some((adaln, params)) = &adaln_params {
            u2 = adaln.modulate::<F, 5>(ctx, u2, params, bt, hw, 1)?;
        }
        if let Some((adaln, params)) = &adaln_params {
            let mlp_out = self.mlp.forward::<F, 5>(ctx, u2)?;
            adaln.gate_residual::<F, 5>(ctx, res, mlp_out, params, bt, hw, 1)
        } else {
            self.mlp.forward_residual::<F, 5>(ctx, u2, res)
        }
    }
}

/// Polymorphic 3D decoder block in Stage 1's `dec_16`.
#[derive(Clone, Debug)]
pub enum Stage1DecBlock3d {
    Swin(SwinBlock3d),
    CSwin(CSwinBlock3d),
}

impl Stage1DecBlock3d {
    pub fn forward<F: CkFloat>(
        &self,
        ctx: &MlContext,
        x: Tensor<5>,
        t_stride: f64,
    ) -> CkResult<Tensor<5>> {
        match self {
            Self::Swin(b) => b.forward::<F>(ctx, x, t_stride, None),
            Self::CSwin(b) => b.forward::<F>(ctx, x, t_stride),
        }
    }
}
