//! 2D spatial transformer blocks used in Stage 2 (`corridorkey/blocks.py`, `corridorkey/stage2.py`):
//! [`FullSpatialBlock2d`], [`CSwinBlock2d`], [`SwinBlock2d`], and [`Stage2Block2d`].

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
            kernels::{
                add_row_bias_kernel, merge_cswin2d_gate_kernel,
                pad_roll_2d_kernel, prep_cswin2d_qkv_kernel, window_partition_2d_kernel,
                window_unpartition_unroll_residual_2d_kernel,
            },
            local_mix::DepthwiseSwiGlu2d,
            mlp::SwiGlu,
            norm::{DEFAULT_RMS_NORM_EPS, RmsNorm},
        },
        rope::Rope2d,
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

/// Device-resident `FullSpatialAttention2D` (`corridorkey/blocks.py`).
#[derive(Clone, Debug)]
pub struct FullSpatialAttention2d {
    pub core: AttentionCore,
    pub rope: Rope2d,
}

impl FullSpatialAttention2d {
    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x_nhwc: Tensor<4>) -> CkResult<Tensor<4>> {
        let [bt, h, w, c] = x_nhwc.dims();
        let n_tokens = h * w;
        let (qkv_4d, bias_applied) = self.core.project_qkv_spatial::<F>(ctx, x_nhwc.clone())?;
        let c_qkv = into_contiguous_cube(qkv_4d)?;

        let (cos_h, sin_h) = self.rope.cos_sin_handles::<F>(ctx, h, w);

        let (q, k, v) = self.core.prep_qkv_norm_rope::<F>(
            ctx,
            c_qkv.handle,
            &cos_h,
            &sin_h,
            bt,
            n_tokens,
            bias_applied,
        );
        let attn = run_unmasked_sdpa::<F>(ctx, q, k, v)?;
        let gate_tokens_2d = x_nhwc.reshape([bt * n_tokens, c]);
        let out_handle =
            self.core
                .gate_and_project_out::<F>(ctx, gate_tokens_2d, attn, bt, n_tokens)?;
        Ok(ctx.tensor_from_handle(out_handle, [bt, h, w, c], elem_dtype::<F>()))
    }
}

/// Device-resident `FullSpatialBlock2D` (`corridorkey/blocks.py`).
#[derive(Clone, Debug)]
pub struct FullSpatialBlock2d {
    pub norm1: RmsNorm,
    pub attn: FullSpatialAttention2d,
    pub local_mix: Option<DepthwiseSwiGlu2d>,
    pub norm2: RmsNorm,
    pub mlp: SwiGlu,
}

impl FullSpatialBlock2d {
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
        let local_mix = load_optional_local_mix::<F>(ctx, st, prefix)?;
        let mlp = SwiGlu::load::<F>(ctx, st, &format!("{prefix}.mlp"))?;
        Ok(Self {
            norm1: RmsNorm::from_slice(ctx, &norm1_w, DEFAULT_RMS_NORM_EPS)?,
            attn: FullSpatialAttention2d { core, rope },
            local_mix,
            norm2: RmsNorm::from_slice(ctx, &norm2_w, DEFAULT_RMS_NORM_EPS)?,
            mlp,
        })
    }

    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x: Tensor<4>) -> CkResult<Tensor<4>> {
        let normed1 = self.norm1.forward::<F, 4>(ctx, x.clone())?;
        let attn_out = self.attn.forward::<F>(ctx, normed1)?;
        let mut h = x + attn_out;
        if let Some(lm) = &self.local_mix {
            h = lm.forward_residual::<F, 4>(ctx, h)?;
        }
        let normed2 = self.norm2.forward::<F, 4>(ctx, h.clone())?;
        self.mlp.forward_residual::<F, 4>(ctx, normed2, h)
    }
}

/// Device-resident `CSwinAttention2D` (`corridorkey/blocks.py`).
#[derive(Clone, Debug)]
pub struct CSwinAttention2d {
    pub core: AttentionCore,
    pub rope: Rope2d,
}

impl CSwinAttention2d {
    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x_nhwc: Tensor<4>) -> CkResult<Tensor<4>> {
        let [b, h, w, c] = x_nhwc.dims();
        let n_tokens = h * w;
        let num_heads = self.core.num_heads;
        let head_dim = self.core.head_dim;
        let h_heads = num_heads / 2;
        let w_heads = num_heads - h_heads;

        let (qkv_4d, bias_applied) = self.core.project_qkv_spatial::<F>(ctx, x_nhwc.clone())?;
        let c_qkv = into_contiguous_cube(qkv_4d)?;

        let (cos_h, sin_h) = self.rope.cos_sin_handles::<F>(ctx, h, 1);
        let (cos_w, sin_w) = self.rope.cos_sin_handles::<F>(ctx, 1, w);

        let total_bhn = b * num_heads * n_tokens;
        let elems_h = b * w * h_heads * h * head_dim;
        let elems_w = b * h * w_heads * w * head_dim;
        let qh = empty::<F>(ctx.client(), elems_h);
        let kh = empty::<F>(ctx.client(), elems_h);
        let vh = empty::<F>(ctx.client(), elems_h);
        let qw = empty::<F>(ctx.client(), elems_w);
        let kw = empty::<F>(ctx.client(), elems_w);
        let vw = empty::<F>(ctx.client(), elems_w);

        let geom_prep = LaunchGeometry::per_row(ctx.client(), total_bhn, pair_units(head_dim));
        let prep_block = geom_prep.cube_dim.x;
        assert!(prep_block as usize * 2 >= head_dim, "prep cube of {prep_block} units < head_dim / 2");
        let c3 = 3 * c;
        let half_d = head_dim / 2;
        unsafe {
            prep_cswin2d_qkv_kernel::launch::<F>(
                ctx.client(),
                geom_prep.cube_count,
                geom_prep.cube_dim,
                BufferArg::from_raw_parts(c_qkv.handle, b * n_tokens * c3),
                BufferArg::from_raw_parts(self.core.qkv_b.clone(), c3),
                BufferArg::from_raw_parts(self.core.q_norm_w.clone(), head_dim),
                BufferArg::from_raw_parts(self.core.k_norm_w.clone(), head_dim),
                BufferArg::from_raw_parts(cos_h, h * half_d),
                BufferArg::from_raw_parts(sin_h, h * half_d),
                BufferArg::from_raw_parts(cos_w, w * half_d),
                BufferArg::from_raw_parts(sin_w, w * half_d),
                BufferArg::from_raw_parts(qh.clone(), elems_h),
                BufferArg::from_raw_parts(kh.clone(), elems_h),
                BufferArg::from_raw_parts(vh.clone(), elems_h),
                BufferArg::from_raw_parts(qw.clone(), elems_w),
                BufferArg::from_raw_parts(kw.clone(), elems_w),
                BufferArg::from_raw_parts(vw.clone(), elems_w),
                total_bhn as u32,
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
        let ah_handle =
            run_unmasked_sdpa_handles::<F>(ctx, qh, kh, vh, b * w, h_heads, h, head_dim)?;
        let aw_handle =
            run_unmasked_sdpa_handles::<F>(ctx, qw, kw, vw, b * h, w_heads, w, head_dim)?;

        let gate_logits = ctx.matmul(
            x_nhwc.reshape([b * n_tokens, c]),
            self.core.gate_w_t.clone(),
        );
        let c_gate = into_contiguous_cube(gate_logits)?;

        let total_elems = b * n_tokens * c;
        let merged = empty::<F>(ctx.client(), total_elems);
        let geom_merge = LaunchGeometry::elementwise(ctx.client(), total_elems);
        unsafe {
            merge_cswin2d_gate_kernel::launch::<F>(
                ctx.client(),
                geom_merge.cube_count.clone(),
                geom_merge.cube_dim,
                BufferArg::from_raw_parts(ah_handle, elems_h),
                BufferArg::from_raw_parts(aw_handle, elems_w),
                BufferArg::from_raw_parts(c_gate.handle, b * n_tokens * num_heads),
                BufferArg::from_raw_parts(merged.clone(), total_elems),
                total_elems as u32,
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
        Ok(ctx.tensor_from_handle(c_proj.handle, [b, h, w, c], dt))
    }
}

/// Device-resident `CSwinBlock2D` (`corridorkey/blocks.py`).
#[derive(Clone, Debug)]
pub struct CSwinBlock2d {
    pub norm1: RmsNorm,
    pub attn: CSwinAttention2d,
    pub local_mix: Option<DepthwiseSwiGlu2d>,
    pub norm2: RmsNorm,
    pub mlp: SwiGlu,
}

impl CSwinBlock2d {
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
        let local_mix = load_optional_local_mix::<F>(ctx, st, prefix)?;
        let mlp = SwiGlu::load::<F>(ctx, st, &format!("{prefix}.mlp"))?;
        Ok(Self {
            norm1: RmsNorm::from_slice(ctx, &norm1_w, DEFAULT_RMS_NORM_EPS)?,
            attn: CSwinAttention2d { core, rope },
            local_mix,
            norm2: RmsNorm::from_slice(ctx, &norm2_w, DEFAULT_RMS_NORM_EPS)?,
            mlp,
        })
    }

    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x: Tensor<4>) -> CkResult<Tensor<4>> {
        let normed1 = self.norm1.forward::<F, 4>(ctx, x.clone())?;
        let attn_out = self.attn.forward::<F>(ctx, normed1)?;
        let mut h = x + attn_out;
        if let Some(lm) = &self.local_mix {
            h = lm.forward_residual::<F, 4>(ctx, h)?;
        }
        let normed2 = self.norm2.forward::<F, 4>(ctx, h.clone())?;
        self.mlp.forward_residual::<F, 4>(ctx, normed2, h)
    }
}

/// Device-resident `SwinBlock2D` (`corridorkey/blocks.py`).
#[derive(Clone, Debug)]
pub struct SwinBlock2d {
    pub norm1: RmsNorm,
    pub core: AttentionCore,
    pub rope: Rope2d,
    pub window_size: usize,
    pub shift: (usize, usize),
    pub local_mix: Option<DepthwiseSwiGlu2d>,
    pub norm2: RmsNorm,
    pub mlp: SwiGlu,
    region_cache: Arc<Mutex<BTreeMap<(usize, usize), ShiftedRegionGroups>>>,
}

impl SwinBlock2d {
    pub fn load<F: CkFloat>(
        ctx: &MlContext,
        st: &SafetensorsFile,
        prefix: &str,
        num_heads: usize,
        window_size: usize,
        shift: (usize, usize),
        rope: Rope2d,
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
            window_size,
            shift,
            local_mix,
            norm2: RmsNorm::from_slice(ctx, &norm2_w, DEFAULT_RMS_NORM_EPS)?,
            mlp,
            region_cache: Arc::new(Mutex::new(BTreeMap::new())),
        })
    }

    fn region_groups(&self, h: usize, w: usize, hp: usize, wp: usize) -> ShiftedRegionGroups {
        let key = (h, w);
        let mut guard = self.region_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(groups) = guard.get(&key) {
            return groups.clone();
        }
        let (sh, sw) = self.shift;
        let groups = ShiftedRegionGroups::build_2d(h, w, hp, wp, self.window_size, sh, sw);
        guard.insert(key, groups.clone());
        groups
    }

    fn partition_2d<F: CkFloat>(
        ctx: &MlContext,
        src: &Handle,
        b: usize,
        hp: usize,
        wp: usize,
        win: usize,
        ch: usize,
    ) -> Handle {
        let total = b * hp * wp * ch;
        let out = empty::<F>(ctx.client(), total);
        let geom = LaunchGeometry::elementwise(ctx.client(), total);
        unsafe {
            window_partition_2d_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(src.clone(), total),
                BufferArg::from_raw_parts(out.clone(), total),
                total as u32,
                hp as u32,
                wp as u32,
                win as u32,
                ch as u32,
            );
        }
        out
    }

    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x: Tensor<4>) -> CkResult<Tensor<4>> {
        let [b, h, w, c] = x.dims();
        let win = self.window_size;
        let (sh, sw) = self.shift;
        let hp = h.div_ceil(win) * win;
        let wp = w.div_ceil(win) * win;
        let has_shift = sh != 0 || sw != 0;
        let has_pad = hp != h || wp != w;

        let normed1 = ctx.op("swin norm1", || self.norm1.forward::<F, 4>(ctx, x.clone()))?;
        let dt = elem_dtype::<F>();

        // Match Python's exact order for shifted vs unshifted + qkv_dw:
        // - When shifted: pad to (Hp, Wp) and roll by (-sh, -sw) BEFORE qkv + qkv_dw.
        // - When unshifted: if qkv_dw is Some, run qkv + qkv_dw on (H, W) then pad to (Hp, Wp);
        //   if qkv_dw is None, pad to (Hp, Wp) before qkv.
        let (qkv_padded_handle, gate_padded_handle, bias_applied) =
            if has_shift || (has_pad && self.core.qkv_dw.is_none()) {
                let c_norm1 = into_contiguous_cube(normed1)?;
                let total_p = b * hp * wp * c;
                let rolled = empty::<F>(ctx.client(), total_p);
                let geom = LaunchGeometry::elementwise(ctx.client(), total_p);
                unsafe {
                    pad_roll_2d_kernel::launch::<F>(
                        ctx.client(),
                        geom.cube_count,
                        geom.cube_dim,
                        BufferArg::from_raw_parts(c_norm1.handle, b * h * w * c),
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
                let rolled_t = ctx.tensor_from_handle(rolled.clone(), [b, hp, wp, c], dt);
                let (qkv_4d, bias_done) = ctx.op("swin qkv proj+dw", || {
                    self.core.project_qkv_spatial::<F>(ctx, rolled_t)
                })?;
                let c_qkv = into_contiguous_cube(qkv_4d)?;
                (c_qkv.handle, rolled, bias_done)
            } else if has_pad {
                // sh == 0, sw == 0, has_pad == true, qkv_dw.is_some()
                let (qkv_orig, bias_done) = ctx.op("swin qkv proj+dw", || {
                    self.core.project_qkv_spatial::<F>(ctx, normed1.clone())
                })?;
                let c_qkv_orig = into_contiguous_cube(qkv_orig)?;
                let c_norm1 = into_contiguous_cube(normed1)?;
                let c3 = 3 * c;
                let total_qkv_p = b * hp * wp * c3;
                let total_gate_p = b * hp * wp * c;
                let qkv_p = empty::<F>(ctx.client(), total_qkv_p);
                let gate_p = empty::<F>(ctx.client(), total_gate_p);
                let geom_qkv = LaunchGeometry::elementwise(ctx.client(), total_qkv_p);
                let geom_gate = LaunchGeometry::elementwise(ctx.client(), total_gate_p);
                unsafe {
                    pad_roll_2d_kernel::launch::<F>(
                        ctx.client(),
                        geom_qkv.cube_count,
                        geom_qkv.cube_dim,
                        BufferArg::from_raw_parts(c_qkv_orig.handle, b * h * w * c3),
                        BufferArg::from_raw_parts(qkv_p.clone(), total_qkv_p),
                        total_qkv_p as u32,
                        h as u32,
                        w as u32,
                        hp as u32,
                        wp as u32,
                        c3 as u32,
                        0,
                        0,
                    );
                    pad_roll_2d_kernel::launch::<F>(
                        ctx.client(),
                        geom_gate.cube_count,
                        geom_gate.cube_dim,
                        BufferArg::from_raw_parts(c_norm1.handle, b * h * w * c),
                        BufferArg::from_raw_parts(gate_p.clone(), total_gate_p),
                        total_gate_p as u32,
                        h as u32,
                        w as u32,
                        hp as u32,
                        wp as u32,
                        c as u32,
                        0,
                        0,
                    );
                }
                (qkv_p, gate_p, bias_done)
            } else {
                let (qkv_4d, bias_done) = ctx.op("swin qkv proj+dw", || {
                    self.core.project_qkv_spatial::<F>(ctx, normed1.clone())
                })?;
                let c_qkv = into_contiguous_cube(qkv_4d)?;
                let c_norm1 = into_contiguous_cube(normed1)?;
                (c_qkv.handle, c_norm1.handle, bias_done)
            };

        let c3 = 3 * c;
        let (qkv_win, gate_win) = ctx.op("swin partition", || {
            (
                Self::partition_2d::<F>(ctx, &qkv_padded_handle, b, hp, wp, win, c3),
                Self::partition_2d::<F>(ctx, &gate_padded_handle, b, hp, wp, win, c),
            )
        });

        let nw_total = b * (hp / win) * (wp / win);
        let n_win_tokens = win * win;
        let (cos_h, sin_h) = self.rope.cos_sin_handles::<F>(ctx, win, win);

        let (q, k, v) = ctx.op("swin qk norm+rope", || {
            self.core.prep_qkv_norm_rope::<F>(
                ctx,
                qkv_win,
                &cos_h,
                &sin_h,
                nw_total,
                n_win_tokens,
                bias_applied,
            )
        });

        let attn = ctx.op("swin window sdpa", || {
            if has_shift || has_pad {
                let groups = self.region_groups(h, w, hp, wp);
                groups.run_sdpa::<F>(ctx, q, k, v)
            } else {
                run_unmasked_sdpa::<F>(ctx, q, k, v)
            }
        })?;

        let gate_tokens_2d = ctx.tensor_from_handle(gate_win, [nw_total * n_win_tokens, c], dt);
        let proj_win_handle = ctx.op("swin gate+proj", || {
            self.core
                .gate_and_project_out::<F>(ctx, gate_tokens_2d, attn, nw_total, n_win_tokens)
        })?;

        // Fused window unpartition + reverse cyclic shift (+sh, +sw) + crop to (H, W) + residual add
        let unpartition = ctx.op("swin unpartition+residual", || -> CkResult<Handle> {
            let c_shortcut = into_contiguous_cube(x)?;
            let total_out = b * h * w * c;
            let out_handle = empty::<F>(ctx.client(), total_out);
            let geom_out = LaunchGeometry::elementwise(ctx.client(), total_out);
            unsafe {
                window_unpartition_unroll_residual_2d_kernel::launch::<F>(
                    ctx.client(),
                    geom_out.cube_count,
                    geom_out.cube_dim,
                    BufferArg::from_raw_parts(proj_win_handle, b * hp * wp * c),
                    BufferArg::from_raw_parts(c_shortcut.handle, total_out),
                    BufferArg::from_raw_parts(out_handle.clone(), total_out),
                    total_out as u32,
                    h as u32,
                    w as u32,
                    hp as u32,
                    wp as u32,
                    win as u32,
                    c as u32,
                    sh as u32,
                    sw as u32,
                );
            }

            Ok(out_handle)
        })?;
        let mut res = ctx.tensor_from_handle(unpartition, [b, h, w, c], dt);
        if let Some(lm) = &self.local_mix {
            res = ctx.op("swin local mix", || lm.forward_residual::<F, 4>(ctx, res))?;
        }
        let normed2 = ctx.op("swin norm2", || {
            self.norm2.forward::<F, 4>(ctx, res.clone())
        })?;
        ctx.op("swin mlp", || {
            self.mlp.forward_residual::<F, 4>(ctx, normed2, res)
        })
    }
}

/// Polymorphic 2D block used in Stage 2's `pre_blocks` and `final_blocks`.
#[derive(Clone, Debug)]
pub enum Stage2Block2d {
    Full(FullSpatialBlock2d),
    Swin(SwinBlock2d),
    CSwin(CSwinBlock2d),
}

impl Stage2Block2d {
    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x: Tensor<4>) -> CkResult<Tensor<4>> {
        match self {
            Self::Full(b) => b.forward::<F>(ctx, x),
            Self::Swin(b) => b.forward::<F>(ctx, x),
            Self::CSwin(b) => b.forward::<F>(ctx, x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use corridorkey_core::ComputeTarget;

    #[test]
    fn shifted_region_groups_2d_partition_matches_expected_counts() {
        // H=10, W=10, window=4 -> Hp=12, Wp=12 (9 windows of 16 tokens), shift=(2, 2)
        // Region 0 is [0..8, 0..8]; after rolling by (-2, -2) it spans [0..6, 0..6], so
        // window (0, 0) ([0..4, 0..4]) is completely inside region 0.
        let groups = ShiftedRegionGroups::build_2d(10, 10, 12, 12, 4, 2, 2);
        assert_eq!(groups.num_windows, 9);
        assert_eq!(groups.full_windows, vec![0]);
        let mut covered = groups.full_windows.len() * 16;
        for (toks, wins) in &groups.partial_groups {
            covered += toks.len() * wins.len();
        }
        assert_eq!(covered, 100);
    }

    #[test]
    fn stage2_blocks_load_and_execute_on_real_checkpoint() {
        let ckpt_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../weights/corridorkey_v2.safetensors");
        if !ckpt_path.exists() {
            return;
        }
        let st = SafetensorsFile::open(&ckpt_path).unwrap();
        let rope = Rope2d::new(80, 100.0).unwrap();

        for target in ComputeTarget::available() {
            let ctx = MlContext::new(target).unwrap();
            // pre_blocks.0 is FullSpatialBlock2d, pre_blocks.1 is SwinBlock2d(shift=(8,8) + qkv_dw),
            // pre_blocks.2 is CSwinBlock2d
            let blk0 =
                FullSpatialBlock2d::load::<f32>(&ctx, &st, "stage2.pre_blocks.0", 16, rope.clone())
                    .unwrap();
            let blk1 = SwinBlock2d::load::<f32>(
                &ctx,
                &st,
                "stage2.pre_blocks.1",
                16,
                16,
                (8, 8),
                rope.clone(),
            )
            .unwrap();
            let blk2 =
                CSwinBlock2d::load::<f32>(&ctx, &st, "stage2.pre_blocks.2", 16, rope.clone())
                    .unwrap();

            let (bt, h, w, c) = (1, 16, 16, 1280);
            let input: Vec<f32> = (0..bt * h * w * c)
                .map(|i| ((i as f32) * 0.013).sin() * 0.1)
                .collect();
            let x = ctx.upload_tensor(&input, [bt, h, w, c]).unwrap();
            let y0 = blk0.forward::<f32>(&ctx, x).unwrap();
            let y1 = blk1.forward::<f32>(&ctx, y0).unwrap();
            let y2 = blk2.forward::<f32>(&ctx, y1).unwrap();
            let out = ctx.download_tensor::<f32, 4>(y2).unwrap();
            assert_eq!(out.len(), bt * h * w * c);
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{target} produced non-finite output"
            );
        }
    }
}
