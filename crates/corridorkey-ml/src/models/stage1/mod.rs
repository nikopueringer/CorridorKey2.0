//! Stage 1 (`/16 ↔ /64` spatiotemporal transformer baton producer, `corridorkey/stage1.py`).
//!
//! Operates end-to-end on channels-last `[B, T, H, W, C]` tensors on the device with zero
//! `NCHW <-> NHWC` permutations.

use burn_tensor::Tensor;
use corridorkey_base::core::{
    CkFloat,
    buffer::{empty, upload},
};
use corridorkey_core::{CkError, CkResult, compute::LaunchGeometry};
use cubecl::{prelude::BufferArg, server::Handle};

use crate::{
    Provenance,
    hub::{ModelConfig, SafetensorsFile},
    models::{
        blocks::{
            CSwinBlock3d, Depatchify, Downsample4, FullSpatialBlock3d, FullSpatioTemporalBlock,
            FusionLinSwiglu, PatchifyConv, Stage1DecBlock3d, SwinBlock3d, Upsample4Icnr,
            kernels::add_scaled_skip_kernel,
        },
        rope::{Rope2d, Rope3d},
    },
    provenance::corridorkey_v2_provenance,
    runtime::{MlContext, elem_dtype, into_contiguous_cube},
};

/// Spatial quantum (`/64`) required by Stage 1.
pub const STAGE1_SHAPE_QUANTUM: usize = 64;

/// Patch stride at Stage 1's high-resolution token grid (`/16`).
pub const STAGE1_PATCH_SIZE: usize = 16;

/// Output of [`Stage1::forward`].
#[derive(Clone, Debug)]
pub struct Stage1Output {
    /// `[B, T, H / 16, W / 16, d_16]` baton tensor feeding [`crate::Stage2`].
    pub baton: Tensor<5>,
    /// `[B, T, H / 2, W / 2, 4]` auxiliary half-res prediction (`[alpha_logit, fg_rgb]`).
    pub aux: Tensor<5>,
}

fn add_scaled_skip<F: CkFloat>(
    ctx: &MlContext,
    a: Tensor<5>,
    b: Tensor<5>,
    scale: &Handle,
    dim: usize,
) -> CkResult<Tensor<5>> {
    let shape = a.dims();
    let total: usize = shape.iter().product();
    let c_a = into_contiguous_cube(a)?;
    let c_b = into_contiguous_cube(b)?;
    let out = empty::<F>(ctx.client(), total);
    let geom = LaunchGeometry::elementwise(ctx.client(), total);
    unsafe {
        add_scaled_skip_kernel::launch::<F>(
            ctx.client(),
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(c_a.handle, total),
            BufferArg::from_raw_parts(c_b.handle, total),
            BufferArg::from_raw_parts(scale.clone(), dim),
            BufferArg::from_raw_parts(out.clone(), total),
            total as u32,
            dim as u32,
        );
    }
    Ok(ctx.tensor_from_handle(out, shape, elem_dtype::<F>()))
}

/// Device-resident Stage 1 spatiotemporal network (`corridorkey/stage1.py`).
#[derive(Clone, Debug)]
pub struct Stage1 {
    pub stem: PatchifyConv,
    pub enc_16: Vec<FullSpatialBlock3d>,
    pub down: Downsample4,
    pub fuse: Option<FusionLinSwiglu>,
    pub body_64: Vec<FullSpatioTemporalBlock>,
    pub mid_up: Upsample4Icnr,
    pub mid_skip_scale: Handle,
    pub mid_16: Vec<SwinBlock3d>,
    pub mid_down: Downsample4,
    pub up: Upsample4Icnr,
    pub uskip_scale: Handle,
    pub patch_late_skip_scale: Handle,
    pub dec_16: Vec<Stage1DecBlock3d>,
    pub aux_depatchify: Depatchify,
    pub d_16: usize,
    pub d_64: usize,
    pub vfm_channels: usize,
}

impl Stage1 {
    pub fn provenance() -> Provenance {
        corridorkey_v2_provenance()
    }

    /// Loads `Stage1` weights from `SafetensorsFile` under `prefix` (typically `"stage1"`).
    pub fn load<F: CkFloat>(
        ctx: &MlContext,
        st: &SafetensorsFile,
        cfg: &ModelConfig,
        prefix: &str,
    ) -> CkResult<Self> {
        let d_16 = cfg.d_16;
        let d_64 = cfg.d_64;
        let num_heads_16 = cfg.num_heads_16;
        let num_heads_64 = cfg.num_heads_64;
        let win = cfg.window_hw;
        let hw = win / 2;

        let rope_16_spatial = Rope2d::new(d_16 / num_heads_16, cfg.rope_base)?;
        let rope_16 = Rope3d::new(d_16 / num_heads_16, cfg.rope_base)?;
        let rope_64 = Rope3d::new(d_64 / num_heads_64, cfg.rope_base)?;

        let stem = PatchifyConv::load::<F>(ctx, st, &format!("{prefix}.stem"), STAGE1_PATCH_SIZE)?;

        let mut enc_16 = Vec::with_capacity(cfg.n_enc_16);
        for i in 0..cfg.n_enc_16 {
            enc_16.push(FullSpatialBlock3d::load::<F>(
                ctx,
                st,
                &format!("{prefix}.enc_16.{i}"),
                num_heads_16,
                rope_16_spatial.clone(),
            )?);
        }

        let down = Downsample4::load::<F>(ctx, st, &format!("{prefix}.down"))?;
        let fuse = if cfg.stage1_body64_direct_concat {
            None
        } else {
            Some(FusionLinSwiglu::load::<F>(
                ctx,
                st,
                &format!("{prefix}.fuse"),
            )?)
        };
        let vfm_channels = match &fuse {
            Some(f) => f.in_dim().saturating_sub(d_16),
            None => d_64.saturating_sub(d_16),
        };

        let mut body_64 = Vec::with_capacity(cfg.n_body_64);
        for i in 0..cfg.n_body_64 {
            body_64.push(FullSpatioTemporalBlock::load::<F>(
                ctx,
                st,
                &format!("{prefix}.body_64.{i}"),
                num_heads_64,
                rope_64.clone(),
            )?);
        }

        let mid_up = Upsample4Icnr::load::<F>(ctx, st, &format!("{prefix}.mid_up"), 4)?;
        let (_, mid_skip_vec) = st.read_as::<F>(&format!("{prefix}.mid_skip_scale"))?;
        let mid_skip_scale = upload(ctx.client(), &mid_skip_vec);

        let mid_phases = [(0, 0), (hw, hw), (hw, 0), (0, hw)];
        let mut mid_16 = Vec::with_capacity(cfg.stage1_mid_16_blocks);
        for i in 0..cfg.stage1_mid_16_blocks {
            mid_16.push(SwinBlock3d::load::<F>(
                ctx,
                st,
                &format!("{prefix}.mid_16.{i}"),
                num_heads_16,
                win,
                mid_phases[i % mid_phases.len()],
                rope_16.clone(),
            )?);
        }
        let mid_down = Downsample4::load::<F>(ctx, st, &format!("{prefix}.mid_down"))?;

        let up = Upsample4Icnr::load::<F>(ctx, st, &format!("{prefix}.up"), 4)?;
        let (_, uskip_vec) = st.read_as::<F>(&format!("{prefix}.uskip_scale"))?;
        let (_, late_skip_vec) = st.read_as::<F>(&format!("{prefix}.patch_late_skip_scale"))?;
        let uskip_scale = upload(ctx.client(), &uskip_vec);
        let patch_late_skip_scale = upload(ctx.client(), &late_skip_vec);

        let dec_specs: &[(&str, (usize, usize))] = match cfg.n_dec_16 {
            8 => &[
                ("swin", (0, 0)),
                ("swin", (hw, hw)),
                ("cswin", (0, 0)),
                ("swin", (hw, 0)),
                ("swin", (0, hw)),
                ("cswin", (0, 0)),
                ("swin", (hw, 0)),
                ("swin", (0, hw)),
            ],
            6 => &[
                ("swin", (0, 0)),
                ("swin", (hw, hw)),
                ("cswin", (0, 0)),
                ("cswin", (0, 0)),
                ("swin", (hw, 0)),
                ("swin", (0, hw)),
            ],
            other => {
                return Err(CkError::InvalidConfig(format!(
                    "Stage1 n_dec_16 must be 6 or 8, got {other}"
                )));
            }
        };

        let mut dec_16 = Vec::with_capacity(dec_specs.len());
        for (i, &(kind, shift)) in dec_specs.iter().enumerate() {
            let blk_prefix = format!("{prefix}.dec_16.{i}");
            let blk = match kind {
                "swin" => Stage1DecBlock3d::Swin(SwinBlock3d::load::<F>(
                    ctx,
                    st,
                    &blk_prefix,
                    num_heads_16,
                    win,
                    shift,
                    rope_16.clone(),
                )?),
                "cswin" => Stage1DecBlock3d::CSwin(CSwinBlock3d::load::<F>(
                    ctx,
                    st,
                    &blk_prefix,
                    num_heads_16,
                    rope_16.clone(),
                )?),
                _ => unreachable!(),
            };
            dec_16.push(blk);
        }

        let aux_depatchify =
            Depatchify::load::<F>(ctx, st, &format!("{prefix}.aux_depatchify"), 8)?;

        Ok(Self {
            stem,
            enc_16,
            down,
            fuse,
            body_64,
            mid_up,
            mid_skip_scale,
            mid_16,
            mid_down,
            up,
            uskip_scale,
            patch_late_skip_scale,
            dec_16,
            aux_depatchify,
            d_16,
            d_64,
            vfm_channels,
        })
    }

    /// Runs Stage 1 to produce the `/16` `baton` (`[B, T, H / 16, W / 16, d_16]`), skipping the
    /// training-only `aux_depatchify` head.
    pub fn forward_baton<F: CkFloat>(
        &self,
        ctx: &MlContext,
        rgb_hint_bthwc: Tensor<5>,
        vfm_64_bthwc: Tensor<5>,
        hint_quality_bt: Option<Tensor<2>>,
        t_stride: f64,
    ) -> CkResult<Tensor<5>> {
        let [b, t, h, w, c_in] = rgb_hint_bthwc.dims();
        if c_in != 4
            || !h.is_multiple_of(STAGE1_SHAPE_QUANTUM)
            || !w.is_multiple_of(STAGE1_SHAPE_QUANTUM)
        {
            return Err(CkError::Shape(format!(
                "Stage1 expects rgb_hint [B, T, H, W, 4] with /64-aligned H, W; got [{b}, {t}, {h}, {w}, {c_in}]"
            )));
        }
        let bt = b * t;
        let h16 = h / 16;
        let w16 = w / 16;
        let h64 = h / 64;
        let w64 = w / 64;
        if vfm_64_bthwc.dims() != [b, t, h64, w64, self.vfm_channels] {
            return Err(CkError::Shape(format!(
                "Stage1 expected vfm_64 [{b}, {t}, {h64}, {w64}, {}], got {:?}",
                self.vfm_channels,
                vfm_64_bthwc.dims()
            )));
        }

        let f_16_patch = self
            .stem
            .forward::<F>(ctx, rgb_hint_bthwc.reshape([bt, h, w, 4]))?
            .reshape([b, t, h16, w16, self.d_16]);

        let mut f_16_enc = f_16_patch.clone();
        for blk in &self.enc_16 {
            f_16_enc = blk.forward::<F>(ctx, f_16_enc, hint_quality_bt.clone())?;
        }

        let f_64_down = self
            .down
            .forward::<F>(ctx, f_16_enc.clone().reshape([bt, h16, w16, self.d_16]))?
            .reshape([b, t, h64, w64, self.d_16]);

        let combined = Tensor::cat(vec![vfm_64_bthwc, f_64_down], 4);
        let mut f_64 = match &self.fuse {
            Some(fuse) => fuse.forward::<F, 5>(ctx, combined)?,
            None => combined,
        };

        let split = self.body_64.len() / 2;
        for blk in &self.body_64[..split] {
            f_64 = blk.forward::<F>(ctx, f_64, t_stride)?;
        }

        let f_16_mid_up = self
            .mid_up
            .forward::<F>(ctx, f_64.clone().reshape([bt, h64, w64, self.d_64]))?
            .reshape([b, t, h16, w16, self.d_16]);
        let mut f_16_mid =
            add_scaled_skip::<F>(ctx, f_16_mid_up, f_16_enc, &self.mid_skip_scale, self.d_16)?;
        for blk in &self.mid_16 {
            f_16_mid = blk.forward::<F>(ctx, f_16_mid, t_stride, None)?;
        }
        let f_64_delta = self
            .mid_down
            .forward::<F>(ctx, f_16_mid.clone().reshape([bt, h16, w16, self.d_16]))?
            .reshape([b, t, h64, w64, self.d_64]);
        f_64 = f_64 + f_64_delta;

        for blk in &self.body_64[split..] {
            f_64 = blk.forward::<F>(ctx, f_64, t_stride)?;
        }

        let f_16_up = self
            .up
            .forward::<F>(ctx, f_64.reshape([bt, h64, w64, self.d_64]))?
            .reshape([b, t, h16, w16, self.d_16]);
        let mut x_dec = add_scaled_skip::<F>(ctx, f_16_up, f_16_mid, &self.uskip_scale, self.d_16)?;

        let late_skip_idx = self.dec_16.len().saturating_sub(2);
        for (i, blk) in self.dec_16.iter().enumerate() {
            if i == late_skip_idx {
                x_dec = add_scaled_skip::<F>(
                    ctx,
                    x_dec,
                    f_16_patch.clone(),
                    &self.patch_late_skip_scale,
                    self.d_16,
                )?;
            }
            x_dec = blk.forward::<F>(ctx, x_dec, t_stride)?;
        }
        Ok(x_dec)
    }

    /// Profiles the individual sub-components of Stage 1 with device-synchronized timings.
    pub fn profile_breakdown<F: CkFloat>(
        &self,
        ctx: &MlContext,
        rgb_hint_bthwc: Tensor<5>,
        vfm_64_bthwc: Tensor<5>,
        hint_quality_bt: Option<Tensor<2>>,
        t_stride: f64,
    ) -> CkResult<()> {
        use corridorkey_core::compute::bench::time_device;
        let [b, t, h, w, _] = rgb_hint_bthwc.dims();
        let bt = b * t;
        let h16 = h / 16;
        let w16 = w / 16;
        let h64 = h / 64;
        let w64 = w / 64;

        let d_stem = time_device(ctx.client(), 3, || {
            let _ = self
                .stem
                .forward::<F>(ctx, rgb_hint_bthwc.clone().reshape([bt, h, w, 4]))
                .unwrap();
        });
        println!(
            "    Stage 1 stem (patchify):                   {:6.2} ms",
            d_stem.as_secs_f64() * 1e3
        );

        let f_16_patch = self
            .stem
            .forward::<F>(ctx, rgb_hint_bthwc.reshape([bt, h, w, 4]))?
            .reshape([b, t, h16, w16, self.d_16]);
        let d_enc = time_device(ctx.client(), 3, || {
            let mut f = f_16_patch.clone();
            for blk in &self.enc_16 {
                f = blk.forward::<F>(ctx, f, hint_quality_bt.clone()).unwrap();
            }
        });
        println!(
            "    Stage 1 enc_16 (2 FullSpatial blocks):      {:6.2} ms ({:.2} ms/blk)",
            d_enc.as_secs_f64() * 1e3,
            (d_enc.as_secs_f64() * 1e3) / self.enc_16.len() as f64
        );
        let mut f_16_enc = f_16_patch.clone();
        for blk in &self.enc_16 {
            f_16_enc = blk.forward::<F>(ctx, f_16_enc, hint_quality_bt.clone())?;
        }

        let d_down_fuse = time_device(ctx.client(), 3, || {
            let f_64_down = self
                .down
                .forward::<F>(ctx, f_16_enc.clone().reshape([bt, h16, w16, self.d_16]))
                .unwrap()
                .reshape([b, t, h64, w64, self.d_16]);
            let combined = Tensor::cat(vec![vfm_64_bthwc.clone(), f_64_down], 4);
            let _ = match &self.fuse {
                Some(fuse) => fuse.forward::<F, 5>(ctx, combined).unwrap(),
                None => combined,
            };
        });
        println!(
            "    Stage 1 downsample + fusion:               {:6.2} ms",
            d_down_fuse.as_secs_f64() * 1e3
        );
        let f_64_down = self
            .down
            .forward::<F>(ctx, f_16_enc.clone().reshape([bt, h16, w16, self.d_16]))?
            .reshape([b, t, h64, w64, self.d_16]);
        let combined = Tensor::cat(vec![vfm_64_bthwc, f_64_down], 4);
        let mut f_64 = match &self.fuse {
            Some(fuse) => fuse.forward::<F, 5>(ctx, combined)?,
            None => combined,
        };

        let split = self.body_64.len() / 2;
        let d_body1 = time_device(ctx.client(), 3, || {
            let mut f = f_64.clone();
            for blk in &self.body_64[..split] {
                f = blk.forward::<F>(ctx, f, t_stride).unwrap();
            }
        });
        println!(
            "    Stage 1 body_64 first half (5 blocks):     {:6.2} ms ({:.2} ms/blk)",
            d_body1.as_secs_f64() * 1e3,
            (d_body1.as_secs_f64() * 1e3) / split as f64
        );
        for blk in &self.body_64[..split] {
            f_64 = blk.forward::<F>(ctx, f_64, t_stride)?;
        }

        let d_mid_up = time_device(ctx.client(), 3, || {
            let _ = self
                .mid_up
                .forward::<F>(ctx, f_64.clone().reshape([bt, h64, w64, self.d_64]))
                .unwrap();
        });
        println!(
            "    Stage 1 mid_up (Upsample4Icnr):            {:6.2} ms",
            d_mid_up.as_secs_f64() * 1e3
        );

        let f_16_mid_up = self
            .mid_up
            .forward::<F>(ctx, f_64.clone().reshape([bt, h64, w64, self.d_64]))?
            .reshape([b, t, h16, w16, self.d_16]);
        let mut f_16_mid =
            add_scaled_skip::<F>(ctx, f_16_mid_up, f_16_enc, &self.mid_skip_scale, self.d_16)?;

        for (i, blk) in self.mid_16.iter().enumerate() {
            let d_blk = time_device(ctx.client(), 3, || {
                let _ = blk
                    .forward::<F>(ctx, f_16_mid.clone(), t_stride, None)
                    .unwrap();
            });
            println!(
                "      mid_16 block {i} (shift {:?}):         {:6.2} ms",
                blk.shift,
                d_blk.as_secs_f64() * 1e3
            );
            f_16_mid = blk.forward::<F>(ctx, f_16_mid, t_stride, None)?;
        }

        let d_mid_down = time_device(ctx.client(), 3, || {
            let _ = self
                .mid_down
                .forward::<F>(ctx, f_16_mid.clone().reshape([bt, h16, w16, self.d_16]))
                .unwrap();
        });
        println!(
            "    Stage 1 mid_down (Downsample4):            {:6.2} ms",
            d_mid_down.as_secs_f64() * 1e3
        );
        let f_64_delta = self
            .mid_down
            .forward::<F>(ctx, f_16_mid.clone().reshape([bt, h16, w16, self.d_16]))?
            .reshape([b, t, h64, w64, self.d_64]);
        f_64 = f_64 + f_64_delta;

        let d_body2 = time_device(ctx.client(), 3, || {
            let mut f = f_64.clone();
            for blk in &self.body_64[split..] {
                f = blk.forward::<F>(ctx, f, t_stride).unwrap();
            }
        });
        println!(
            "    Stage 1 body_64 second half (5 blocks):    {:6.2} ms ({:.2} ms/blk)",
            d_body2.as_secs_f64() * 1e3,
            (d_body2.as_secs_f64() * 1e3) / (self.body_64.len() - split) as f64
        );
        for blk in &self.body_64[split..] {
            f_64 = blk.forward::<F>(ctx, f_64, t_stride)?;
        }

        let d_dec = time_device(ctx.client(), 3, || {
            let f_16_up = self
                .up
                .forward::<F>(ctx, f_64.clone().reshape([bt, h64, w64, self.d_64]))
                .unwrap()
                .reshape([b, t, h16, w16, self.d_16]);
            let mut x_dec =
                add_scaled_skip::<F>(ctx, f_16_up, f_16_mid.clone(), &self.uskip_scale, self.d_16)
                    .unwrap();
            let late_skip_idx = self.dec_16.len().saturating_sub(2);
            for (i, blk) in self.dec_16.iter().enumerate() {
                if i == late_skip_idx {
                    x_dec = add_scaled_skip::<F>(
                        ctx,
                        x_dec,
                        f_16_patch.clone(),
                        &self.patch_late_skip_scale,
                        self.d_16,
                    )
                    .unwrap();
                }
                x_dec = blk.forward::<F>(ctx, x_dec, t_stride).unwrap();
            }
        });
        println!(
            "    Stage 1 dec_16 (8 mixed Swin/CSwin):       {:6.2} ms ({:.2} ms/blk)",
            d_dec.as_secs_f64() * 1e3,
            (d_dec.as_secs_f64() * 1e3) / self.dec_16.len() as f64
        );
        Ok(())
    }

    /// Runs Stage 1 and returns both `baton` (`[B, T, H / 16, W / 16, d_16]`) and `aux`
    /// (`[B, T, H / 2, W / 2, 4]`).
    pub fn forward<F: CkFloat>(
        &self,
        ctx: &MlContext,
        rgb_hint_bthwc: Tensor<5>,
        vfm_64_bthwc: Tensor<5>,
        hint_quality_bt: Option<Tensor<2>>,
        t_stride: f64,
    ) -> CkResult<Stage1Output> {
        let [b, t, h, w, _] = rgb_hint_bthwc.dims();
        let bt = b * t;
        let h16 = h / 16;
        let w16 = w / 16;
        let baton =
            self.forward_baton::<F>(ctx, rgb_hint_bthwc, vfm_64_bthwc, hint_quality_bt, t_stride)?;
        let aux_4d = self
            .aux_depatchify
            .forward::<F>(ctx, baton.clone().reshape([bt, h16, w16, self.d_16]))?;
        let aux = aux_4d.reshape([b, t, h / 2, w / 2, 4]);
        Ok(Stage1Output { baton, aux })
    }
}
