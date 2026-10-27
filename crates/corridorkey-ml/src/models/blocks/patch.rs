//! NHWC patchify, depatchify, downsample (`/16 -> /64`), and ICNR upsample (`/64 -> /16`)
//! blocks (`corridorkey/blocks.py`).
//!
//! All modules operate directly on channels-last `[BT, H, W, C]` tensors on the device without
//! NCHW <-> NHWC layout permutations.

use burn_cubecl::kernel::conv::{ConvStrategy, conv_forward_nhwc};
use burn_tensor::{Tensor, ops::ConvOptions};
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
            add_gated_rgb_skip_kernel, add_row_bias_kernel, col2im_nhwc_kernel, im2col_nhwc_kernel,
        },
        norm::{DEFAULT_RMS_NORM_EPS, RmsNorm},
    },
    runtime::{MlContext, elem_dtype, from_cube, into_contiguous_cube, into_cube},
};

/// Permutes a PyTorch `Conv2d` weight from OIHW `[c_out, c_in, k, k]` to
/// `[k * k * c_in, c_out]` (`(ky, kx, c_in, c_out)`) for direct `im2col_nhwc_kernel` + GEMM.
pub fn conv_weight_oihw_to_k2io<F: Copy>(
    weight_oihw: &[F],
    c_out: usize,
    c_in: usize,
    k: usize,
) -> Vec<F> {
    let k2 = k * k;
    assert_eq!(weight_oihw.len(), c_out * c_in * k2);
    let mut out = Vec::with_capacity(weight_oihw.len());
    for ky in 0..k {
        for kx in 0..k {
            for ci in 0..c_in {
                for co in 0..c_out {
                    let src = ((co * c_in + ci) * k + ky) * k + kx;
                    out.push(weight_oihw[src]);
                }
            }
        }
    }
    out
}

/// Permutes a PyTorch `Conv2d` weight from OIHW `[c_out, c_in, kh, kw]` to
/// CubeCL's NHWC-native OHWI `[c_out, kh, kw, c_in]`.
pub fn conv_weight_oihw_to_ohwi<F: Copy>(
    weight_oihw: &[F],
    c_out: usize,
    c_in: usize,
    kh: usize,
    kw: usize,
) -> Vec<F> {
    assert_eq!(weight_oihw.len(), c_out * c_in * kh * kw);
    let mut out = Vec::with_capacity(weight_oihw.len());
    for o in 0..c_out {
        for y in 0..kh {
            for x in 0..kw {
                for i in 0..c_in {
                    let src = ((o * c_in + i) * kh + y) * kw + x;
                    out.push(weight_oihw[src]);
                }
            }
        }
    }
    out
}

/// Permutes a PyTorch `ConvTranspose2d` weight from IOHW `[c_in, c_out, k, k]` to
/// `[c_in, k * k * c_out]` (`(c_in, ky, kx, c_out)`) for direct NHWC GEMM + `col2im_nhwc_kernel`.
pub fn deconv_weight_iohw_to_ik2o<F: Copy>(
    weight_iohw: &[F],
    c_in: usize,
    c_out: usize,
    k: usize,
) -> Vec<F> {
    let k2 = k * k;
    assert_eq!(weight_iohw.len(), c_in * c_out * k2);
    let mut out = Vec::with_capacity(weight_iohw.len());
    for i in 0..c_in {
        for ky in 0..k {
            for kx in 0..k {
                for o in 0..c_out {
                    let src = ((i * c_out + o) * k + ky) * k + kx;
                    out.push(weight_iohw[src]);
                }
            }
        }
    }
    out
}

/// Permutes a PyTorch `Linear` / `1×1 Conv2d` weight feeding `F.pixel_shuffle(r)` from
/// `[c_out * r * r, c_in]` (where `c_out` is outermost in PyTorch's pixel shuffle) to
/// `[c_in, r * r * c_out]` (`(c_in, ry, rx, c_out)`), and bias `[c_out * r * r]` to
/// `[r * r * c_out]`.
pub fn pixel_shuffle_linear_to_ir2o<F: Copy>(
    weight_oi: &[F],
    bias_o: &[F],
    c_in: usize,
    c_out: usize,
    r: usize,
) -> (Vec<F>, Vec<F>) {
    let r2 = r * r;
    assert_eq!(weight_oi.len(), c_out * r2 * c_in);
    assert_eq!(bias_o.len(), c_out * r2);
    let mut w_out = Vec::with_capacity(weight_oi.len());
    for i in 0..c_in {
        for ry in 0..r {
            for rx in 0..r {
                for o in 0..c_out {
                    let row = (o * r + ry) * r + rx;
                    w_out.push(weight_oi[row * c_in + i]);
                }
            }
        }
    }
    let mut b_out = Vec::with_capacity(bias_o.len());
    for ry in 0..r {
        for rx in 0..r {
            for o in 0..c_out {
                let row = (o * r + ry) * r + rx;
                b_out.push(bias_o[row]);
            }
        }
    }
    (w_out, b_out)
}

fn run_conv2d_nhwc(
    x_nhwc: Tensor<4>,
    weight_ohwi: Tensor<4>,
    bias: Option<Tensor<1>>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> CkResult<Tensor<4>> {
    let c_x = into_cube(x_nhwc)?;
    let c_w = into_cube(weight_ohwi)?;
    let c_b = match bias {
        Some(b) => Some(into_cube(b)?),
        None => None,
    };
    let options = ConvOptions::new(stride, padding, [1, 1], 1);
    let out = conv_forward_nhwc::<2>(c_x, c_w, c_b, options, ConvStrategy::default())
        .map_err(|e| CkError::Shape(format!("conv_forward_nhwc failed: {e:?}")))?;
    Ok(from_cube(out))
}

/// Device-resident `PatchifyConv` (`corridorkey/blocks.py`).
///
/// Turns NHWC `[BT, H, W, C_in]` into NHWC `[BT, H / patch, W / patch, C_out]` using
/// `kernel = overlap_mult * patch`, `stride = patch`, `padding = (kernel - patch) / 2`.
/// Implemented via `im2col_nhwc_kernel` + tiled GEMM so `in_channels = 3` or `4` runs on
/// tensor cores / tiled matmul instead of falling back to scalar direct convolution.
#[derive(Clone, Debug)]
pub struct PatchifyConv {
    weight_k2io: Tensor<2>,
    bias: Handle,
    in_channels: usize,
    out_channels: usize,
    patch: usize,
    overlap_mult: usize,
}

impl PatchifyConv {
    pub fn from_slices<F: CkFloat>(
        ctx: &MlContext,
        in_channels: usize,
        out_channels: usize,
        patch: usize,
        overlap_mult: usize,
        weight_oihw: &[F],
        bias: &[F],
    ) -> CkResult<Self> {
        let k = overlap_mult * patch;
        if !(k - patch).is_multiple_of(2) {
            return Err(CkError::InvalidConfig(format!(
                "PatchifyConv kernel ({k}) - patch ({patch}) must be even"
            )));
        }
        let k2_i = k * k * in_channels;
        if weight_oihw.len() != out_channels * k2_i || bias.len() != out_channels {
            return Err(CkError::Shape(
                "PatchifyConv weight/bias shape mismatch".into(),
            ));
        }
        let k2io = conv_weight_oihw_to_k2io(weight_oihw, out_channels, in_channels, k);
        Ok(Self {
            weight_k2io: ctx.upload_tensor(&k2io, [k2_i, out_channels])?,
            bias: upload(ctx.client(), bias),
            in_channels,
            out_channels,
            patch,
            overlap_mult,
        })
    }

    pub fn load<F: CkFloat>(
        ctx: &MlContext,
        st: &SafetensorsFile,
        prefix: &str,
        patch: usize,
    ) -> CkResult<Self> {
        let (w_shape, weight) = st.read_as::<F>(&format!("{prefix}.conv.weight"))?;
        let (_, bias) = st.read_as::<F>(&format!("{prefix}.conv.bias"))?;
        if w_shape.len() != 4 || w_shape[2] != w_shape[3] || !w_shape[2].is_multiple_of(patch) {
            return Err(CkError::Weights(format!(
                "unexpected PatchifyConv weight shape {w_shape:?} at '{prefix}'"
            )));
        }
        let overlap_mult = w_shape[2] / patch;
        Self::from_slices(
            ctx,
            w_shape[1],
            w_shape[0],
            patch,
            overlap_mult,
            &weight,
            &bias,
        )
    }

    pub const fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x_nhwc: Tensor<4>) -> CkResult<Tensor<4>> {
        let [bt, h, w, c] = x_nhwc.dims();
        if c != self.in_channels {
            return Err(CkError::Shape(format!(
                "PatchifyConv expected in_channels={}, got {c}",
                self.in_channels
            )));
        }
        if !h.is_multiple_of(self.patch) || !w.is_multiple_of(self.patch) {
            return Err(CkError::Shape(format!(
                "PatchifyConv spatial dims ({h}, {w}) must divide patch={}",
                self.patch
            )));
        }
        let h_out = h / self.patch;
        let w_out = w / self.patch;
        let tokens = bt * h_out * w_out;
        let k = self.overlap_mult * self.patch;
        let pad = (k - self.patch) / 2;
        let k2_i = k * k * self.in_channels;

        let c_x = into_contiguous_cube(x_nhwc)?;
        let total_cols = tokens * k2_i;
        let cols_h = empty::<F>(ctx.client(), total_cols);
        let geom_col = LaunchGeometry::elementwise(ctx.client(), total_cols);
        unsafe {
            im2col_nhwc_kernel::launch::<F>(
                ctx.client(),
                geom_col.cube_count,
                geom_col.cube_dim,
                BufferArg::from_raw_parts(c_x.handle, bt * h * w * c),
                BufferArg::from_raw_parts(cols_h.clone(), total_cols),
                total_cols as u32,
                h as u32,
                w as u32,
                c as u32,
                h_out as u32,
                w_out as u32,
                k as u32,
                self.patch as u32,
                pad as u32,
            );
        }

        let cols_2d = ctx.tensor_from_handle(cols_h, [tokens, k2_i], elem_dtype::<F>());
        let out_2d = ctx.matmul(cols_2d, self.weight_k2io.clone());
        let c_out = into_contiguous_cube(out_2d)?;
        let total_out = tokens * self.out_channels;
        let geom_out = LaunchGeometry::elementwise(ctx.client(), total_out);
        unsafe {
            add_row_bias_kernel::launch::<F>(
                ctx.client(),
                geom_out.cube_count,
                geom_out.cube_dim,
                BufferArg::from_raw_parts(c_out.handle.clone(), total_out),
                BufferArg::from_raw_parts(self.bias.clone(), self.out_channels),
                total_out as u32,
                self.out_channels as u32,
            );
        }
        Ok(ctx.tensor_from_handle(
            c_out.handle,
            [bt, h_out, w_out, self.out_channels],
            elem_dtype::<F>(),
        ))
    }
}

/// Device-resident `Depatchify` (`corridorkey/blocks.py`).
///
/// Maps NHWC tokens `[BT, H_in, W_in, C_in]` to NHWC pixels `[BT, H_in * r, W_in * r, C_out]`,
/// supporting both `overlap_mult == 1` (`Linear + PixelShuffle`) and `overlap_mult > 1`
/// (`ConvTranspose2d` with `kernel = overlap_mult * r, stride = r, padding = (kernel - r) / 2`).
#[derive(Clone, Debug)]
pub struct Depatchify {
    weight_ik2o: Tensor<2>,
    col_bias: Option<Handle>,
    out_bias: Handle,
    in_dim: usize,
    out_channels: usize,
    r: usize,
    overlap_mult: usize,
}

impl Depatchify {
    pub fn from_deconv_slices<F: CkFloat>(
        ctx: &MlContext,
        in_dim: usize,
        out_channels: usize,
        r: usize,
        overlap_mult: usize,
        weight_iohw: &[F],
        bias: &[F],
    ) -> CkResult<Self> {
        let k = overlap_mult * r;
        let k2_o = k * k * out_channels;
        if weight_iohw.len() != in_dim * k2_o || bias.len() != out_channels {
            return Err(CkError::Shape(
                "Depatchify deconv weight/bias shape mismatch".into(),
            ));
        }
        let ik2o = deconv_weight_iohw_to_ik2o(weight_iohw, in_dim, out_channels, k);
        Ok(Self {
            weight_ik2o: ctx.upload_tensor(&ik2o, [in_dim, k2_o])?,
            col_bias: None,
            out_bias: upload(ctx.client(), bias),
            in_dim,
            out_channels,
            r,
            overlap_mult,
        })
    }

    pub fn from_linear_slices<F: CkFloat>(
        ctx: &MlContext,
        in_dim: usize,
        out_channels: usize,
        r: usize,
        weight_oi: &[F],
        bias_o: &[F],
    ) -> CkResult<Self> {
        let r2_o = r * r * out_channels;
        if weight_oi.len() != r2_o * in_dim || bias_o.len() != r2_o {
            return Err(CkError::Shape(
                "Depatchify linear weight/bias shape mismatch".into(),
            ));
        }
        let (w_ir2o, b_r2o) =
            pixel_shuffle_linear_to_ir2o(weight_oi, bias_o, in_dim, out_channels, r);
        let zeros = vec![corridorkey_base::core::cast::<F>(0.0); out_channels];
        Ok(Self {
            weight_ik2o: ctx.upload_tensor(&w_ir2o, [in_dim, r2_o])?,
            col_bias: Some(upload(ctx.client(), &b_r2o)),
            out_bias: upload(ctx.client(), &zeros),
            in_dim,
            out_channels,
            r,
            overlap_mult: 1,
        })
    }

    pub fn load<F: CkFloat>(
        ctx: &MlContext,
        st: &SafetensorsFile,
        prefix: &str,
        r: usize,
    ) -> CkResult<Self> {
        let deconv_key = format!("{prefix}.deconv.weight");
        if st.entry(&deconv_key).is_ok() {
            let (w_shape, weight) = st.read_as::<F>(&deconv_key)?;
            let (_, bias) = st.read_as::<F>(&format!("{prefix}.deconv.bias"))?;
            if w_shape.len() != 4 || w_shape[2] != w_shape[3] || !w_shape[2].is_multiple_of(r) {
                return Err(CkError::Weights(format!(
                    "unexpected Depatchify deconv shape {w_shape:?} at '{prefix}'"
                )));
            }
            let overlap_mult = w_shape[2] / r;
            Self::from_deconv_slices(ctx, w_shape[0], w_shape[1], r, overlap_mult, &weight, &bias)
        } else {
            let (w_shape, weight) = st.read_as::<F>(&format!("{prefix}.linear.weight"))?;
            let (_, bias) = st.read_as::<F>(&format!("{prefix}.linear.bias"))?;
            if w_shape.len() != 2 || !w_shape[0].is_multiple_of(r * r) {
                return Err(CkError::Weights(format!(
                    "unexpected Depatchify linear shape {w_shape:?} at '{prefix}'"
                )));
            }
            let out_channels = w_shape[0] / (r * r);
            Self::from_linear_slices(ctx, w_shape[1], out_channels, r, &weight, &bias)
        }
    }

    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x_nhwc: Tensor<4>) -> CkResult<Tensor<4>> {
        let [bt, h_in, w_in, c_in] = x_nhwc.dims();
        if c_in != self.in_dim {
            return Err(CkError::Shape(format!(
                "Depatchify expected in_dim={}, got {c_in}",
                self.in_dim
            )));
        }
        let tokens = bt * h_in * w_in;
        let k = self.overlap_mult * self.r;
        let k2_o = k * k * self.out_channels;
        let flat = x_nhwc.reshape([tokens, c_in]);
        let cols_2d = ctx.matmul(flat, self.weight_ik2o.clone());
        let c_cols = into_contiguous_cube(cols_2d)?;
        if let Some(col_b) = &self.col_bias {
            let total_cols = tokens * k2_o;
            let geom = LaunchGeometry::elementwise(ctx.client(), total_cols);
            unsafe {
                add_row_bias_kernel::launch::<F>(
                    ctx.client(),
                    geom.cube_count,
                    geom.cube_dim,
                    BufferArg::from_raw_parts(c_cols.handle.clone(), total_cols),
                    BufferArg::from_raw_parts(col_b.clone(), k2_o),
                    total_cols as u32,
                    k2_o as u32,
                );
            }
        }
        let cols_handle = c_cols.handle;

        let h_out = h_in * self.r;
        let w_out = w_in * self.r;
        let total_out = bt * h_out * w_out * self.out_channels;
        let out_handle = empty::<F>(ctx.client(), total_out);
        let geom = LaunchGeometry::elementwise(ctx.client(), total_out);
        unsafe {
            col2im_nhwc_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(cols_handle, tokens * k2_o),
                BufferArg::from_raw_parts(self.out_bias.clone(), self.out_channels),
                BufferArg::from_raw_parts(out_handle.clone(), total_out),
                total_out as u32,
                h_in as u32,
                w_in as u32,
                self.out_channels as u32,
                self.r as u32,
                self.overlap_mult as u32,
            );
        }
        Ok(ctx.tensor_from_handle(
            out_handle,
            [bt, h_out, w_out, self.out_channels],
            elem_dtype::<F>(),
        ))
    }
}

/// Device-resident `Downsample4` (`corridorkey/blocks.py`): `Conv2d(k=8, s=4, pad=2)` + optional `RmsNorm`.
/// Implemented via `im2col_nhwc_kernel` + tiled GEMM so large channel-count downsampling runs at
/// full Tensor Core speed instead of falling back to scalar direct convolution.
#[derive(Clone, Debug)]
pub struct Downsample4 {
    weight_k2io: Tensor<2>,
    bias: Handle,
    norm: Option<RmsNorm>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
}

impl Downsample4 {
    pub fn from_slices<F: CkFloat>(
        ctx: &MlContext,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        weight_oihw: &[F],
        bias: &[F],
        norm_w: Option<&[F]>,
    ) -> CkResult<Self> {
        let pad = (kernel - stride) / 2;
        let k2_i = kernel * kernel * in_channels;
        let k2io = conv_weight_oihw_to_k2io(weight_oihw, out_channels, in_channels, kernel);
        let norm = match norm_w {
            Some(nw) => Some(RmsNorm::from_slice(ctx, nw, DEFAULT_RMS_NORM_EPS)?),
            None => None,
        };
        Ok(Self {
            weight_k2io: ctx.upload_tensor(&k2io, [k2_i, out_channels])?,
            bias: upload(ctx.client(), bias),
            norm,
            in_channels,
            out_channels,
            kernel,
            stride,
            pad,
        })
    }

    pub fn load<F: CkFloat>(ctx: &MlContext, st: &SafetensorsFile, prefix: &str) -> CkResult<Self> {
        let (w_shape, weight) = st.read_as::<F>(&format!("{prefix}.conv.weight"))?;
        let (_, bias) = st.read_as::<F>(&format!("{prefix}.conv.bias"))?;
        let norm_key = format!("{prefix}.norm.weight");
        let norm_w = if st.entry(&norm_key).is_ok() {
            Some(st.read_as::<F>(&norm_key)?.1)
        } else {
            None
        };
        if w_shape.len() != 4 || w_shape[2] != w_shape[3] {
            return Err(CkError::Weights(format!(
                "unexpected Downsample4 shape {w_shape:?} at '{prefix}'"
            )));
        }
        Self::from_slices(
            ctx,
            w_shape[1],
            w_shape[0],
            w_shape[2],
            4,
            &weight,
            &bias,
            norm_w.as_deref(),
        )
    }

    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x_nhwc: Tensor<4>) -> CkResult<Tensor<4>> {
        let [bt, h, w, c] = x_nhwc.dims();
        if c != self.in_channels {
            return Err(CkError::Shape(format!(
                "Downsample4 expected in_channels={}, got {c}",
                self.in_channels
            )));
        }
        if !h.is_multiple_of(self.stride) || !w.is_multiple_of(self.stride) {
            return Err(CkError::Shape(format!(
                "Downsample4 spatial dims ({h}, {w}) must divide stride={}",
                self.stride
            )));
        }
        let h_out = h / self.stride;
        let w_out = w / self.stride;
        let tokens = bt * h_out * w_out;
        let k2_i = self.kernel * self.kernel * self.in_channels;

        let c_x = into_contiguous_cube(x_nhwc)?;
        let total_cols = tokens * k2_i;
        let cols_h = empty::<F>(ctx.client(), total_cols);
        let geom_col = LaunchGeometry::elementwise(ctx.client(), total_cols);
        unsafe {
            im2col_nhwc_kernel::launch::<F>(
                ctx.client(),
                geom_col.cube_count,
                geom_col.cube_dim,
                BufferArg::from_raw_parts(c_x.handle, bt * h * w * c),
                BufferArg::from_raw_parts(cols_h.clone(), total_cols),
                total_cols as u32,
                h as u32,
                w as u32,
                c as u32,
                h_out as u32,
                w_out as u32,
                self.kernel as u32,
                self.stride as u32,
                self.pad as u32,
            );
        }

        let cols_2d = ctx.tensor_from_handle(cols_h, [tokens, k2_i], elem_dtype::<F>());
        let out_2d = ctx.matmul(cols_2d, self.weight_k2io.clone());
        let c_out = into_contiguous_cube(out_2d)?;
        let total_out = tokens * self.out_channels;
        let geom_out = LaunchGeometry::elementwise(ctx.client(), total_out);
        unsafe {
            add_row_bias_kernel::launch::<F>(
                ctx.client(),
                geom_out.cube_count,
                geom_out.cube_dim,
                BufferArg::from_raw_parts(c_out.handle.clone(), total_out),
                BufferArg::from_raw_parts(self.bias.clone(), self.out_channels),
                total_out as u32,
                self.out_channels as u32,
            );
        }

        let conv_out = ctx.tensor_from_handle(
            c_out.handle,
            [bt, h_out, w_out, self.out_channels],
            elem_dtype::<F>(),
        );
        match &self.norm {
            Some(norm) => norm.forward::<F, 4>(ctx, conv_out),
            None => Ok(conv_out),
        }
    }
}

/// Device-resident `Upsample4ICNR` (`corridorkey/blocks.py`):
/// `1×1 proj + PixelShuffle(4)` + residual `3×3 smooth conv` + optional `RmsNorm`.
/// Smooth convolution runs via `im2col_nhwc_kernel` + tiled GEMM for hardware acceleration.
#[derive(Clone, Debug)]
pub struct Upsample4Icnr {
    proj_shuffle: Depatchify,
    smooth_weight_k2io: Tensor<2>,
    smooth_bias: Handle,
    norm: Option<RmsNorm>,
    out_channels: usize,
}

impl Upsample4Icnr {
    #[allow(clippy::too_many_arguments)]
    pub fn from_slices<F: CkFloat>(
        ctx: &MlContext,
        in_channels: usize,
        out_channels: usize,
        r: usize,
        proj_w_oi: &[F],
        proj_b: &[F],
        smooth_w_oihw: &[F],
        smooth_b: &[F],
        norm_w: Option<&[F]>,
    ) -> CkResult<Self> {
        let proj_shuffle =
            Depatchify::from_linear_slices(ctx, in_channels, out_channels, r, proj_w_oi, proj_b)?;
        let k2_i = 3 * 3 * out_channels;
        let smooth_k2io = conv_weight_oihw_to_k2io(smooth_w_oihw, out_channels, out_channels, 3);
        let norm = match norm_w {
            Some(nw) => Some(RmsNorm::from_slice(ctx, nw, DEFAULT_RMS_NORM_EPS)?),
            None => None,
        };
        Ok(Self {
            proj_shuffle,
            smooth_weight_k2io: ctx.upload_tensor(&smooth_k2io, [k2_i, out_channels])?,
            smooth_bias: upload(ctx.client(), smooth_b),
            norm,
            out_channels,
        })
    }

    pub fn load<F: CkFloat>(
        ctx: &MlContext,
        st: &SafetensorsFile,
        prefix: &str,
        r: usize,
    ) -> CkResult<Self> {
        let (p_shape, proj_w) = st.read_as::<F>(&format!("{prefix}.proj.weight"))?;
        let (_, proj_b) = st.read_as::<F>(&format!("{prefix}.proj.bias"))?;
        let (s_shape, smooth_w) = st.read_as::<F>(&format!("{prefix}.smooth.weight"))?;
        let (_, smooth_b) = st.read_as::<F>(&format!("{prefix}.smooth.bias"))?;
        let norm_key = format!("{prefix}.norm.weight");
        let norm_w = if st.entry(&norm_key).is_ok() {
            Some(st.read_as::<F>(&norm_key)?.1)
        } else {
            None
        };
        if p_shape.len() != 4 || s_shape.len() != 4 {
            return Err(CkError::Weights(format!(
                "unexpected Upsample4ICNR shapes proj={p_shape:?} smooth={s_shape:?} at '{prefix}'"
            )));
        }
        Self::from_slices(
            ctx,
            p_shape[1],
            s_shape[0],
            r,
            &proj_w,
            &proj_b,
            &smooth_w,
            &smooth_b,
            norm_w.as_deref(),
        )
    }

    pub fn forward<F: CkFloat>(&self, ctx: &MlContext, x_nhwc: Tensor<4>) -> CkResult<Tensor<4>> {
        let up = self.proj_shuffle.forward::<F>(ctx, x_nhwc)?;
        let [bt, h, w, c] = up.dims();
        let tokens = bt * h * w;
        let k2_i = 3 * 3 * self.out_channels;

        let c_up = into_contiguous_cube(up.clone())?;
        let total_cols = tokens * k2_i;
        let cols_h = empty::<F>(ctx.client(), total_cols);
        let geom_col = LaunchGeometry::elementwise(ctx.client(), total_cols);
        unsafe {
            im2col_nhwc_kernel::launch::<F>(
                ctx.client(),
                geom_col.cube_count,
                geom_col.cube_dim,
                BufferArg::from_raw_parts(c_up.handle, bt * h * w * c),
                BufferArg::from_raw_parts(cols_h.clone(), total_cols),
                total_cols as u32,
                h as u32,
                w as u32,
                c as u32,
                h as u32,
                w as u32,
                3u32,
                1u32,
                1u32,
            );
        }

        let cols_2d = ctx.tensor_from_handle(cols_h, [tokens, k2_i], elem_dtype::<F>());
        let out_2d = ctx.matmul(cols_2d, self.smooth_weight_k2io.clone());
        let c_out = into_contiguous_cube(out_2d)?;
        let total_out = tokens * self.out_channels;
        let geom_out = LaunchGeometry::elementwise(ctx.client(), total_out);
        unsafe {
            add_row_bias_kernel::launch::<F>(
                ctx.client(),
                geom_out.cube_count,
                geom_out.cube_dim,
                BufferArg::from_raw_parts(c_out.handle.clone(), total_out),
                BufferArg::from_raw_parts(self.smooth_bias.clone(), self.out_channels),
                total_out as u32,
                self.out_channels as u32,
            );
        }

        let smooth = ctx.tensor_from_handle(
            c_out.handle,
            [bt, h, w, self.out_channels],
            elem_dtype::<F>(),
        );
        let sum = up + smooth;
        match &self.norm {
            Some(norm) => norm.forward::<F, 4>(ctx, sum),
            None => Ok(sum),
        }
    }
}

/// Device-resident late RGB skip gate (`patch_2` + `patch_2_gate` in `Stage2`).
#[derive(Clone, Debug)]
pub struct GatedRgbSkip {
    patch_2: PatchifyConv,
    gate_w_t: Tensor<2>,
    gate_b: Handle,
    d: usize,
    d_baton: usize,
}

impl GatedRgbSkip {
    pub fn from_slices<F: CkFloat>(
        ctx: &MlContext,
        patch_2: PatchifyConv,
        d: usize,
        d_baton: usize,
        gate_w: &[F],
        gate_b: &[F],
    ) -> CkResult<Self> {
        let d_rgb = d - d_baton;
        if gate_w.len() != d_rgb * d || gate_b.len() != d_rgb {
            return Err(CkError::Shape(
                "GatedRgbSkip gate weight/bias shape mismatch".into(),
            ));
        }
        Ok(Self {
            patch_2,
            gate_w_t: ctx.upload_transposed(gate_w, d_rgb, d)?,
            gate_b: upload(ctx.client(), gate_b),
            d,
            d_baton,
        })
    }

    pub fn load<F: CkFloat>(
        ctx: &MlContext,
        st: &SafetensorsFile,
        prefix: &str,
        d: usize,
        d_baton: usize,
        patch: usize,
    ) -> CkResult<Self> {
        let patch_2 = PatchifyConv::load::<F>(ctx, st, &format!("{prefix}.patch_2"), patch)?;
        let (_, gate_w) = st.read_as::<F>(&format!("{prefix}.patch_2_gate.weight"))?;
        let (_, gate_b) = st.read_as::<F>(&format!("{prefix}.patch_2_gate.bias"))?;
        Self::from_slices(ctx, patch_2, d, d_baton, &gate_w, &gate_b)
    }

    pub fn forward<F: CkFloat>(
        &self,
        ctx: &MlContext,
        x_nhwc: Tensor<4>,
        rgb_nhwc: Tensor<4>,
    ) -> CkResult<Tensor<4>> {
        let [bt, h, w, c] = x_nhwc.dims();
        if c != self.d {
            return Err(CkError::Shape(format!(
                "GatedRgbSkip expected d={}, got {c}",
                self.d
            )));
        }
        let d_rgb = self.d - self.d_baton;
        let tokens = bt * h * w;
        let patch2 = self.patch_2.forward::<F>(ctx, rgb_nhwc)?;
        let gate_logits = ctx.matmul(
            x_nhwc.clone().reshape([tokens, self.d]),
            self.gate_w_t.clone(),
        );

        let c_x = into_contiguous_cube(x_nhwc)?;
        let c_p2 = into_contiguous_cube(patch2)?;
        let c_gl = into_contiguous_cube(gate_logits)?;
        let total = tokens * self.d;
        let out = empty::<F>(ctx.client(), total);
        let geom = LaunchGeometry::elementwise(ctx.client(), total);
        unsafe {
            add_gated_rgb_skip_kernel::launch::<F>(
                ctx.client(),
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(c_x.handle, total),
                BufferArg::from_raw_parts(c_p2.handle, tokens * d_rgb),
                BufferArg::from_raw_parts(c_gl.handle, tokens * d_rgb),
                BufferArg::from_raw_parts(self.gate_b.clone(), d_rgb),
                BufferArg::from_raw_parts(out.clone(), total),
                total as u32,
                self.d as u32,
                self.d_baton as u32,
            );
        }
        Ok(ctx.tensor_from_handle(out, [bt, h, w, self.d], elem_dtype::<F>()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use corridorkey_core::ComputeTarget;

    #[test]
    fn depatchify_overlap_and_linear_match_f64_reference() {
        let (bt, h_in, w_in, c_in, c_out, r, overlap_mult) = (1, 3, 4, 5, 4, 2, 2);
        let k = overlap_mult * r;
        let pad = (k - r) / 2;
        let h_out = h_in * r;
        let w_out = w_in * r;

        let make_vec = |n: usize, s: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32) * s + 0.1).sin() * 0.3).collect()
        };
        let x = make_vec(bt * h_in * w_in * c_in, 0.13);
        let w_iohw = make_vec(c_in * c_out * k * k, 0.07);
        let bias = make_vec(c_out, 0.19);

        // Host f64 ConvTranspose2d in NHWC
        let mut want = vec![0.0f64; bt * h_out * w_out * c_out];
        for b in 0..bt {
            for oy in 0..h_out {
                for ox in 0..w_out {
                    for co in 0..c_out {
                        want[((b * h_out + oy) * w_out + ox) * c_out + co] = bias[co] as f64;
                    }
                }
            }
            for iy in 0..h_in {
                for ix in 0..w_in {
                    for ci in 0..c_in {
                        let xv = x[((b * h_in + iy) * w_in + ix) * c_in + ci] as f64;
                        for ky in 0..k {
                            let oy_s = (iy * r + ky) as isize - (pad as isize);
                            if oy_s < 0 || oy_s >= (h_out as isize) {
                                continue;
                            }
                            let oy = oy_s as usize;
                            for kx in 0..k {
                                let ox_s = (ix * r + kx) as isize - (pad as isize);
                                if ox_s < 0 || ox_s >= (w_out as isize) {
                                    continue;
                                }
                                let ox = ox_s as usize;
                                for co in 0..c_out {
                                    let wv = w_iohw[((ci * c_out + co) * k + ky) * k + kx] as f64;
                                    want[((b * h_out + oy) * w_out + ox) * c_out + co] += xv * wv;
                                }
                            }
                        }
                    }
                }
            }
        }

        for target in ComputeTarget::available() {
            let ctx = MlContext::new(target).unwrap();
            let dep =
                Depatchify::from_deconv_slices(&ctx, c_in, c_out, r, overlap_mult, &w_iohw, &bias)
                    .unwrap();
            let t_in = ctx.upload_tensor(&x, [bt, h_in, w_in, c_in]).unwrap();
            let t_out = dep.forward::<f32>(&ctx, t_in).unwrap();
            let got = ctx.download_tensor::<f32, 4>(t_out).unwrap();
            for (i, (&g, &ref_v)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g as f64 - ref_v).abs() < 1e-5,
                    "{target} deconv [{i}]: {g} vs {ref_v}"
                );
            }
        }
    }

    #[test]
    fn patchify_conv_matches_f64_reference() {
        let (bt, h_in, w_in, c_in, c_out, patch, overlap_mult) = (1, 8, 12, 3, 6, 4, 2);
        let k = overlap_mult * patch;
        let pad = (k - patch) / 2;
        let h_out = h_in / patch;
        let w_out = w_in / patch;

        let make_vec = |n: usize, s: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32) * s - 0.2).cos() * 0.25)
                .collect()
        };
        let x = make_vec(bt * h_in * w_in * c_in, 0.11);
        let w_oihw = make_vec(c_out * c_in * k * k, 0.05);
        let bias = make_vec(c_out, 0.17);

        let mut want = vec![0.0f64; bt * h_out * w_out * c_out];
        for b in 0..bt {
            for oy in 0..h_out {
                for ox in 0..w_out {
                    for co in 0..c_out {
                        let mut acc = bias[co] as f64;
                        for ky in 0..k {
                            let iy_s = (oy * patch + ky) as isize - (pad as isize);
                            if iy_s < 0 || iy_s >= (h_in as isize) {
                                continue;
                            }
                            let iy = iy_s as usize;
                            for kx in 0..k {
                                let ix_s = (ox * patch + kx) as isize - (pad as isize);
                                if ix_s < 0 || ix_s >= (w_in as isize) {
                                    continue;
                                }
                                let ix = ix_s as usize;
                                for ci in 0..c_in {
                                    let xv = x[((b * h_in + iy) * w_in + ix) * c_in + ci] as f64;
                                    let wv = w_oihw[((co * c_in + ci) * k + ky) * k + kx] as f64;
                                    acc += xv * wv;
                                }
                            }
                        }
                        want[((b * h_out + oy) * w_out + ox) * c_out + co] = acc;
                    }
                }
            }
        }

        for target in ComputeTarget::available() {
            let ctx = MlContext::new(target).unwrap();
            let patchify =
                PatchifyConv::from_slices(&ctx, c_in, c_out, patch, overlap_mult, &w_oihw, &bias)
                    .unwrap();
            let t_in = ctx.upload_tensor(&x, [bt, h_in, w_in, c_in]).unwrap();
            let t_out = patchify.forward::<f32>(&ctx, t_in).unwrap();
            let got = ctx.download_tensor::<f32, 4>(t_out).unwrap();
            for (i, (&g, &ref_v)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g as f64 - ref_v).abs() < 2e-5,
                    "{target} patchify [{i}]: {g} vs {ref_v}"
                );
            }
        }
    }
}
