//! Core transformer building blocks (`corridorkey/blocks.py`): channels-last [`RmsNorm`],
//! [`SwiGlu`], [`QkvDepthwiseSpatial`] and [`DepthwiseSwiGlu2d`].

pub mod attn_core;
pub mod blocks_2d;
pub mod blocks_3d;
pub mod fusion;
pub mod kernels;
pub mod local_mix;
pub mod mlp;
pub mod norm;
pub mod patch;

pub use attn_core::{AttentionCore, ShiftedRegionGroups};
pub use blocks_2d::{
    CSwinAttention2d, CSwinBlock2d, FullSpatialAttention2d, FullSpatialBlock2d, Stage2Block2d,
    SwinBlock2d,
};
pub use blocks_3d::{
    CSwinBlock3d, FullSpatialBlock3d, FullSpatioTemporalBlock, Stage1DecBlock3d, SwinBlock3d,
};
pub use fusion::{AdaLnZero, FusionLinSwiglu};
pub use local_mix::{DepthwiseSwiGlu2d, QkvDepthwiseSpatial, depthwise_weight_to_tap_major};
pub use mlp::{SwiGlu, transpose_2d};
pub use norm::{DEFAULT_RMS_NORM_EPS, RmsNorm, rms_norm};
pub use patch::{
    Depatchify, Downsample4, GatedRgbSkip, PatchifyConv, Upsample4Icnr, conv_weight_oihw_to_k2io,
    conv_weight_oihw_to_ohwi, deconv_weight_iohw_to_ik2o, pixel_shuffle_linear_to_ir2o,
};
