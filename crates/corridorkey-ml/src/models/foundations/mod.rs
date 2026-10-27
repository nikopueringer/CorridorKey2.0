//! The frozen foundation backbones whose features feed Stage 1 (`corridorkey/foundations.py`):
//! MoGe-2 ViT-B, RVM-small, C-RADIOv4-SO400M; shared plain-ViT pieces in [`vit`].

pub mod cradio;
pub mod moge;
pub mod rvm;
pub mod vit;

pub use cradio::{CRADIO_PATCH, CRadio};
pub use moge::{MOGE_PATCH, MogeEncoder};
pub use rvm::{RVM_PATCH, RvmSmall};
pub use vit::{
    Attention, LayerNorm, Linear, Mlp, TORCH_LAYER_NORM_EPS, VIT_LAYER_NORM_EPS, VitBlock, ones,
    residual_add, resize_pos_embed_bicubic, resize_table_bilinear,
};
