//! `corridorkey-ml`: the CorridorKey networks on Burn and CubeCL.
//!
//! - [`provenance`]: paper, code, license and weights attribution (`Attributed`).
//! - [`hub`]: `.safetensors` header/offset reader, `ModelConfig` and `FoundationStats`.
//! - [`runtime`]: zero-copy bridge between `ComputeTarget`, CubeCL `Handle`s and Burn `Tensor`s.
//! - [`models`]: rotary embeddings (`rope`), transformer building blocks (`blocks`), stages and the
//!   frozen backbones (`foundations`).
//! - [`pipeline`]: keying a clip on the device (`Keyer`), decoder / encoder behind traits.

pub mod hub;
pub mod models;
pub mod pipeline;
pub mod provenance;
pub mod runtime;

pub use hub::{
    CHECKPOINT_FORMAT, CheckpointHeader, CheckpointMetadata, FoundationStats, ModelConfig,
    SafetensorsFile, SafetensorsMap, TensorEntry,
};
pub use models::{
    blocks::{
        AdaLnZero, AttentionCore, CSwinAttention2d, CSwinBlock2d, CSwinBlock3d,
        DEFAULT_RMS_NORM_EPS, Depatchify, DepthwiseSwiGlu2d, Downsample4, FullSpatialAttention2d,
        FullSpatialBlock2d, FullSpatialBlock3d, FullSpatioTemporalBlock, FusionLinSwiglu,
        GatedRgbSkip, PatchifyConv, QkvDepthwiseSpatial, RmsNorm, ShiftedRegionGroups,
        Stage1DecBlock3d, Stage2Block2d, SwiGlu, SwinBlock2d, SwinBlock3d, Upsample4Icnr, rms_norm,
        transpose_2d,
    },
    foundations::{CRadio, MogeEncoder, RvmSmall},
    rope::{DEFAULT_ROPE_BASE, Rope2d, Rope3d, apply_rotary},
    stage1::{STAGE1_PATCH_SIZE, STAGE1_SHAPE_QUANTUM, Stage1, Stage1Output},
    stage2::{STAGE2_OUT_CHANNELS, STAGE2_PATCH_SIZE, Stage2},
};
pub use provenance::{
    ArtifactSource, Attributed, Paper, Provenance, ProvenanceKind, UpstreamCode,
    corridorkey_v2_provenance, cradio_v4_provenance, moge2_vitb_provenance, rvm_small_provenance,
};
pub use runtime::{MlContext, elem_dtype, from_cube, into_contiguous_cube, into_cube};
