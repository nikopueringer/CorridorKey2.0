//! Typed configuration and metadata parsed from the header of `corridorkey_v2.safetensors`.
//!
//! Matches `CorridorKeyV3.__init__` in `corridorkey/model.py` and the header written by
//! `scripts/convert_to_safetensors.py`.

use std::collections::BTreeMap;

use corridorkey_core::{CkError, CkResult};
use serde::{Deserialize, Serialize};

/// Expected `format` tag in the `.safetensors` header.
pub const CHECKPOINT_FORMAT: &str = "corridorkey_v2_runtime_checkpoint";

/// Architecture parameters serialized in the `model_config` header entry of
/// `corridorkey_v2.safetensors`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub d_16: usize,
    pub d_64: usize,
    pub d_stage2: usize,
    pub n_enc_16: usize,
    pub n_body_64: usize,
    pub n_dec_16: usize,
    pub num_heads_16: usize,
    pub num_heads_64: usize,
    pub num_heads_stage2: usize,
    #[serde(rename = "window_HW")]
    pub window_hw: usize,
    pub window_size_stage2: usize,
    pub stage1_local_mix_kernel: usize,
    pub stage1_local_mix_init_std: Option<f64>,
    pub stage1_body64_local_mix_kernel: usize,
    pub stage1_body64_local_mix_init_std: Option<f64>,
    pub stage1_body64_mlp_hidden: Option<usize>,
    pub stage1_body64_direct_concat: bool,
    #[serde(default)]
    pub stage1_recurrent_64_sites: Vec<usize>,
    pub stage1_hint_adaln_blocks: usize,
    pub stage1_mid_16_blocks: usize,
    pub n_blocks_stage2: usize,
    pub n_final_blocks_stage2: usize,
    pub stage2_local_mix_kernel: usize,
    pub stage2_local_mix_init_std: Option<f64>,
    #[serde(default)]
    pub refine4_blocks: usize,
    #[serde(default = "default_refine4_width")]
    pub refine4_width: usize,
    #[serde(default = "default_refine4_token_channels")]
    pub refine4_token_channels: usize,
    #[serde(default = "default_refine4_kernel")]
    pub refine4_kernel: usize,
    #[serde(default = "default_refine4_expansion")]
    pub refine4_expansion: f64,
    pub mlp_mult: f64,
    pub mlp_dw_kernel: usize,
    pub qkv_dw_kernel: usize,
    pub qkv_dw_fraction: f64,
    pub use_moge: bool,
    pub use_cradio: bool,
    pub cradio_repo: String,
    pub moge_checkpoint: String,
    pub rvm_variant: String,
    pub rvm_ckpt: Option<String>,
    #[serde(default)]
    pub gradient_checkpoint: bool,
    pub overlap_patchify: bool,
    pub overlap_mult: usize,
    pub depatch_fourier_features: usize,
    pub depatch_fourier_kernel: usize,
    pub enable_mae_mask_tokens: bool,
    pub bottleneck_head_stride: Option<usize>,
    pub rope_shift_coords: Option<f64>,
    pub rope_jitter_coords: Option<f64>,
    pub rope_rescale_coords: Option<f64>,
    pub rope_base: f64,
    pub foundation_stats_path: Option<String>,
}

fn default_refine4_width() -> usize {
    96
}
fn default_refine4_token_channels() -> usize {
    48
}
fn default_refine4_kernel() -> usize {
    7
}
fn default_refine4_expansion() -> f64 {
    2.0
}

impl ModelConfig {
    /// Per-head dimension at Stage 1's `/16` resolution (`d_16 / num_heads_16`).
    pub fn head_dim_16(&self) -> usize {
        self.d_16 / self.num_heads_16
    }

    /// Per-head dimension at Stage 1's `/64` bottleneck (`d_64 / num_heads_64`).
    pub fn head_dim_64(&self) -> usize {
        self.d_64 / self.num_heads_64
    }

    /// Per-head dimension in Stage 2 (`d_stage2 / num_heads_stage2`).
    pub fn head_dim_stage2(&self) -> usize {
        self.d_stage2 / self.num_heads_stage2
    }

    /// SwiGLU hidden width for a given embedding dimension (`round(dim * mlp_mult)`).
    pub fn mlp_hidden(&self, dim: usize) -> usize {
        (dim as f64 * self.mlp_mult).round() as usize
    }

    /// Effective overlap multiplier for patchify/depatchify (`overlap_patchify=true` forces 3).
    pub fn effective_overlap_mult(&self) -> usize {
        if self.overlap_patchify {
            3
        } else {
            self.overlap_mult.max(1)
        }
    }

    /// Number of pre-patch2 blocks in Stage 2 (`n_blocks_stage2 - n_final_blocks_stage2`).
    pub fn stage2_num_pre_blocks(&self) -> usize {
        self.n_blocks_stage2.saturating_sub(self.n_final_blocks_stage2)
    }

    /// QKV depthwise kernel size per Stage 2 block (`_stage2_qkv_kernels` in `corridorkey/stage2.py`).
    pub fn stage2_qkv_kernels(&self) -> Vec<usize> {
        let total = self.n_blocks_stage2;
        let kernel = self.qkv_dw_kernel;
        let frac = self.qkv_dw_fraction.clamp(0.0, 1.0);
        if total == 0 || kernel == 0 || frac <= 0.0 {
            return vec![0; total];
        }
        if total != 10 {
            let active = if frac >= 1.0 {
                total
            } else {
                ((total as f64) * frac).round() as usize
            };
            return (0..total).map(|i| if i < active { kernel } else { 0 }).collect();
        }
        let active = if (frac - 0.25).abs() < 1e-12 {
            4
        } else if frac >= 1.0 {
            total
        } else {
            ((total as f64) * frac).round() as usize
        };
        let priority = [1, 3, 6, 9, 7, 8, 0, 4, 2, 5];
        let mut plan = vec![0; total];
        for &idx in priority.iter().take(active.min(total)) {
            plan[idx] = kernel;
        }
        plan
    }
}

/// Public runtime metadata serialized in the `metadata` header entry of
/// `corridorkey_v2.safetensors`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub public_model_name: String,
    pub pre_recurrence: bool,
    pub shape_quantum: usize,
    #[serde(default)]
    pub required_vfms: Vec<String>,
}

/// Parsed `__metadata__` section of a CorridorKey v2 `.safetensors` checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointHeader {
    pub format: String,
    pub format_version: u32,
    pub model_config: ModelConfig,
    pub metadata: CheckpointMetadata,
}

impl CheckpointHeader {
    /// Parses and validates a `CheckpointHeader` from the raw string map in `__metadata__`.
    pub fn from_raw_metadata(raw: &BTreeMap<String, String>) -> CkResult<Self> {
        let format = raw
            .get("format")
            .ok_or_else(|| CkError::Weights("missing 'format' in safetensors header".into()))?
            .clone();
        if format != CHECKPOINT_FORMAT {
            return Err(CkError::Weights(format!(
                "unexpected checkpoint format {format:?}, expected {CHECKPOINT_FORMAT:?}"
            )));
        }
        let format_version = raw
            .get("format_version")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(1);
        let config_json = raw
            .get("model_config")
            .ok_or_else(|| CkError::Weights("missing 'model_config' in safetensors header".into()))?;
        let model_config: ModelConfig = serde_json::from_str(config_json)
            .map_err(|e| CkError::Weights(format!("invalid 'model_config' JSON: {e}")))?;
        let meta_json = raw
            .get("metadata")
            .ok_or_else(|| CkError::Weights("missing 'metadata' in safetensors header".into()))?;
        let metadata: CheckpointMetadata = serde_json::from_str(meta_json)
            .map_err(|e| CkError::Weights(format!("invalid 'metadata' JSON: {e}")))?;

        Ok(Self {
            format,
            format_version,
            model_config,
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use crate::hub::{FoundationStats, SafetensorsFile};
    use crate::models::blocks::{DepthwiseSwiGlu2d, RmsNorm, SwiGlu, DEFAULT_RMS_NORM_EPS};
    use crate::runtime::MlContext;
    use corridorkey_core::ComputeTarget;

    #[test]
    fn loads_real_checkpoint_header_and_executes_stage2_blocks_when_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
        let ckpt_path = root.join("weights/corridorkey_v2.safetensors");
        let stats_path = root.join("weights/foundation_stats.safetensors");
        if !ckpt_path.exists() || !stats_path.exists() {
            return;
        }
        let st = SafetensorsFile::open(&ckpt_path).unwrap();
        let hdr = st.checkpoint_header().unwrap();
        assert_eq!(hdr.format, CHECKPOINT_FORMAT);
        assert_eq!(hdr.model_config.d_16, 512);
        assert_eq!(hdr.model_config.d_64, 1536);
        assert_eq!(hdr.model_config.d_stage2, 1280);
        assert_eq!(hdr.model_config.stage2_qkv_kernels(), vec![0, 7, 0, 7, 0, 0, 7, 0, 0, 7]);

        let stats = FoundationStats::from_file(&stats_path).unwrap();
        assert!(stats.moge_dim() > 0 && stats.cradio_dim() > 0 && stats.rvm_dim() > 0);

        let target = ComputeTarget::from_env().unwrap();
        let ctx = MlContext::new(target).unwrap();
        let norm1 = RmsNorm::load::<f32>(&ctx, &st, "stage2.final_blocks.0.norm1", DEFAULT_RMS_NORM_EPS).unwrap();
        let local_mix = DepthwiseSwiGlu2d::load::<f32>(&ctx, &st, "stage2.final_blocks.0.local_mix").unwrap();
        let mlp = SwiGlu::load::<f32>(&ctx, &st, "stage2.final_blocks.0.mlp").unwrap();
        assert_eq!(norm1.dim(), 1280);
        assert_eq!(mlp.dim(), 1280);
        assert_eq!(mlp.hidden(), 3200);

        let (b, h, w, c) = (1, 4, 4, 1280);
        let input: Vec<f32> = (0..b * h * w * c).map(|i| ((i as f32) * 0.01).sin() * 0.2).collect();
        let x = ctx.upload_tensor(&input, [b, h, w, c]).unwrap();
        let y = norm1.forward::<f32, 4>(&ctx, x).unwrap();
        let y = local_mix.forward_residual::<f32, 4>(&ctx, y).unwrap();
        let y = mlp.forward::<f32, 4>(&ctx, y).unwrap();
        let host = ctx.download_tensor::<f32, 4>(y).unwrap();
        assert_eq!(host.len(), b * h * w * c);
        assert!(host.iter().all(|v| v.is_finite()));
    }
}
