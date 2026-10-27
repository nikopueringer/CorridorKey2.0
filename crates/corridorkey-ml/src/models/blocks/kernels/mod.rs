//! CubeCL kernels for transformer building blocks (RMSNorm, SwiGLU gating, depthwise spatial
//! filters, bias addition).

pub mod adaln;
pub mod attn;
pub mod attn_3d;
pub mod depthwise;
pub mod flash;
pub mod norm;
pub mod patch;
pub mod recurrent;
pub mod sdpa;
pub mod swiglu;
pub mod vit;

pub use adaln::{adaln_gate_residual_kernel, adaln_modulate_kernel, silu_bias_kernel};
pub use attn::{
    gather_window_tokens_kernel, merge_cswin2d_gate_kernel, merge_heads_gate_kernel,
    pad_roll_2d_kernel, prep_cswin2d_qkv_kernel, prep_qkv_norm_rope_kernel,
    scatter_window_tokens_kernel, sigmoid, window_partition_2d_kernel,
    window_unpartition_unroll_residual_2d_kernel, zero_buffer_kernel,
};
pub use attn_3d::{
    add_scaled_skip_kernel, merge_cswin3d_gate_kernel, prep_cswin3d_qkv_kernel,
    window_partition_3d_kernel, window_unpartition_3d_kernel,
};
pub use depthwise::{boundary_coord, depthwise_conv2d_nhwc_kernel, depthwise_swiglu_2d_kernel};
pub use flash::{SDPA_ROWS_HEAD_DIMS, launch_sdpa_rows, sdpa_rows_key_tile, sdpa_rows_kernel};
pub use norm::rms_norm_kernel;
pub use patch::{add_gated_rgb_skip_kernel, col2im_nhwc_kernel, im2col_nhwc_kernel};
pub use sdpa::{SDPA_K_TILE, SDPA_MAX_HEAD_DIM, launch_sdpa_window, sdpa_window_kernel};
pub use swiglu::{
    add_row_bias_kernel, add_row_bias_residual_kernel, silu, swiglu_gate_fused_kernel,
    swiglu_gate_kernel,
};
pub use vit::{
    assemble_tokens_kernel, bias_gelu_kernel, drop_prefix_normalize_kernel, bias_layerscale_residual_kernel, layer_norm_kernel,
    merge_heads_kernel, split_heads_kernel,
};
pub use recurrent::{copy_range_kernel, gru_gates_kernel, state_update_kernel};
