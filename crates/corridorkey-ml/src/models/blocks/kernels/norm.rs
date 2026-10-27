//! CubeCL kernel for channels-last RMSNorm (`RMSNorm` and `LayerNorm2d` in `corridorkey/blocks.py`).
//!
//! Because tensors stay in channels-last `[..., C]` order throughout the Rust pipeline, both
//! `RMSNorm` and `LayerNorm2d` normalize contiguous rows of length `dim`.

use cubecl::prelude::*;

use corridorkey_base::core::CkFloat;
use corridorkey_core::compute::row_position;

/// Normalizes each of `num_rows` contiguous rows of width `dim`:
/// `output[r, c] = input[r, c] * rsqrt(mean_c(input[r, c]²) + eps) * weight[c]`.
///
/// One cube per row: its `block_size` units stride over the channels (neighbouring units read
/// neighbouring addresses), sum squares in f32 whatever `F` is, and reduce through shared memory.
/// `block_size` is a power of two. `input` and `output` must not alias.
#[cube(launch)]
pub fn rms_norm_kernel<F: CkFloat>(
    input: &[F],
    weight: &[F],
    output: &mut [F],
    num_rows: u32,
    dim: u32,
    eps: f32,
    #[comptime] block_size: u32,
) {
    // Shared memory is declared outside any branch (the CUDA LLVM lowering rejects it inside one).
    let mut partial = Shared::<[f32]>::new_slice(block_size as usize);
    let row = row_position();
    if row < num_rows {
        let tid = UNIT_POS_X;
        let base = row * dim;

        let mut sum_sq = 0.0f32;
        let mut c = tid;
        while c < dim {
            let v = f32::cast_from(input[(base + c) as usize]);
            sum_sq += v * v;
            c += block_size;
        }
        partial[tid as usize] = sum_sq;
        sync_cube();

        // From the runtime `CUBE_DIM_X` (= block_size): a `let mut` initialised from the comptime
        // `block_size` is itself comptime and cannot take the runtime halving (dsp-kitchen §6b).
        let mut stride = CUBE_DIM_X / 2u32;
        while stride > 0u32 {
            if tid < stride {
                let sum = partial[tid as usize] + partial[(tid + stride) as usize];
                partial[tid as usize] = sum;
            }
            sync_cube();
            stride /= 2u32;
        }

        let inv_rms = 1.0f32 / f32::sqrt(partial[0] / f32::cast_from(dim) + eps);
        let mut co = tid;
        while co < dim {
            let i = (base + co) as usize;
            let v = f32::cast_from(input[i]) * inv_rms * f32::cast_from(weight[co as usize]);
            output[i] = F::cast_from(v);
            co += block_size;
        }
    }
}
