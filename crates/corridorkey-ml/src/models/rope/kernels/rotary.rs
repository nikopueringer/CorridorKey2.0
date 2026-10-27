//! CubeCL kernel for interleaved 2D/3D Rotary Position Embeddings (`apply_rotary` in
//! `corridorkey/rope.py`).
//!
//! Each unit handles one `(x[2p], x[2p + 1])` pair across `[batches, n_tokens, head_dim]`,
//! indexing the `[n_tokens, head_dim / 2]` `cos` and `sin` tables by `p % (n_tokens * half_dim)`
//! without allocating intermediate strided `0::2` / `1::2` views.

use cubecl::prelude::*;

use corridorkey_base::core::CkFloat;

/// Rotates interleaved pairs `(x[2p], x[2p + 1])` by `(cos[t], sin[t])` where
/// `t = p % table_pairs` and `table_pairs = n_tokens * (head_dim / 2)`.
/// `input` and `output` may alias for in-place execution.
#[cube(launch)]
pub fn apply_rotary_kernel<F: CkFloat>(
    input: &[F],
    cos: &[F],
    sin: &[F],
    output: &mut [F],
    total_pairs: u32,
    table_pairs: u32,
) {
    let p = ABSOLUTE_POS as u32;
    if p < total_pairs {
        let t = (p % table_pairs) as usize;
        let idx0 = (p * 2) as usize;
        let idx1 = idx0 + 1;
        let x1 = input[idx0];
        let x2 = input[idx1];
        let c = cos[t];
        let s = sin[t];
        output[idx0] = x1 * c - x2 * s;
        output[idx1] = x1 * s + x2 * c;
    }
}
