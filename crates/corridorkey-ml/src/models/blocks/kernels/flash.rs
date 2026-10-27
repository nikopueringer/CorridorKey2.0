//! Region-masked flash attention for long sequences (`sdpa_rows_kernel`): Stage 1 3D windows
//! (`T · 256` tokens), Stage 2 shifted 2D windows, CSwin strips and full attention.
//!
//! One unit per query row: the row's `q` and its output accumulator stay in f32 registers
//! (`head_dim` is comptime, loops unrolled), keys and values stream through shared memory in tiles
//! of `key_tile` tokens that every unit of the cube reads (broadcast reads), and the softmax runs
//! online in f32 whatever `F` is. Shifted-window masks are evaluated in kernel from per-token
//! region ids (`-1` = padding), so no `N × N` mask is ever stored.
//!
//! Compared with [`super::sdpa::sdpa_window_kernel`] (8 queries per cube, softmax state in `F`):
//! each K/V tile serves a whole cube of queries instead of 8, and no reduction runs in `F`.

use cubecl::{prelude::*, server::Handle};

use corridorkey_base::core::{CkFloat, buffer::empty};
use corridorkey_core::{
    CkError, CkResult,
    compute::{LaunchGeometry, negative_infinity, row_position},
};

/// Head sizes [`sdpa_rows_kernel`] is compiled for (`head_dim` is a comptime parameter so the
/// per-row registers stay registers): RVM's recurrent core (48), Stage 1 `/16`, DINOv2 and RVM's
/// encoder (64), C-RADIO SO400M (72), Stage 2 (80), Stage 1 `/64` (96).
pub const SDPA_ROWS_HEAD_DIMS: [usize; 5] = [48, 64, 72, 80, 96];

/// Attention over `[groups, n_tokens, head_dim]` (`groups` = windows × heads), queries
/// `q_base .. q_base + CUBE_DIM_X` of group `g` per cube, cubes ordered `(g, query tile)`.
///
/// `region_ids` (`[num_windows, n_tokens]`, window = `g / num_heads % num_windows`) is read when
/// `use_regions != 0`: a key is visible to a query when both ids are equal and `>= 0`; queries with
/// a negative id write zeros.
#[cube(launch)]
pub fn sdpa_rows_kernel<F: CkFloat>(
    q: &[F],
    k: &[F],
    v: &[F],
    region_ids: &[i32],
    out: &mut [F],
    num_groups: u32,
    q_tiles: u32,
    num_heads: u32,
    num_windows: u32,
    n_tokens: u32,
    use_regions: u32,
    scale: f32,
    #[comptime] head_dim: usize,
    #[comptime] key_tile: usize,
) {
    // Shared memory first, outside any branch (CUDA LLVM lowering).
    let mut s_k = Shared::<[F]>::new_slice(key_tile * head_dim);
    let mut s_v = Shared::<[F]>::new_slice(key_tile * head_dim);
    let mut s_r = Shared::<[i32]>::new_slice(key_tile);

    let cube = row_position();
    let group = cube / q_tiles;
    let in_grid = group < num_groups;
    let query = (cube % q_tiles) * CUBE_DIM_X + UNIT_POS_X;
    let active = in_grid && query < n_tokens;
    let base = group * n_tokens * head_dim as u32;
    let region_base = (group / num_heads % num_windows) * n_tokens;

    let mut q_row = Array::<f32>::new(head_dim);
    let mut acc = Array::<f32>::new(head_dim);
    // Index 0 for inactive units: reads stay in bounds even if both arms are evaluated.
    let q_off = if active { base + query * head_dim as u32 } else { 0u32 };
    #[unroll]
    for d in 0..head_dim {
        let qv = f32::cast_from(q[(q_off + d as u32) as usize]);
        q_row[d] = if active { qv * scale } else { 0.0f32 };
        acc[d] = 0.0f32;
    }
    // `if` expressions, not a `let mut` from a literal reassigned at run time (dsp-kitchen §6b).
    let rq_idx = if active && use_regions != 0u32 { region_base + query } else { 0u32 };
    let rq_read = region_ids[rq_idx as usize];
    let rq = if active { if use_regions != 0u32 { rq_read } else { 0i32 } } else { -1i32 };

    let mut m = negative_infinity::<f32>();
    let mut l = 0.0f32;
    let tile_elems = (key_tile * head_dim) as u32;
    let mut k_base = 0u32;
    // Every unit of the cube runs every tile (the loads and syncs are cube-wide); the cube's
    // group is uniform, so out-of-grid cubes simply never read their (zeroed) tiles.
    while k_base < n_tokens {
        // Stage K, V rows [k_base, k_base + key_tile) and their region ids (coalesced).
        let mut i = UNIT_POS_X;
        while i < tile_elems {
            let key = k_base + i / head_dim as u32;
            let in_range = in_grid && key < n_tokens;
            // Index 0 for padding: the read stays in bounds even if both arms are evaluated.
            let g_off = if in_range { base + k_base * head_dim as u32 + i } else { 0u32 };
            let kv = k[g_off as usize];
            let vv = v[g_off as usize];
            s_k[i as usize] = if in_range { kv } else { F::new(0.0f32) };
            s_v[i as usize] = if in_range { vv } else { F::new(0.0f32) };
            i += CUBE_DIM_X;
        }
        let mut j = UNIT_POS_X;
        while j < key_tile as u32 {
            let key = k_base + j;
            let in_range = in_grid && key < n_tokens;
            let r_idx = if in_range && use_regions != 0u32 { region_base + key } else { 0u32 };
            let r_read = region_ids[r_idx as usize];
            let r = if in_range { if use_regions != 0u32 { r_read } else { 0i32 } } else { -1i32 };
            s_r[j as usize] = r;
            j += CUBE_DIM_X;
        }
        sync_cube();

        if rq >= 0i32 {
            // Keys stay a run-time loop; only the head_dim loops unroll (registers).
            for jj in 0..key_tile {
                let rk = s_r[jj];
                if rk == rq {
                    let k_off = jj * head_dim;
                    let mut s = 0.0f32;
                    #[unroll]
                    for d in 0..head_dim {
                        s += q_row[d] * f32::cast_from(s_k[k_off + d]);
                    }
                    if s > m {
                        // New running max: rescale what was accumulated under the old one.
                        let correction = f32::exp(m - s);
                        l *= correction;
                        #[unroll]
                        for d in 0..head_dim {
                            acc[d] *= correction;
                        }
                        m = s;
                    }
                    let p = f32::exp(s - m);
                    l += p;
                    #[unroll]
                    for d in 0..head_dim {
                        acc[d] += p * f32::cast_from(s_v[k_off + d]);
                    }
                }
            }
        }
        sync_cube();
        k_base += key_tile as u32;
    }

    if active {
        let out_off = base + query * head_dim as u32;
        let inv_l = if l > 0.0f32 { 1.0f32 / l } else { 0.0f32 };
        #[unroll]
        for d in 0..head_dim {
            out[(out_off + d as u32) as usize] = F::cast_from(acc[d] * inv_l);
        }
    }
}

/// Launches [`sdpa_rows_kernel`] on contiguous `[groups, n_tokens, head_dim]` `q`, `k`, `v`
/// (`groups = windows × heads`, heads fastest), with optional per-window region ids
/// (`[num_windows, n_tokens]`). Returns the output handle.
///
/// The cube size comes from the runtime for a row of `n_tokens` queries; the key tile from the
/// device's shared memory (two `key_tile × head_dim` tiles of `F` plus ids), as a power of two.
#[allow(clippy::too_many_arguments)]
pub fn launch_sdpa_rows<F: CkFloat>(
    client: &Client,
    q: Handle,
    k: Handle,
    v: Handle,
    region_ids: Option<(&Handle, usize)>,
    groups: usize,
    num_heads: usize,
    n_tokens: usize,
    head_dim: usize,
) -> CkResult<Handle> {
    if !SDPA_ROWS_HEAD_DIMS.contains(&head_dim) {
        return Err(CkError::InvalidConfig(format!(
            "sdpa_rows_kernel is compiled for head_dim in {SDPA_ROWS_HEAD_DIMS:?}, got {head_dim}"
        )));
    }
    let total = groups * n_tokens * head_dim;
    let out = empty::<F>(client, total);
    let units = LaunchGeometry::per_row(client, 1, n_tokens).cube_dim.x as usize;
    let q_tiles = n_tokens.div_ceil(units);
    let geom = LaunchGeometry::per_row(client, groups * q_tiles, n_tokens);
    let key_tile = sdpa_rows_key_tile(client, head_dim, size_of::<F>());
    let (regions, regions_len, num_windows, use_regions) = match region_ids {
        Some((h, nw)) => (h.clone(), nw * n_tokens, nw, 1u32),
        // An unused 1-element buffer, never the output (no aliased binding).
        None => (empty::<i32>(client, 1), 1, 1, 0u32),
    };
    unsafe {
        sdpa_rows_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(q, total),
            BufferArg::from_raw_parts(k, total),
            BufferArg::from_raw_parts(v, total),
            BufferArg::from_raw_parts(regions, regions_len),
            BufferArg::from_raw_parts(out.clone(), total),
            groups as u32,
            q_tiles as u32,
            num_heads as u32,
            num_windows as u32,
            n_tokens as u32,
            use_regions,
            (1.0 / (head_dim as f64).sqrt()) as f32,
            head_dim,
            key_tile,
        );
    }
    Ok(out)
}

/// Keys per shared-memory tile: the largest power of two whose K and V tiles (`F`) and region ids
/// fit the device's shared memory (16 KiB guaranteed on WebGPU, more natively).
pub fn sdpa_rows_key_tile(client: &Client, head_dim: usize, elem_bytes: usize) -> usize {
    let shared = client.properties().hardware.max_shared_memory_size;
    let per_key = 2 * head_dim * elem_bytes + size_of::<i32>();
    let fit = (shared / per_key).max(1);
    1 << (usize::BITS - 1 - fit.leading_zeros())
}

#[cfg(test)]
mod tests {
    use super::*;
    use corridorkey_base::core::buffer::{download, upload};
    use corridorkey_base::core::{cast_f32, to_f64};
    use corridorkey_core::ComputeTarget;
    use corridorkey_core::compute::ComputeTask;
    use half::f16;

    /// f64 reference: per group, softmax over keys whose region equals the query's (`>= 0`).
    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        regions: &[i32],
        (groups, heads, windows, n, d): (usize, usize, usize, usize, usize),
    ) -> Vec<f64> {
        let mut out = vec![0.0; groups * n * d];
        let scale = 1.0 / (d as f64).sqrt();
        for g in 0..groups {
            let w = g / heads % windows;
            for i in 0..n {
                let rq = regions[w * n + i];
                if rq < 0 {
                    continue;
                }
                let qi = &q[(g * n + i) * d..(g * n + i + 1) * d];
                let scores: Vec<Option<f64>> = (0..n)
                    .map(|j| {
                        (regions[w * n + j] == rq).then(|| {
                            let kj = &k[(g * n + j) * d..(g * n + j + 1) * d];
                            qi.iter().zip(kj).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() * scale
                        })
                    })
                    .collect();
                let m = scores.iter().flatten().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let l: f64 = scores.iter().flatten().map(|s| (s - m).exp()).sum();
                for (j, s) in scores.iter().enumerate() {
                    if let Some(s) = s {
                        let p = (s - m).exp() / l;
                        for c in 0..d {
                            out[(g * n + i) * d + c] += p * v[(g * n + j) * d + c] as f64;
                        }
                    }
                }
            }
        }
        out
    }

    struct Case {
        n: usize,
        d: usize,
    }

    impl ComputeTask for Case {
        type Output = ();
        fn run(self, client: Client) {
            let (heads, windows) = (3, 2);
            let groups = heads * windows;
            let (n, d) = (self.n, self.d);
            let len = groups * n * d;
            let wave = |s: f32| (0..len).map(move |i| (i as f32 * s).sin() * 1.7).collect::<Vec<f32>>();
            let (q, k, v) = (wave(0.37), wave(0.53), wave(0.71));
            // Three regions plus padding (-1), interleaved so masks cut every key tile.
            let regions: Vec<i32> = (0..windows * n).map(|i| ((i * 7 + i / 5) % 4) as i32 - 1).collect();
            let want = reference(&q, &k, &v, &regions, (groups, heads, windows, n, d));
            let r = upload(&client, &regions);

            let got32: Vec<f32> = {
                let (hq, hk, hv) = (upload(&client, &q), upload(&client, &k), upload(&client, &v));
                let out = launch_sdpa_rows::<f32>(&client, hq, hk, hv, Some((&r, windows)), groups, heads, n, d).unwrap();
                download(&client, out)
            };
            let max32 = got32.iter().zip(&want).fold(0f64, |m, (&g, &w)| m.max((g as f64 - w).abs()));
            assert!(max32 < 1e-5, "{} f32 n={n} d={d}: max |err| {max32:.3e}", client.name());

            let to16 = |x: &[f32]| upload(&client, &cast_f32::<f16>(x));
            let out16 = launch_sdpa_rows::<f16>(&client, to16(&q), to16(&k), to16(&v), Some((&r, windows)), groups, heads, n, d)
                .unwrap();
            let got16: Vec<f16> = download(&client, out16);
            let max16 = got16.iter().zip(&want).fold(0f64, |m, (&g, &w)| m.max((to_f64(g) - w).abs()));
            // f16 inputs and output (|out| <= 1.7): rounding of q, k, v and the result, f32 inside.
            assert!(max16 < 1e-2, "{} f16 n={n} d={d}: max |err| {max16:.3e}", client.name());
        }
    }

    #[test]
    fn matches_f64_reference_with_region_masks_on_odd_lengths() {
        for target in ComputeTarget::available() {
            for d in SDPA_ROWS_HEAD_DIMS {
                for n in [77, 300] {
                    target.run(Case { n, d }).unwrap();
                }
            }
        }
    }
}
