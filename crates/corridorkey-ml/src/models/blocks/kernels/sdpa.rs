//! Fused shared-memory tiled FlashAttention CubeCL kernel (`sdpa_window_kernel`) for local
//! windows (`SwinBlock2d`, `SwinBlock3d`), axial strips (`CSwinBlock2d`, `CSwinBlock3d`), and
//! general sequences on backends without CMMA Tensor Cores.
//!
//! Each cube processes a tile of `SDPA_Q_TILE = 8` queries and streams keys/values through shared
//! SRAM in tiles of `SDPA_K_TILE = 16`:
//! - `Q`, `K`, and `V` are staged in `Shared` SRAM (`12.6 KB` total per workgroup, within WebGPU's
//!   16 KB limit), cutting global-memory `K`/`V` traffic by `8×` and keeping all `Q·K^T` and `P·V`
//!   inner loops in on-chip shared memory.
//! - Supports optional per-window `region_ids: [num_windows * n_tokens]` (`i32`, `-1` for padded
//!   out-of-bounds tokens) so shifted and padded Swin windows execute in a **single kernel launch**
//!   and skip entire `16`-key tiles that have no region overlap with the `8` active queries.

use cubecl::{prelude::*, server::Handle};

use corridorkey_base::core::{CkFloat, buffer::empty, cast};
use corridorkey_core::{
    CkError, CkResult,
    compute::{LaunchGeometry, negative_infinity, row_position},
};

/// Maximum per-head dimension staged in shared memory (`64` in Stage 1 `/16`, `80` in Stage 2,
/// `96` in Stage 1 `/64`).
pub const SDPA_MAX_HEAD_DIM: u32 = 96;

/// Number of query tokens processed per cube (`B_r = 8`).
pub const SDPA_Q_TILE: u32 = 8;

/// Number of key/value tokens staged per shared-memory tile (`B_c = 16`).
pub const SDPA_K_TILE: u32 = 16;

/// Number of `(q_r, k_c)` score pairs per tile (`8 * 16 = 128`).
pub const SDPA_SCORE_ELEMS: u32 = SDPA_Q_TILE * SDPA_K_TILE;

/// Shared-memory size for `s_q` and `s_acc` (`8 * 96 = 768` floats each).
pub const SDPA_Q_SMEM_LEN: u32 = SDPA_Q_TILE * SDPA_MAX_HEAD_DIM;

/// Shared-memory size for `s_kv` (`16 * 96 = 1536` floats, reused for `K` then `V`).
pub const SDPA_KV_SMEM_LEN: u32 = SDPA_K_TILE * SDPA_MAX_HEAD_DIM;

/// Target threads per cube for `LaunchGeometry::per_row` (`128` threads = 1 thread per `(q_r, k_c)` pair).
pub const SDPA_TARGET_THREADS: usize = 128;

/// Fused shared-memory tiled FlashAttention kernel (`B_r = 8, B_c = 16`).
#[cube(launch)]
pub fn sdpa_window_kernel<F: CkFloat>(
    q: &[F],
    k: &[F],
    v: &[F],
    region_ids: &[i32],
    out: &mut [F],
    total_q_tiles: u32,
    q_tiles_per_head: u32,
    num_windows: u32,
    num_heads: u32,
    n_tokens: u32,
    head_dim: u32,
    use_region_mask: u32,
    scale: F,
    #[comptime] block_size: u32,
) {
    let mut s_q = Shared::<[F]>::new_slice(SDPA_Q_SMEM_LEN as usize);
    let mut s_acc = Shared::<[F]>::new_slice(SDPA_Q_SMEM_LEN as usize);
    let mut s_kv = Shared::<[F]>::new_slice(SDPA_KV_SMEM_LEN as usize);
    let mut s_scores = Shared::<[F]>::new_slice(SDPA_SCORE_ELEMS as usize);
    let mut s_m = Shared::<[F]>::new_slice(SDPA_Q_TILE as usize);
    let mut s_l = Shared::<[F]>::new_slice(SDPA_Q_TILE as usize);
    let mut s_rescale = Shared::<[F]>::new_slice(SDPA_Q_TILE as usize);
    let mut s_rq = Shared::<[i32]>::new_slice(SDPA_Q_TILE as usize);
    let mut s_rk = Shared::<[i32]>::new_slice(SDPA_K_TILE as usize);
    let mut s_tile_active = Shared::<[u32]>::new_slice(1usize);

    let tile_row = row_position();
    let tid = UNIT_POS_X;

    if tile_row < total_q_tiles {
        let q_tile_idx = tile_row % q_tiles_per_head;
        let wh = tile_row / q_tiles_per_head;
        let win_idx = wh / num_heads;
        let w_local = win_idx % num_windows;

        let q_tile_base = q_tile_idx * SDPA_Q_TILE;
        let bh_base = wh * n_tokens * head_dim;
        let win_region_base = w_local * n_tokens;

        // 1. Load query region IDs and initialize online softmax state
        let mut qr_init = tid;
        while qr_init < SDPA_Q_TILE {
            let q_tok = q_tile_base + qr_init;
            let rq = if q_tok < n_tokens {
                if use_region_mask != 0u32 {
                    region_ids[(win_region_base + q_tok) as usize]
                } else {
                    0i32
                }
            } else {
                -1i32
            };
            s_rq[qr_init as usize] = rq;
            s_m[qr_init as usize] = negative_infinity::<F>();
            s_l[qr_init as usize] = F::new(0.0f32);
            qr_init += block_size;
        }

        // Load Q tile [SDPA_Q_TILE, head_dim] into s_q and zero s_acc
        let q_tile_elems = SDPA_Q_TILE * head_dim;
        let mut qi = tid;
        while qi < q_tile_elems {
            let qr = qi / head_dim;
            let d = qi % head_dim;
            let q_tok = q_tile_base + qr;
            let smem_idx = (qr * SDPA_MAX_HEAD_DIM + d) as usize;
            let val = if q_tok < n_tokens {
                q[(bh_base + q_tok * head_dim + d) as usize]
            } else {
                F::new(0.0f32)
            };
            s_q[smem_idx] = val;
            s_acc[smem_idx] = F::new(0.0f32);
            qi += block_size;
        }
        sync_cube();

        // 2. Stream K and V in tiles of SDPA_K_TILE = 16
        let kv_tile_elems = SDPA_K_TILE * head_dim;
        let mut k_base = 0u32;
        while k_base < n_tokens {
            let mut kc_init = tid;
            while kc_init < SDPA_K_TILE {
                let k_tok = k_base + kc_init;
                let rk = if k_tok < n_tokens {
                    if use_region_mask != 0u32 {
                        region_ids[(win_region_base + k_tok) as usize]
                    } else {
                        0i32
                    }
                } else {
                    -1i32
                };
                s_rk[kc_init as usize] = rk;
                kc_init += block_size;
            }
            if tid == 0u32 {
                s_tile_active[0] = if use_region_mask == 0u32 { 1u32 } else { 0u32 };
            }
            sync_cube();

            if use_region_mask != 0u32 {
                let mut pair = tid;
                while pair < SDPA_SCORE_ELEMS {
                    let qr = pair / SDPA_K_TILE;
                    let kc = pair % SDPA_K_TILE;
                    let rq = s_rq[qr as usize];
                    let rk = s_rk[kc as usize];
                    if rq >= 0i32 && rq == rk {
                        s_tile_active[0] = 1u32;
                    }
                    pair += block_size;
                }
                sync_cube();
            }

            if s_tile_active[0] != 0u32 {
                // Stage K tile [SDPA_K_TILE, head_dim] into s_kv
                let mut ki = tid;
                while ki < kv_tile_elems {
                    let kc = ki / head_dim;
                    let d = ki % head_dim;
                    let k_tok = k_base + kc;
                    let smem_idx = (kc * SDPA_MAX_HEAD_DIM + d) as usize;
                    s_kv[smem_idx] = if k_tok < n_tokens {
                        k[(bh_base + k_tok * head_dim + d) as usize]
                    } else {
                        F::new(0.0f32)
                    };
                    ki += block_size;
                }
                sync_cube();

                // Compute 8 × 16 Q·K^T scores in shared memory
                let mut pair = tid;
                while pair < SDPA_SCORE_ELEMS {
                    let qr = pair / SDPA_K_TILE;
                    let kc = pair % SDPA_K_TILE;
                    let rq = s_rq[qr as usize];
                    let rk = s_rk[kc as usize];
                    if rq >= 0i32 && rq == rk {
                        let q_off = qr * SDPA_MAX_HEAD_DIM;
                        let k_off = kc * SDPA_MAX_HEAD_DIM;
                        let mut dot = F::new(0.0f32);
                        for cd in 0..head_dim {
                            dot += s_q[(q_off + cd) as usize] * s_kv[(k_off + cd) as usize];
                        }
                        s_scores[pair as usize] = dot * scale;
                    } else {
                        s_scores[pair as usize] = negative_infinity::<F>();
                    }
                    pair += block_size;
                }
                sync_cube();

                // Parallel online softmax per query row (threads 0..8) + simultaneous V tile load (all threads)
                let mut qr_soft = tid;
                while qr_soft < SDPA_Q_TILE {
                    let row_off = qr_soft * SDPA_K_TILE;
                    let mut row_max = negative_infinity::<F>();
                    for kc in 0..SDPA_K_TILE {
                        let sc = s_scores[(row_off + kc) as usize];
                        if sc > row_max {
                            row_max = sc;
                        }
                    }
                    if row_max > negative_infinity::<F>() {
                        let m_old = s_m[qr_soft as usize];
                        let m_new = if row_max > m_old { row_max } else { m_old };
                        let rescale = if m_old == negative_infinity::<F>() {
                            F::new(0.0f32)
                        } else {
                            F::exp(m_old - m_new)
                        };
                        let mut row_sum = F::new(0.0f32);
                        for kc in 0..SDPA_K_TILE {
                            let idx = (row_off + kc) as usize;
                            let sc = s_scores[idx];
                            let p = if sc == negative_infinity::<F>() {
                                F::new(0.0f32)
                            } else {
                                F::exp(sc - m_new)
                            };
                            s_scores[idx] = p;
                            row_sum += p;
                        }
                        s_m[qr_soft as usize] = m_new;
                        s_l[qr_soft as usize] = s_l[qr_soft as usize] * rescale + row_sum;
                        s_rescale[qr_soft as usize] = rescale;
                    } else {
                        for kc in 0..SDPA_K_TILE {
                            s_scores[(row_off + kc) as usize] = F::new(0.0f32);
                        }
                        s_rescale[qr_soft as usize] = F::new(1.0f32);
                    }
                    qr_soft += block_size;
                }

                // Simultaneously stage V tile [SDPA_K_TILE, head_dim] into s_kv
                let mut vi = tid;
                while vi < kv_tile_elems {
                    let kc = vi / head_dim;
                    let d = vi % head_dim;
                    let k_tok = k_base + kc;
                    let smem_idx = (kc * SDPA_MAX_HEAD_DIM + d) as usize;
                    s_kv[smem_idx] = if k_tok < n_tokens {
                        v[(bh_base + k_tok * head_dim + d) as usize]
                    } else {
                        F::new(0.0f32)
                    };
                    vi += block_size;
                }
                sync_cube();

                // Accumulate P · V into s_acc[qr, d] from shared memory
                let mut oi = tid;
                while oi < q_tile_elems {
                    let qr = oi / head_dim;
                    let d = oi % head_dim;
                    let acc_idx = (qr * SDPA_MAX_HEAD_DIM + d) as usize;
                    let row_off = qr * SDPA_K_TILE;
                    let mut acc = s_acc[acc_idx] * s_rescale[qr as usize];
                    for kc in 0..SDPA_K_TILE {
                        let p = s_scores[(row_off + kc) as usize];
                        let vv = s_kv[(kc * SDPA_MAX_HEAD_DIM + d) as usize];
                        acc += p * vv;
                    }
                    s_acc[acc_idx] = acc;
                    oi += block_size;
                }
                sync_cube();
            }

            k_base += SDPA_K_TILE;
        }

        // 3. Normalize by running denominator s_l[qr] and write to global memory
        let mut out_i = tid;
        while out_i < q_tile_elems {
            let qr = out_i / head_dim;
            let d = out_i % head_dim;
            let q_tok = q_tile_base + qr;
            if q_tok < n_tokens {
                let l_val = s_l[qr as usize];
                let val = if s_rq[qr as usize] >= 0i32 && l_val > F::new(0.0f32) {
                    s_acc[(qr * SDPA_MAX_HEAD_DIM + d) as usize] / l_val
                } else {
                    F::new(0.0f32)
                };
                out[(bh_base + q_tok * head_dim + d) as usize] = val;
            }
            out_i += block_size;
        }
    }
}

/// Launches [`sdpa_window_kernel`] on contiguous `[total_wins, num_heads, n_tokens, head_dim]`
/// `Q`, `K`, `V` handles, returning the output `Handle`.
#[allow(clippy::too_many_arguments)]
pub fn launch_sdpa_window<F: CkFloat>(
    client: &Client,
    q: Handle,
    k: Handle,
    v: Handle,
    region_ids: Option<(&Handle, usize)>,
    total_wins: usize,
    num_heads: usize,
    n_tokens: usize,
    head_dim: usize,
) -> CkResult<Handle> {
    if head_dim > SDPA_MAX_HEAD_DIM as usize {
        return Err(CkError::InvalidConfig(format!(
            "sdpa_window_kernel head_dim={head_dim} exceeds SDPA_MAX_HEAD_DIM={SDPA_MAX_HEAD_DIM}"
        )));
    }
    let total_rows = total_wins * num_heads * n_tokens;
    let total_elems = total_rows * head_dim;
    let out = empty::<F>(client, total_elems);
    let q_tiles_per_head = n_tokens.div_ceil(SDPA_Q_TILE as usize);
    let total_q_tiles = total_wins * num_heads * q_tiles_per_head;
    let hw = &client.properties().hardware;
    let plane = LaunchGeometry::plane_lanes(client);
    let target = (SDPA_TARGET_THREADS as u32).min(hw.max_units_per_cube).max(plane);
    let units = 1 << (u32::BITS - 1 - target.leading_zeros());
    let geom = LaunchGeometry::per_row(client, total_q_tiles, units as usize);
    let scale = cast::<F>(1.0 / (head_dim as f64).sqrt());

    let (reg_h, reg_len, num_windows, use_region) = match region_ids {
        Some((rh, nw)) => (rh.clone(), nw * n_tokens, nw, 1u32),
        None => (out.clone(), 1, total_wins.max(1), 0u32),
    };

    unsafe {
        sdpa_window_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(q, total_elems),
            BufferArg::from_raw_parts(k, total_elems),
            BufferArg::from_raw_parts(v, total_elems),
            BufferArg::from_raw_parts(reg_h, reg_len.max(1)),
            BufferArg::from_raw_parts(out.clone(), total_elems),
            total_q_tiles as u32,
            q_tiles_per_head as u32,
            num_windows as u32,
            num_heads as u32,
            n_tokens as u32,
            head_dim as u32,
            use_region,
            scale,
            geom.cube_dim.x,
        );
    }

    Ok(out)
}
