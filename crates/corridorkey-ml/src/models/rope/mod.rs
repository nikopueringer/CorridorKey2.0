//! 2D and 3D Rotary Position Embeddings (`corridorkey/rope.py`).
//!
//! Builds DINOv3/EUPE cell-centre normalized `cos`/`sin` frequency tables in `f64` on the host
//! and applies interleaved pair rotation on the device in a single pass ([`apply_rotary`]).

pub mod kernels;

use std::{
    collections::HashMap,
    f64::consts::PI,
    sync::{Arc, Mutex},
};

use corridorkey_base::core::{CkFloat, buffer::upload};
use corridorkey_core::compute::LaunchGeometry;
use corridorkey_core::{CkError, CkResult};
use cubecl::device::DeviceId;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::runtime::MlContext;

use kernels::apply_rotary_kernel;

/// Default frequency base in `corridorkey/rope.py`.
pub const DEFAULT_ROPE_BASE: f64 = 100.0;

/// Period lengths `base^(2i / dim)` for `i in 0..dim/2`.
pub fn compute_periods(dim: usize, base: f64) -> Vec<f64> {
    assert_eq!(dim % 2, 0, "RoPE axis dim ({dim}) must be even");
    if dim == 0 {
        return Vec::new();
    }
    let half = dim / 2;
    (0..half)
        .map(|i| base.powf((2 * i) as f64 / dim as f64))
        .collect()
}

/// Cell-centre normalized coordinates `2 * ((i + 0.5) / n) - 1` in `[-1, 1]`.
pub fn normalized_coords(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|i| 2.0 * ((i as f64 + 0.5) / n as f64) - 1.0)
        .collect()
}

/// Per-axis `(cos, sin)` tables of shape `[n, dim / 2]`.
fn axis_cos_sin(coords: &[f64], periods: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let half = periods.len();
    let mut cos = Vec::with_capacity(coords.len() * half);
    let mut sin = Vec::with_capacity(coords.len() * half);
    for &c in coords {
        for &p in periods {
            let angle = 2.0 * PI * c / p;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

type Rope2dCache = Arc<Mutex<HashMap<(DeviceId, usize, usize, usize), (Handle, Handle)>>>;
type Rope3dCache =
    Arc<Mutex<HashMap<(DeviceId, usize, usize, usize, u64, usize), (Handle, Handle)>>>;

/// 2D Rotary Position Embedding (`RoPE2D` in `corridorkey/rope.py`) for Stage 2 spatial blocks.
#[derive(Debug, Clone)]
pub struct Rope2d {
    head_dim: usize,
    base: f64,
    periods_h: Vec<f64>,
    periods_w: Vec<f64>,
    cache: Rope2dCache,
}

impl Rope2d {
    pub fn new(head_dim: usize, base: f64) -> CkResult<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(4) {
            return Err(CkError::InvalidConfig(format!(
                "RoPE2D head_dim ({head_dim}) must be a positive multiple of 4"
            )));
        }
        let dim_each = head_dim / 2;
        Ok(Self {
            head_dim,
            base,
            periods_h: compute_periods(dim_each, base),
            periods_w: compute_periods(dim_each, base),
            cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn base(&self) -> f64 {
        self.base
    }

    /// Returns device-resident `(cos, sin)` handles for `(num_h, num_w)`, building and uploading
    /// once per device/shape/dtype and reusing across subsequent calls.
    pub fn cos_sin_handles<F: CkFloat>(
        &self,
        ctx: &MlContext,
        num_h: usize,
        num_w: usize,
    ) -> (Handle, Handle) {
        let key = (ctx.device_id(), num_h, num_w, size_of::<F>());
        let mut guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pair) = guard.get(&key) {
            return pair.clone();
        }
        let (cos_vec, sin_vec) = self.tables::<F>(num_h, num_w);
        let pair = (
            upload(ctx.client(), &cos_vec),
            upload(ctx.client(), &sin_vec),
        );
        guard.insert(key, pair.clone());
        pair
    }

    /// Builds flattened `(cos, sin)` tables of shape `[num_h * num_w, head_dim / 2]` in `f64`.
    pub fn tables_f64(&self, num_h: usize, num_w: usize) -> (Vec<f64>, Vec<f64>) {
        let coord_h = normalized_coords(num_h);
        let coord_w = normalized_coords(num_w);
        let (cos_h, sin_h) = axis_cos_sin(&coord_h, &self.periods_h);
        let (cos_w, sin_w) = axis_cos_sin(&coord_w, &self.periods_w);
        let pairs_h = self.periods_h.len();
        let pairs_w = self.periods_w.len();
        let half = pairs_h + pairs_w;
        let n = num_h * num_w;
        let mut cos = Vec::with_capacity(n * half);
        let mut sin = Vec::with_capacity(n * half);
        for h in 0..num_h {
            let ch = &cos_h[h * pairs_h..(h + 1) * pairs_h];
            let sh = &sin_h[h * pairs_h..(h + 1) * pairs_h];
            for w in 0..num_w {
                let cw = &cos_w[w * pairs_w..(w + 1) * pairs_w];
                let sw = &sin_w[w * pairs_w..(w + 1) * pairs_w];
                cos.extend_from_slice(ch);
                cos.extend_from_slice(cw);
                sin.extend_from_slice(sh);
                sin.extend_from_slice(sw);
            }
        }
        (cos, sin)
    }

    /// Builds `(cos, sin)` tables cast to `F`.
    pub fn tables<F: CkFloat>(&self, num_h: usize, num_w: usize) -> (Vec<F>, Vec<F>) {
        let (c, s) = self.tables_f64(num_h, num_w);
        (
            c.into_iter().map(corridorkey_base::core::cast).collect(),
            s.into_iter().map(corridorkey_base::core::cast).collect(),
        )
    }
}

/// 3D Rotary Position Embedding (`RoPE3D` in `corridorkey/rope.py`) for Stage 1 spatiotemporal
/// blocks.
#[derive(Debug, Clone)]
pub struct Rope3d {
    head_dim: usize,
    base: f64,
    periods_t: Vec<f64>,
    periods_h: Vec<f64>,
    periods_w: Vec<f64>,
    cache: Rope3dCache,
}

impl Rope3d {
    pub fn new(head_dim: usize, base: f64) -> CkResult<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(CkError::InvalidConfig(format!(
                "RoPE3D head_dim ({head_dim}) must be a positive even integer"
            )));
        }
        let total_pairs = head_dim / 2;
        let base_pairs = total_pairs / 3;
        let rem = total_pairs - base_pairs * 3;
        let pairs_t = base_pairs;
        let pairs_h = base_pairs + usize::from(rem >= 2);
        let pairs_w = base_pairs + usize::from(rem >= 1);
        Ok(Self {
            head_dim,
            base,
            periods_t: compute_periods(pairs_t * 2, base),
            periods_h: compute_periods(pairs_h * 2, base),
            periods_w: compute_periods(pairs_w * 2, base),
            cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Returns device-resident `(cos, sin)` handles for `(num_t, num_h, num_w, t_stride)`,
    /// building and uploading once per device/configuration and reusing across subsequent calls.
    pub fn cos_sin_handles<F: CkFloat>(
        &self,
        ctx: &MlContext,
        num_t: usize,
        num_h: usize,
        num_w: usize,
        t_stride: f64,
    ) -> (Handle, Handle) {
        let key = (
            ctx.device_id(),
            num_t,
            num_h,
            num_w,
            t_stride.to_bits(),
            size_of::<F>(),
        );
        let mut guard = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pair) = guard.get(&key) {
            return pair.clone();
        }
        let (cos_vec, sin_vec) = self.tables::<F>(num_t, num_h, num_w, t_stride);
        let pair = (
            upload(ctx.client(), &cos_vec),
            upload(ctx.client(), &sin_vec),
        );
        guard.insert(key, pair.clone());
        pair
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn axis_dims(&self) -> (usize, usize, usize) {
        (
            self.periods_t.len() * 2,
            self.periods_h.len() * 2,
            self.periods_w.len() * 2,
        )
    }

    /// Builds flattened `(cos, sin)` tables of shape `[num_t * num_h * num_w, head_dim / 2]` in `f64`.
    pub fn tables_f64(
        &self,
        num_t: usize,
        num_h: usize,
        num_w: usize,
        t_stride: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let coord_t: Vec<f64> = normalized_coords(num_t)
            .into_iter()
            .map(|c| c * t_stride)
            .collect();
        let coord_h = normalized_coords(num_h);
        let coord_w = normalized_coords(num_w);
        let (cos_t, sin_t) = axis_cos_sin(&coord_t, &self.periods_t);
        let (cos_h, sin_h) = axis_cos_sin(&coord_h, &self.periods_h);
        let (cos_w, sin_w) = axis_cos_sin(&coord_w, &self.periods_w);
        let (pt, ph, pw) = (
            self.periods_t.len(),
            self.periods_h.len(),
            self.periods_w.len(),
        );
        let half = pt + ph + pw;
        let n = num_t * num_h * num_w;
        let mut cos = Vec::with_capacity(n * half);
        let mut sin = Vec::with_capacity(n * half);
        for t in 0..num_t {
            let ct = &cos_t[t * pt..(t + 1) * pt];
            let st = &sin_t[t * pt..(t + 1) * pt];
            for h in 0..num_h {
                let ch = &cos_h[h * ph..(h + 1) * ph];
                let sh = &sin_h[h * ph..(h + 1) * ph];
                for w in 0..num_w {
                    let cw = &cos_w[w * pw..(w + 1) * pw];
                    let sw = &sin_w[w * pw..(w + 1) * pw];
                    cos.extend_from_slice(ct);
                    cos.extend_from_slice(ch);
                    cos.extend_from_slice(cw);
                    sin.extend_from_slice(st);
                    sin.extend_from_slice(sh);
                    sin.extend_from_slice(sw);
                }
            }
        }
        (cos, sin)
    }

    /// Builds `(cos, sin)` tables cast to `F`.
    pub fn tables<F: CkFloat>(
        &self,
        num_t: usize,
        num_h: usize,
        num_w: usize,
        t_stride: f64,
    ) -> (Vec<F>, Vec<F>) {
        let (c, s) = self.tables_f64(num_t, num_h, num_w, t_stride);
        (
            c.into_iter().map(corridorkey_base::core::cast).collect(),
            s.into_iter().map(corridorkey_base::core::cast).collect(),
        )
    }

    pub const fn base(&self) -> f64 {
        self.base
    }
}

/// Applies rotary position embeddings on the device to `input` of shape
/// `[batches, n_tokens, head_dim]` using `cos` and `sin` tables of shape
/// `[n_tokens, head_dim / 2]`. `input` and `output` may be the same [`Handle`].
pub fn apply_rotary<F: CkFloat>(
    client: &Client,
    input: &Handle,
    cos: &Handle,
    sin: &Handle,
    output: &Handle,
    batches: usize,
    n_tokens: usize,
    head_dim: usize,
) {
    assert_eq!(head_dim % 2, 0, "head_dim ({head_dim}) must be even");
    let half_dim = head_dim / 2;
    let table_pairs = n_tokens * half_dim;
    let total_pairs = batches * table_pairs;
    if total_pairs == 0 {
        return;
    }
    let total_elems = total_pairs * 2;
    let geom = LaunchGeometry::elementwise(client, total_pairs);
    unsafe {
        apply_rotary_kernel::launch::<F>(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(input.clone(), total_elems),
            BufferArg::from_raw_parts(cos.clone(), table_pairs),
            BufferArg::from_raw_parts(sin.clone(), table_pairs),
            BufferArg::from_raw_parts(output.clone(), total_elems),
            total_pairs as u32,
            table_pairs as u32,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use corridorkey_base::core::buffer::{download, empty, upload};
    use corridorkey_core::ComputeTarget;

    fn host_apply_rotary(
        x: &[f64],
        cos: &[f64],
        sin: &[f64],
        batches: usize,
        n_tokens: usize,
        head_dim: usize,
    ) -> Vec<f64> {
        let half = head_dim / 2;
        let table_pairs = n_tokens * half;
        let mut out = vec![0.0; x.len()];
        for b in 0..batches {
            for p in 0..table_pairs {
                let idx = (b * table_pairs + p) * 2;
                let (x1, x2) = (x[idx], x[idx + 1]);
                let (c, s) = (cos[p], sin[p]);
                out[idx] = x1 * c - x2 * s;
                out[idx + 1] = x1 * s + x2 * c;
            }
        }
        out
    }

    #[test]
    fn rope3d_splits_even_and_non_divisible_head_dims_like_python() {
        // Stage 1 /16 has head_dim = 512 / 8 = 64 (not divisible by 3): pairs = 10, 11, 11 -> dims 20, 22, 22
        let r64 = Rope3d::new(64, DEFAULT_ROPE_BASE).unwrap();
        assert_eq!(r64.axis_dims(), (20, 22, 22));
        // Stage 1 /64 has head_dim = 1536 / 16 = 96 (divisible by 3): dims 32, 32, 32
        let r96 = Rope3d::new(96, DEFAULT_ROPE_BASE).unwrap();
        assert_eq!(r96.axis_dims(), (32, 32, 32));
    }

    #[test]
    fn rotary_kernel_matches_f64_reference_on_odd_token_counts() {
        let rope2d = Rope2d::new(80, DEFAULT_ROPE_BASE).unwrap();
        let (batches, num_h, num_w, head_dim) = (3, 5, 7, 80); // odd H, W, batches
        let n_tokens = num_h * num_w;
        let (cos64, sin64) = rope2d.tables_f64(num_h, num_w);
        let (cos32, sin32) = rope2d.tables::<f32>(num_h, num_w);

        let len = batches * n_tokens * head_dim;
        let x64: Vec<f64> = (0..len).map(|i| ((i as f64) * 0.137).sin()).collect();
        let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
        let x64_from_f32: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
        let want = host_apply_rotary(&x64_from_f32, &cos64, &sin64, batches, n_tokens, head_dim);

        for target in ComputeTarget::available() {
            let client = target.client().unwrap();
            let d_x = upload(&client, &x32);
            let d_cos = upload(&client, &cos32);
            let d_sin = upload(&client, &sin32);
            let d_out = empty::<f32>(&client, len);
            apply_rotary::<f32>(
                &client, &d_x, &d_cos, &d_sin, &d_out, batches, n_tokens, head_dim,
            );
            let got: Vec<f32> = download(&client, d_out);
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g as f64 - w).abs() < 1e-6,
                    "{} [{i}]: {g} vs {w}",
                    client.name()
                );
            }
        }
    }
}
