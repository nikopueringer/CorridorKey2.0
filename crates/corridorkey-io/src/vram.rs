//! The Stage 1 window from the device memory budget (`select_vram_profile`, `apply_auto_vram`),
//! with this engine's own peak model: the product's table holds PyTorch's peaks, about twice ours.
//!
//! Peak device memory (driver-reported, CUDA context and allocator pools included) is modelled as
//! `fixed + per_mp · MP + per_mp_frame · MP · T`, MP the padded frame's megapixels and T the window.
//! Fitted from `corridorkey bench --sweep` on an RTX 2070 (8 GB, CUDA, 2026-10-27,
//! `data/peak_sweep_rtx2070_cuda.jsonl`) over 768x512, 1280x768 and 1920x1080 and windows 1–24,
//! on the runs that stayed below 6.5 GB (above it CubeCL's pools reclaim under pressure and the
//! readings measure the card, not the need); every measured point lies at or under the model.

use crate::clip::Precision;

/// Peak device memory model, in GB (2^30 bytes).
#[derive(Debug, Clone, Copy)]
pub struct PeakModel {
    pub fixed_gb: f64,
    pub per_mp_gb: f64,
    pub per_mp_frame_gb: f64,
}

impl PeakModel {
    pub fn estimate_gb(&self, megapixels: f64, frames: usize) -> f64 {
        self.fixed_gb + self.per_mp_gb * megapixels + self.per_mp_frame_gb * megapixels * frames as f64
    }
}

/// f16 networks (measured 2.66 GB at 0.39 MP × 1 to 6.04 GB at 2.21 MP × 8).
pub const F16_PEAK: PeakModel = PeakModel { fixed_gb: 2.55, per_mp_gb: 0.30, per_mp_frame_gb: 0.20 };
/// f32 networks: weights and activations twice as large (5.26 GB at 0.39 MP × 1).
pub const F32_PEAK: PeakModel = PeakModel { fixed_gb: 5.0, per_mp_gb: 0.60, per_mp_frame_gb: 0.35 };

/// Windows tried, largest first (bigger windows give better temporal consistency).
const WINDOWS: [usize; 10] = [80, 64, 48, 32, 24, 16, 8, 4, 2, 1];
/// Safety margin on the estimate, as the product's (`_VRAM_MARGIN_SCALE`, `_VRAM_MARGIN_GB`).
const MARGIN_SCALE: f64 = 1.1;
const MARGIN_GB: f64 = 0.3;

/// A window chosen by [`select_window`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowChoice {
    pub chunk: usize,
    pub stride: usize,
    pub estimate_gb: f64,
    /// Whether the estimate fits the budget (a single-frame window is returned either way).
    pub fits: bool,
}

/// The largest window whose estimated peak (with margin) fits `budget_gb` for a clip of `frames`
/// frames of `megapixels` (padded), stride half of it.
pub fn select_window(megapixels: f64, frames: usize, budget_gb: f64, precision: Precision) -> WindowChoice {
    let model = if precision == Precision::F16 { F16_PEAK } else { F32_PEAK };
    let mut estimate_gb = 0.0;
    for chunk in WINDOWS {
        estimate_gb = model.estimate_gb(megapixels, chunk.min(frames.max(1)));
        if estimate_gb * MARGIN_SCALE + MARGIN_GB <= budget_gb {
            return WindowChoice { chunk, stride: (chunk / 2).max(1), estimate_gb, fits: true };
        }
    }
    WindowChoice { chunk: 1, stride: 1, estimate_gb, fits: false }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Measured peaks of the sweep (precision, MP, window, GB) that the model must not underestimate.
    const MEASURED: &[(Precision, f64, usize, f64)] = &[
        (Precision::F16, 0.393, 1, 2.66),
        (Precision::F16, 0.393, 8, 3.07),
        (Precision::F16, 0.393, 24, 4.47),
        (Precision::F16, 0.983, 8, 4.12),
        (Precision::F16, 2.212, 1, 3.27),
        (Precision::F16, 2.212, 4, 4.38),
        (Precision::F16, 2.212, 8, 6.04),
        (Precision::F32, 0.393, 1, 5.26),
        (Precision::F32, 0.393, 8, 6.06),
        (Precision::F32, 0.983, 4, 6.45),
        (Precision::F32, 2.212, 1, 6.65),
    ];

    #[test]
    fn model_covers_the_measurements() {
        for &(p, mp, t, gb) in MEASURED {
            let model = if p == Precision::F16 { F16_PEAK } else { F32_PEAK };
            let est = model.estimate_gb(mp, t);
            assert!(est >= gb && est < gb * 1.25, "{p:?} {mp} MP × {t}: estimate {est:.2} GB vs measured {gb} GB");
        }
    }

    #[test]
    fn picks_the_largest_window_that_fits() {
        // 8 GB card, ~7.6 GB free, 1080p f16: a window of 8 is estimated at 6.75 GB (7.73 with the
        // margin), so 4 (4.98 GB, 5.78 with the margin)
        let c = select_window(2.212, 1000, 7.6, Precision::F16);
        assert_eq!((c.chunk, c.stride, c.fits), (4, 2, true));
        // 768x512 f16: up to 49 frames fit, so 48
        assert_eq!(select_window(0.393, 1000, 7.6, Precision::F16).chunk, 48);
        // a short clip caps the estimate at its length
        assert_eq!(select_window(0.393, 12, 7.6, Precision::F16).chunk, 80);
        // nothing fits: a single frame, flagged
        let c = select_window(8.4, 100, 3.0, Precision::F32);
        assert_eq!((c.chunk, c.fits), (1, false));
    }
}
