//! Colour encodings of frames (`corridorkey/color_utils.py`, `video_io.py`).

use serde::{Deserialize, Serialize};

/// Knee of the signed asinh HDR encoding (`HDR_ASINH_KNEE` in `color_utils.py`).
pub const HDR_ASINH_KNEE: f64 = 0.008;
/// Reference level of the signed asinh HDR encoding (`HDR_ASINH_REF` in `color_utils.py`).
pub const HDR_ASINH_REF: f64 = 16.0;

/// Transfer function of a buffer's values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorSpace {
    /// sRGB-encoded (IEC 61966-2-1), as decoded from 8/10-bit video.
    Srgb,
    /// Scene-linear, as read from EXR or after decoding sRGB.
    Linear,
    /// Signed asinh of scene-linear ([`HDR_ASINH_KNEE`], [`HDR_ASINH_REF`]): the model's input.
    Asinh,
}

/// Brightness / contrast applied to scene-linear input before the model and undone after
/// (`apply_linear_adjust` / `invert_linear_adjust`): `gain · (pivot + contrast · (x − pivot))`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LinearAdjust {
    pub gain: f64,
    pub contrast: f64,
    pub pivot: f64,
}

impl Default for LinearAdjust {
    fn default() -> Self {
        Self { gain: 1.0, contrast: 1.0, pivot: 0.18 }
    }
}

impl LinearAdjust {
    /// Differences from 1 below this are no adjustment (`has_linear_adjust`).
    pub const IDENTITY_TOLERANCE: f64 = 1e-6;

    pub fn is_identity(&self) -> bool {
        (self.gain - 1.0).abs() <= Self::IDENTITY_TOLERANCE && (self.contrast - 1.0).abs() <= Self::IDENTITY_TOLERANCE
    }

    /// `(scale, offset)` with `apply(x) = scale · x + offset`, the form kernels take.
    pub fn affine(&self) -> (f64, f64) {
        if self.is_identity() {
            return (1.0, 0.0);
        }
        (self.gain * self.contrast, self.gain * self.pivot * (1.0 - self.contrast))
    }

    /// `(scale, offset)` of the inverse.
    pub fn inverse_affine(&self) -> (f64, f64) {
        let (a, b) = self.affine();
        (1.0 / a, -b / a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_matches_python_and_inverts() {
        let adj = LinearAdjust { gain: 1.3, contrast: 0.7, pivot: 0.18 };
        let (a, b) = adj.affine();
        let (ia, ib) = adj.inverse_affine();
        for x in [-0.5, 0.0, 0.18, 0.9, 12.0] {
            let python = adj.gain * (adj.pivot + adj.contrast * (x - adj.pivot));
            assert!((a * x + b - python).abs() < 1e-12);
            let inverse = (python / adj.gain - adj.pivot) / adj.contrast + adj.pivot;
            assert!((ia * python + ib - inverse).abs() < 1e-12);
            assert!((inverse - x).abs() < 1e-12);
        }
        assert_eq!(LinearAdjust::default().affine(), (1.0, 0.0));
    }
}
