//! Geometry of a clip (`probe_video_meta` in `video_io.py`).

use serde::{Deserialize, Serialize};

use super::LinearAdjust;
use crate::{CkError, CkResult};

/// Multiple the model input is padded to: the runtime's default and the checkpoint header's
/// `shape_quantum` (`InferenceConfig.shape_quantum`). The network itself needs multiples of 64
/// (stage 1 works at /64); read the value from the checkpoint rather than assuming it.
pub const SHAPE_QUANTUM: usize = 128;

/// Padding around the model-native image inside the model input, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Padding {
    pub left: usize,
    pub right: usize,
    pub top: usize,
    pub bottom: usize,
}

/// Sizes of one clip, all `(height, width)` in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrameMeta {
    /// Decoded frame size; outputs are written at this size.
    pub native_hw: (usize, usize),
    /// Native size scaled by `model_scale`, before padding.
    pub model_native_hw: (usize, usize),
    /// Model input size: `model_native_hw` padded up to the shape quantum, centred.
    pub shape_hw: (usize, usize),
    pub pad: Padding,
    pub model_scale: f64,
    pub fps: f64,
    pub num_frames: usize,
    pub linear_adjust: LinearAdjust,
}

impl FrameMeta {
    /// Geometry for `native_hw` frames run at `model_scale`, padded to multiples of `quantum`
    /// (`probe_video_meta`, native mode).
    pub fn new(native_hw: (usize, usize), model_scale: f64, quantum: usize, fps: f64, num_frames: usize) -> CkResult<Self> {
        if quantum == 0 || !quantum.is_multiple_of(64) {
            return Err(CkError::InvalidConfig(format!("shape quantum must be a positive multiple of 64, got {quantum}")));
        }
        if !(model_scale > 0.0) {
            return Err(CkError::InvalidConfig(format!("model scale must be > 0, got {model_scale}")));
        }
        let (h, w) = native_hw;
        // Python's `round` (ties to even)
        let scaled = |n: usize| ((n as f64 * model_scale).round_ties_even() as usize).max(1);
        let (hm, wm) = (scaled(h), scaled(w));
        let up = |n: usize| (n.div_ceil(quantum) * quantum).max(quantum);
        let (ho, wo) = (up(hm), up(wm));
        let (left, top) = ((wo - wm) / 2, (ho - hm) / 2);
        let pad = Padding { left, right: wo - wm - left, top, bottom: ho - hm - top };
        Ok(Self {
            native_hw,
            model_native_hw: (hm, wm),
            shape_hw: (ho, wo),
            pad,
            model_scale,
            fps,
            num_frames,
            linear_adjust: LinearAdjust::default(),
        })
    }

    /// Model input pixels per frame.
    pub fn model_pixels(&self) -> usize {
        self.shape_hw.0 * self.shape_hw.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_to_the_quantum_centred() {
        let m = FrameMeta::new((1080, 1920), 1.0, SHAPE_QUANTUM, 24.0, 10).unwrap();
        assert_eq!(m.shape_hw, (1152, 1920));
        assert_eq!(m.pad, Padding { left: 0, right: 0, top: 36, bottom: 36 });
        let m = FrameMeta::new((1080, 1920), 1.0, 64, 24.0, 10).unwrap();
        assert_eq!(m.shape_hw, (1088, 1920));
        // Odd padding: Python's floor division puts the extra pixel at the bottom / right
        let m = FrameMeta::new((1001, 777), 0.5, 64, 24.0, 1).unwrap();
        assert_eq!(m.model_native_hw, (500, 388));
        assert_eq!(m.shape_hw, (512, 448));
        assert_eq!(m.pad, Padding { left: 30, right: 30, top: 6, bottom: 6 });
        let m = FrameMeta::new((1001, 1001), 0.5, 128, 24.0, 1).unwrap();
        assert_eq!(m.pad, Padding { left: 6, right: 6, top: 6, bottom: 6 }, "ties to even: 500.5 → 500");
        assert!(FrameMeta::new((10, 10), 0.0, 64, 24.0, 1).is_err());
        assert!(FrameMeta::new((10, 10), 1.0, 96, 24.0, 1).is_err());
    }
}
