//! Overlapping temporal windows blended with a Hann taper.
//!
//! Stage 1 runs on `chunk` frames at a time, windows starting every `stride` frames (the last one
//! moved back to end at the clip's end). Where windows overlap, each frame's result is the
//! weighted average of the windows that cover it, the weights rising and falling over `overlap`
//! frames (a raised cosine), so window seams do not show.

use std::ops::Range;

use crate::{CkError, CkResult};

/// One window of a [`WindowSchedule`].
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalWindow {
    pub index: usize,
    /// Frames of the clip the window covers.
    pub frames: Range<usize>,
    pub is_first: bool,
    pub is_last: bool,
}

/// The windows covering a clip.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowSchedule {
    pub total_frames: usize,
    pub chunk: usize,
    pub stride: usize,
    /// Frames over which the taper weights ramp at a window's inner edges.
    pub overlap: usize,
    windows: Vec<TemporalWindow>,
}

impl WindowSchedule {
    pub fn new(total_frames: usize, chunk: usize, stride: usize, overlap: usize) -> CkResult<Self> {
        if chunk == 0 || stride == 0 {
            return Err(CkError::InvalidConfig(format!("chunk ({chunk}) and stride ({stride}) must be positive")));
        }
        if stride > chunk {
            return Err(CkError::InvalidConfig(format!("stride {stride} > chunk {chunk} would skip frames")));
        }
        // `window_starts`
        let starts: Vec<usize> = if total_frames <= chunk {
            vec![0]
        } else {
            let last = total_frames - chunk;
            let mut s: Vec<usize> = (0..=last).step_by(stride).collect();
            if *s.last().unwrap() != last {
                s.push(last);
            }
            s
        };
        let n = starts.len();
        let windows = starts
            .into_iter()
            .enumerate()
            .map(|(index, start)| TemporalWindow {
                index,
                frames: start..(start + chunk).min(total_frames),
                is_first: index == 0,
                is_last: index + 1 == n,
            })
            .collect();
        Ok(Self { total_frames, chunk, stride, overlap, windows })
    }

    pub fn windows(&self) -> &[TemporalWindow] {
        &self.windows
    }

    pub fn len(&self) -> usize {
        self.windows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Blend weight of each frame of `window` (`taper_weights`): 1 inside, a raised cosine over
    /// `overlap` frames at edges shared with a neighbouring window.
    pub fn weights(&self, window: &TemporalWindow) -> Vec<f64> {
        let len = window.frames.len();
        let mut w = vec![1.0; len];
        if self.overlap == 0 || len <= 1 {
            return w;
        }
        let n = self.overlap.min(len);
        let ramp = |i: usize| 0.5 * (1.0 - (std::f64::consts::PI * (i + 1) as f64 / (self.overlap + 1) as f64).cos());
        for i in 0..n {
            if !window.is_first {
                w[i] = ramp(i);
            }
            if !window.is_last {
                w[len - 1 - i] = ramp(i);
            }
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_match_python_and_cover_the_clip() {
        let s = WindowSchedule::new(100, 32, 24, 8).unwrap();
        let starts: Vec<usize> = s.windows().iter().map(|w| w.frames.start).collect();
        assert_eq!(starts, vec![0, 24, 48, 68], "last window moved back to end at the clip's end");
        assert_eq!(WindowSchedule::new(10, 32, 24, 8).unwrap().windows()[0].frames, 0..10);
        let mut covered = vec![0.0; 100];
        for w in s.windows() {
            for (f, wt) in w.frames.clone().zip(s.weights(w)) {
                covered[f] += wt;
            }
        }
        assert!(covered.iter().all(|&c| c > 0.0), "every frame has weight");
        assert!(WindowSchedule::new(10, 4, 5, 0).is_err());
    }

    #[test]
    fn taper_matches_python() {
        let s = WindowSchedule::new(100, 32, 24, 8).unwrap();
        let w = s.weights(&s.windows()[1]);
        // 0.5 · (1 − cos(π · (i + 1) / 9))
        for i in 0..8 {
            let expected = 0.5 * (1.0 - (std::f64::consts::PI * (i + 1) as f64 / 9.0).cos());
            assert!((w[i] - expected).abs() < 1e-15 && (w[31 - i] - expected).abs() < 1e-15);
        }
        assert!(w[8..24].iter().all(|&x| x == 1.0));
        assert_eq!(s.weights(&s.windows()[0])[0], 1.0, "first window starts at full weight");
    }
}
