//! Temporal windows: how a clip is cut into overlapping chunks for stage 1 and blended back
//! (`window_starts` / `taper_weights` in `video_io.py`).

mod schedule;

pub use schedule::{TemporalWindow, WindowSchedule};
