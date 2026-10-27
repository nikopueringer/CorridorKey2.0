//! Foundational types, layouts and the compute layer for `corridorkey`.
//!
//! No algorithms live here (kernels are in `corridorkey-base`, networks in `corridorkey-ml`):
//! this crate names the runtime ([`ComputeTarget`]), how tensors sit in memory ([`TensorLayout`]),
//! what a clip is ([`FrameMeta`]) and how it is cut into temporal windows ([`WindowSchedule`]).

pub mod error;
pub mod device;
pub mod layout;
pub mod frame;
pub mod window;
pub mod progress;
#[cfg(feature = "compute")]
pub mod compute;

pub use device::{ComputeError, ComputeTarget};
pub use error::{CkError, CkResult};
pub use layout::{MemoryOrder, SwinWindows, TensorLayout};
pub use frame::{ColorSpace, FrameMeta, FrameSource, LinearAdjust, Padding};
pub use progress::{NoProgress, ProgressEvent, ProgressSink};
pub use window::{TemporalWindow, WindowSchedule};
