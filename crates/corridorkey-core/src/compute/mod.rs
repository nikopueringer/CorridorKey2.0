//! The compute layer (`compute` feature): the single entry point through which every crate reaches
//! a CubeCL runtime.
//!
//! - [`ComputeTask`] / [`ComputeTarget::run`](crate::device::ComputeTarget): run generic work on
//!   the selected runtime.
//! - [`LaunchGeometry`]: cube sizes and counts from the runtime's properties.
//! - [`tune`]: CubeCL autotuning for launch settings with no device-independent best value.
//! - [`mod@bench`]: device-synchronised timing.
//!
//! Algorithm crates (`corridorkey-base`, `corridorkey-ml`) take a `Client` and use these; they
//! never inspect or special-case the device.

pub mod bench;
pub mod launch;
pub mod limits;
pub mod pinning;
mod special;
mod target;
pub mod tune;

pub use crate::device::{ComputeError, ComputeTarget, RUNTIME_ENV};
pub use target::{CACHE_DIR_ENV, KERNEL_CACHE_ENV, configure_runtime, open_device, shutdown_devices};
#[cfg(not(target_arch = "wasm32"))]
pub use target::exit_now;
pub use limits::{device_elements, MAX_DEVICE_ELEMENTS};
pub use pinning::{pin_tuned_choices, tuned_choices_pinned, PinnedChoices};
pub use launch::{LaunchGeometry, column_position, pixel_row_position, row_position};
pub use special::{negative_infinity, positive_infinity, NEGATIVE_INFINITY_BITS, POSITIVE_INFINITY_BITS};
pub use target::ComputeTask;
