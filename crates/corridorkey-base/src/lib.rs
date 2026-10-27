//! Image kernels for CorridorKey on the device (any GPU through wgpu, CUDA or HIP, or the CPU),
//! written as CubeCL kernels: colour transfer functions, frame preparation (upload, pad, the
//! model and backbone inputs), and (to come) despill and compositing.
//!
//! # Where things are
//!
//! | Need | Module | Start with |
//! |---|---|---|
//! | sRGB / linear / asinh | [`color`] | [`color::convert`] |
//! | Decoded frames → network inputs | [`frame`] | [`frame::frame_views`] |
//! | Device buffers and shared kernels | [`core`] | [`core::buffer`] |
//!
//! Functions take a cubecl `Client`, from [`corridorkey_core::compute::ComputeTarget`]:
//! `ComputeTarget::from_env()?.client()?` picks the runtime named by `CORRIDORKEY_RUNTIME`,
//! otherwise the first compiled in. Buffers stay on the device between calls; only the caller
//! decides when to download.

#[cfg(test)]
#[macro_use]
mod test_support;

pub mod color;
pub mod core;
pub mod frame;
