//! Device-side building blocks shared by every corridorkey-base kernel.
//!
//! Runtime selection and launch geometry live in [`corridorkey_core::compute`]; this module holds
//! what the kernels of this crate have in common:
//!
//! - [`CkFloat`]: the element type kernels are generic over.
//! - [`buffer`]: allocate, upload and download typed device buffers (sizes from the element type).

pub mod buffer;
pub mod kernels;
mod float;

pub use float::{cast, cast_all, cast_f32, to_f64, CkFloat};
