//! Size limits of device buffers.
//!
//! Kernels index buffers with 32-bit integers: WebGPU (WGSL) has no 64-bit integers, and one
//! kernel source serves every runtime. A buffer, and so one window or batch, holds at most
//! [`MAX_DEVICE_ELEMENTS`] values, e.g. 64 frames of 4K × 8 channels. Past that an index
//! would wrap silently and read or write the wrong element; sizes are checked where they are
//! chosen instead.

use crate::{CkError, CkResult};

/// Most elements one device buffer may hold (`u32::MAX`).
pub const MAX_DEVICE_ELEMENTS: usize = u32::MAX as usize;

/// The product of `dims` as an element count, or an error naming `what` when it exceeds
/// [`MAX_DEVICE_ELEMENTS`] (or overflows).
pub fn device_elements(what: &str, dims: &[usize]) -> CkResult<usize> {
    let total = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
    match total {
        Some(n) if n <= MAX_DEVICE_ELEMENTS => Ok(n),
        _ => Err(CkError::InvalidConfig(format!(
            "{what} {dims:?} needs more than {MAX_DEVICE_ELEMENTS} elements in one device buffer \
             (kernels use 32-bit indices): use smaller batches"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_refuses_oversized_buffers() {
        assert_eq!(device_elements("window", &[384, 60_000]).unwrap(), 384 * 60_000);
        assert!(device_elements("window", &[384, 12_000_000]).is_err());
        assert!(device_elements("window", &[usize::MAX, 2]).is_err(), "overflow is refused too");
    }
}
