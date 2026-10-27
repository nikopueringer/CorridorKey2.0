//! Filling a buffer with one value on the device.

use cubecl::prelude::*;

use crate::core::CkFloat;

/// `out[i] = value` for `i < len`. One unit per element.
#[cube(launch)]
pub fn fill_kernel<F: CkFloat>(out: &mut [F], value: F, len: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < len {
        out[i as usize] = value;
    }
}
