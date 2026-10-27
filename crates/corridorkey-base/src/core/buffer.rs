//! Typed device buffers: sizes come from the element type, never from a literal byte count.

use cubecl::prelude::*;
use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use corridorkey_core::compute::MAX_DEVICE_ELEMENTS;

/// Bytes of `len` elements of `E`, at least one element (runtimes reject empty allocations).
///
/// # Panics
/// If `len` exceeds [`MAX_DEVICE_ELEMENTS`] (kernels index with 32-bit integers): sizes chosen by
/// users are checked earlier with [`corridorkey_core::compute::device_elements`], so this is a backstop
/// against a silent wrap.
pub fn bytes<E: CubeElement>(len: usize) -> usize {
    assert!(len <= MAX_DEVICE_ELEMENTS, "device buffer of {len} elements exceeds the 32-bit index limit ({MAX_DEVICE_ELEMENTS})");
    len.max(1) * size_of::<E>()
}

/// Uninitialized device buffer of `len` elements of `E`.
pub fn empty<E: CubeElement>(client: &Client, len: usize) -> Handle {
    client.empty(bytes::<E>(len))
}

/// Device copy of `data`.
///
/// # Panics
/// If `data` holds more than [`MAX_DEVICE_ELEMENTS`] elements (see [`bytes`]).
pub fn upload<E: CubeElement>(client: &Client, data: &[E]) -> Handle {
    bytes::<E>(data.len());
    // `create` with owned bytes: `create_from_slice` copies through a slower general path
    // (≈2–3× longer for a 92 MB window on wgpu and CUDA; `tests/bench_upload.rs`)
    client.create(Bytes::from_elems(data.to_vec()))
}

/// Writes `data` to the start of the existing device buffer `handle` (which holds at least
/// `data.len()` elements of `E`): no allocation, enqueued after the work already queued on the
/// client's stream, so a buffer reused every window is not overwritten before the previous window's
/// kernels have read it.
pub fn write<E: CubeElement>(client: &Client, handle: &Handle, data: &[E]) {
    bytes::<E>(data.len());
    client.write(handle, Bytes::from_elems(data.to_vec()));
}

/// [`fn@write`] taking ownership of `data`: the bytes move into the transfer without a copy (useful
/// when the buffer was filled on another thread, e.g. a read-ahead loader).
pub fn write_owned<E: CubeElement>(client: &Client, handle: &Handle, data: Vec<E>) {
    bytes::<E>(data.len());
    client.write(handle, Bytes::from_elems(data));
}

/// Device buffer of `len` zeros, filled on the device (cubecl-std's zero kernel): nothing is
/// uploaded.
pub fn zeros<E: CubeElement + Scalar>(client: &Client, len: usize) -> Handle {
    bytes::<E>(len);
    cubecl::std::tensor::TensorHandle::zeros(client, [len.max(1)], E::elem_type_native()).handle
}

/// Device buffer of `len` copies of `value`, filled on the device: nothing is uploaded.
pub fn filled<F: crate::core::CkFloat>(client: &Client, len: usize, value: F) -> Handle {
    let out = empty::<F>(client, len);
    if len > 0 {
        let geom = corridorkey_core::compute::LaunchGeometry::elementwise(client, len);
        unsafe {
            super::kernels::fill_kernel::launch::<F>(
                client,
                geom.cube_count,
                geom.cube_dim,
                BufferArg::from_raw_parts(out.clone(), len),
                value,
                len as u32,
            );
        }
    }
    out
}

/// Host copy of a device buffer of `E` (waits for the queued work that writes it).
pub fn download<E: CubeElement>(client: &Client, handle: Handle) -> Vec<E> {
    E::from_bytes(&client.read_one_unchecked(handle)).to_vec()
}

/// [`download`] that fails when the device reports an error in the work leading to the read. CubeCL
/// logs a failed allocation or launch as a warning and skips that work, so outputs read without this
/// check can be silently wrong (e.g. a window too large for the device memory).
pub fn try_download<E: CubeElement>(client: &Client, handle: Handle) -> corridorkey_core::CkResult<Vec<E>> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| corridorkey_core::CkError::ComputeError(format!("device work failed (out of device memory?): {e:?}")))?;
    Ok(E::from_bytes(&bytes).to_vec())
}

/// Host copy of the first `len` elements of `E` of a device buffer: only those bytes are read, so a
/// buffer sized for the largest batch costs only the current batch's transfer.
pub fn download_prefix<E: CubeElement>(client: &Client, handle: Handle, len: usize) -> Vec<E> {
    let used = handle.size_in_used();
    let want = (len * size_of::<E>()) as u64;
    assert!(want <= used, "{len} elements exceed the buffer's {used} bytes");
    let mut out = download::<E>(client, handle.offset_end(used - want));
    out.truncate(len);
    out
}

/// Host copy of elements `start .. start + len` of `E` of a device buffer: only those bytes are
/// read.
pub fn download_range<E: CubeElement>(client: &Client, handle: Handle, start: usize, len: usize) -> Vec<E> {
    let used = handle.size_in_used();
    let (from, want) = ((start * size_of::<E>()) as u64, (len * size_of::<E>()) as u64);
    assert!(from + want <= used, "elements {start}..{} exceed the buffer's {used} bytes", start + len);
    let mut out = download::<E>(client, handle.offset_start(from).offset_end(used - from - want));
    out.truncate(len);
    out
}

/// `handle` without its last `unused` elements of `E` (a view of a buffer larger than the data).
pub fn truncate<E: CubeElement>(handle: Handle, unused: usize) -> Handle {
    handle.offset_end((unused * size_of::<E>()) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fills_every_element(client: &Client) {
        // Not a multiple of any cube size, so the last cube is partial
        let len = 1_000_003;
        let v: Vec<f32> = download(client, filled::<f32>(client, len, f32::MAX));
        assert_eq!(v.len(), len);
        assert!(v.iter().all(|&x| x == f32::MAX), "{}", client.name());
    }
    runtime_test!(test_filled, fills_every_element);
}
