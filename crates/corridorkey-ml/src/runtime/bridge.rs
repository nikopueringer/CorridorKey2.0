//! Zero-copy bridge between [`corridorkey_core::ComputeTarget`], CubeCL [`Handle`]s and Burn
//! [`Tensor`]s (`burn_cubecl::CubeBackend`).
//!
//! Both [`corridorkey_base`] kernels and Burn's [`CubeBackend`] execute on the same CubeCL
//! [`Client`] and allocate from the same device pool. Converting a `Handle` into a `Tensor<D>` or
//! extracting a contiguous `CubeTensor` from a `Tensor<D>` only exchanges metadata (`Handle` +
//! `Shape`), with zero device-to-device or host-to-device copies.

use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{CubeBackend, CubeDevice};
use burn_tensor::{DType, Device as BurnDevice, Shape, Tensor};
use corridorkey_base::core::CkFloat;
use corridorkey_base::core::buffer::{download, empty, upload};
use corridorkey_core::compute::open_device;
use corridorkey_core::{CkError, CkResult, ComputeTarget};
use cubecl::prelude::{Client, Scalar};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use corridorkey_core::compute::bench::sync;
use cubecl::device::DeviceId;
use cubecl::server::Handle;

/// Shared execution context for a [`ComputeTarget`]: owns the CubeCL [`Client`] (registered with
/// `corridorkey_core`'s exit-shutdown guard) and the matching [`BurnDevice`].
#[derive(Clone)]
pub struct MlContext {
    target: ComputeTarget,
    cube_device: CubeDevice,
    client: Client,
    burn_device: BurnDevice,
    flex32_matmul: bool,
    op_profile: Option<OpProfile>,
}

/// Accumulated device time per named operation (see [`MlContext::with_op_profile`]).
type OpProfile = Arc<Mutex<BTreeMap<&'static str, (Duration, usize)>>>;

impl std::fmt::Debug for MlContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlContext")
            .field("target", &self.target)
            .field("runtime", &self.client.name())
            .finish()
    }
}

impl MlContext {
    /// Opens the default device for `target` and initializes both CubeCL and Burn handles on it.
    pub fn new(target: ComputeTarget) -> CkResult<Self> {
        let cube_device = target.device()?;
        let client = open_device(cube_device.clone());
        let burn_device = BurnDevice::from(cube_device.clone());
        Ok(Self {
            target,
            cube_device,
            client,
            burn_device,
            flex32_matmul: false,
            op_profile: None,
        })
    }

    /// Enables or disables accelerated `DType::Flex32` mode for GEMM operations on this context.
    /// When enabled, `f32` inputs to `matmul` are staged to Tensor Cores (`Cmma`/`Mma`) on GPUs
    /// where native FP32 tensor cores are absent (e.g. Turing `sm_75`, WebGPU).
    pub fn with_flex32_matmul(mut self, enabled: bool) -> Self {
        self.flex32_matmul = enabled;
        self
    }

    /// Element type for fast inference on this device: `F16` on GPU runtimes that store and
    /// compute f16 (CUDA, HIP, wgpu with `shader-f16`), `F32` elsewhere (the CPU runtime, devices
    /// without f16). f16 rather than bf16 on every architecture: Stage 1/2 activations stay within
    /// |x| ≤ 13.7, so f16's 3 extra mantissa bits are free accuracy, and on Turing bf16 has no
    /// hardware path at all. Measured on the RTX 2070 (768×512): f16 2.8× faster than f32 on CUDA,
    /// 1.6× on wgpu, alpha error vs PyTorch fp32 below the product's own bf16 error.
    pub fn preferred_dtype(&self) -> DType {
        let f16 = <half::f16 as Scalar>::elem_type_native();
        if self.target != ComputeTarget::Cpu && self.client.properties().supports_type(f16) {
            DType::F16
        } else {
            DType::F32
        }
    }

    /// Records device time per [`MlContext::op`] span. Every span syncs the device before and
    /// after, which serialises the queue: use for attribution, not for end-to-end timing.
    pub fn with_op_profile(mut self, enabled: bool) -> Self {
        self.op_profile = enabled.then(|| Arc::new(Mutex::new(BTreeMap::new())));
        self
    }

    /// Runs `f` as the named operation: plain call when profiling is off, else synced and timed.
    pub fn op<T>(&self, name: &'static str, f: impl FnOnce() -> T) -> T {
        let Some(profile) = &self.op_profile else {
            return f();
        };
        sync(&self.client);
        let start = Instant::now();
        let out = f();
        sync(&self.client);
        let mut table = profile.lock().unwrap_or_else(|e| e.into_inner());
        let entry = table.entry(name).or_insert((Duration::ZERO, 0));
        entry.0 += start.elapsed();
        entry.1 += 1;
        out
    }

    /// `(name, total time, calls)` recorded so far, longest first; empties the table.
    pub fn take_op_profile(&self) -> Vec<(&'static str, Duration, usize)> {
        let Some(profile) = &self.op_profile else {
            return Vec::new();
        };
        let mut table = profile.lock().unwrap_or_else(|e| e.into_inner());
        let mut rows: Vec<_> = std::mem::take(&mut *table)
            .into_iter()
            .map(|(n, (d, c))| (n, d, c))
            .collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        rows
    }

    /// Whether `DType::Flex32` accelerated matmul mode is enabled.
    pub fn flex32_matmul(&self) -> bool {
        self.flex32_matmul
    }

    /// Executes `a.matmul(b)` on this context. With `flex32_matmul` and `f32` operands, the
    /// operands are relabelled `DType::Flex32` so cubek stages them through `f16` tensor-core
    /// tiles (`f32` accumulate).
    ///
    /// `Tensor::cast(DType::Flex32)` cannot do this: Burn's cast treats Flex32 as F32 and returns
    /// the input unchanged, still labelled `F32`, so cubek never saw Flex32 (measured: bit-identical
    /// output). Flex32 shares the `f32` storage, so relabelling the handles copies nothing.
    ///
    /// Not usable yet (2026-10-26, CubeCL 0.11.3): on CUDA, Flex32 kernels panic in cubecl-llvm
    /// (`to_llvm/constant.rs`, no LLVM type for flex32 constants) and the autotuner keeps a
    /// candidate that returns garbage (Stage 2 alpha mean error 0.7); wgpu/Vulkan has no f16 CMMA
    /// here, so nothing changes. The fast path is native `f16` storage instead.
    pub fn matmul<const D: usize>(&self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
        if self.flex32_matmul && a.dtype() == DType::F32 && b.dtype() == DType::F32 {
            let out = with_dtype(a, DType::Flex32).matmul(with_dtype(b, DType::Flex32));
            with_dtype(out, DType::F32)
        } else {
            a.matmul(b)
        }
    }

    /// Context from `CORRIDORKEY_RUNTIME` or the first compiled-in runtime.
    pub fn from_env() -> CkResult<Self> {
        Self::new(ComputeTarget::from_env()?)
    }

    pub fn target(&self) -> ComputeTarget {
        self.target
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn cube_device(&self) -> &CubeDevice {
        &self.cube_device
    }

    pub fn burn_device(&self) -> &BurnDevice {
        &self.burn_device
    }

    /// Identifies this context's device, for caches of device buffers: a handle is only valid on
    /// the device that allocated it, and model parts (e.g. RoPE tables) may be shared between
    /// contexts on different runtimes.
    pub fn device_id(&self) -> DeviceId {
        self.cube_device.to_id()
    }

    /// Wraps an existing contiguous CubeCL [`Handle`] into a Burn [`Tensor<D>`] without copying.
    pub fn tensor_from_handle<const D: usize>(
        &self,
        handle: Handle,
        shape: [usize; D],
        dtype: DType,
    ) -> Tensor<D> {
        let cube = CubeTensor::new_contiguous(
            self.client.clone(),
            self.cube_device.clone(),
            Shape::from(shape),
            handle,
            dtype,
        );
        Tensor::<D>::from_primitive::<CubeBackend>(cube)
    }

    /// Allocates an uninitialized contiguous `Tensor<D>` of `F` on this device.
    pub fn empty_tensor<F: CkFloat, const D: usize>(&self, shape: [usize; D]) -> Tensor<D> {
        let len: usize = shape.iter().product();
        let handle = empty::<F>(&self.client, len);
        self.tensor_from_handle(handle, shape, elem_dtype::<F>())
    }

    /// Uploads a host slice `&[F]` into a contiguous `Tensor<D>` on this device.
    pub fn upload_tensor<F: CkFloat, const D: usize>(
        &self,
        host: &[F],
        shape: [usize; D],
    ) -> CkResult<Tensor<D>> {
        let expected: usize = shape.iter().product();
        if host.len() != expected {
            return Err(CkError::Shape(format!(
                "upload_tensor: slice len {} != shape {shape:?} ({expected})",
                host.len()
            )));
        }
        let handle = upload(&self.client, host);
        Ok(self.tensor_from_handle(handle, shape, elem_dtype::<F>()))
    }

    /// Uploads a row-major `[rows, cols]` host matrix (PyTorch `nn.Linear` weight layout) and
    /// returns it transposed as a contiguous `[cols, rows]` tensor. The transpose runs once on the
    /// device: the host transpose it replaces was column-strided (~0.5 GB/s, ~5 s of the 2.66 GB
    /// checkpoint load); the resulting buffer is laid out exactly as before.
    pub fn upload_transposed<F: CkFloat>(
        &self,
        host: &[F],
        rows: usize,
        cols: usize,
    ) -> CkResult<Tensor<2>> {
        let stored = self.upload_tensor(host, [rows, cols])?;
        Ok(from_cube(into_contiguous_cube(stored.transpose())?))
    }

    /// Downloads a `Tensor<D>` to a host `Vec<F>` (materializing a contiguous buffer on device if
    /// the tensor is a strided view).
    pub fn download_tensor<F: CkFloat, const D: usize>(
        &self,
        tensor: Tensor<D>,
    ) -> CkResult<Vec<F>> {
        let cube = into_contiguous_cube(tensor)?;
        Ok(download(&cube.client, cube.handle))
    }
}

/// Maps `F: CkFloat` to its Burn [`DType`].
pub fn elem_dtype<F: CkFloat>() -> DType {
    match size_of::<F>() {
        4 => DType::F32,
        8 => DType::F64,
        2 => DType::F16,
        other => unreachable!("unsupported CkFloat byte width {other}"),
    }
}

/// Extracts the underlying [`CubeTensor`] from a Burn [`Tensor<D>`] (zero-copy).
pub fn into_cube<const D: usize>(tensor: Tensor<D>) -> CkResult<CubeTensor> {
    tensor
        .try_into_primitive::<CubeBackend>()
        .map_err(|e| CkError::InvalidConfig(format!("expected CubeBackend tensor: {e:?}")))
}

/// Extracts a contiguous [`CubeTensor`] from a Burn [`Tensor<D>`]. No-op when already contiguous.
pub fn into_contiguous_cube<const D: usize>(tensor: Tensor<D>) -> CkResult<CubeTensor> {
    Ok(into_contiguous(into_cube(tensor)?))
}

/// Relabels the element type of a tensor whose storage `dtype` shares (`F32` ↔ `Flex32`), without
/// a kernel or a copy.
fn with_dtype<const D: usize>(tensor: Tensor<D>, dtype: DType) -> Tensor<D> {
    let mut cube = into_cube(tensor).expect("MlContext tensors live on the CubeBackend");
    cube.dtype = dtype;
    from_cube(cube)
}

/// Re-wraps a [`CubeTensor`] into a Burn [`Tensor<D>`] (zero-copy).
pub fn from_cube<const D: usize>(cube: CubeTensor) -> Tensor<D> {
    Tensor::<D>::from_primitive::<CubeBackend>(cube)
}

#[cfg(test)]
mod tests {
    use super::*;
    use corridorkey_core::ColorSpace;
    use corridorkey_core::LinearAdjust;

    #[test]
    fn shares_handles_between_base_kernels_and_burn_without_copy() {
        for target in ComputeTarget::available() {
            let ctx = MlContext::new(target).unwrap();
            let input: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
            let t = ctx.upload_tensor(&input, [16, 4]).unwrap();

            // Extract raw CubeCL handle, run a corridorkey-base kernel in place, and continue in Burn
            let cube = into_contiguous_cube(t).unwrap();
            corridorkey_base::color::convert::<f32>(
                &cube.client,
                &cube.handle,
                &cube.handle,
                64,
                ColorSpace::Srgb,
                ColorSpace::Linear,
                LinearAdjust::default(),
            );
            let t_linear = from_cube::<2>(cube);
            let scaled = t_linear * 2.0;
            let host = ctx.download_tensor::<f32, 2>(scaled).unwrap();
            assert_eq!(host.len(), 64);
            assert!((host[0] - 0.0).abs() < 1e-6);
            assert!(host[63] > 1.8);
        }
    }

    /// `MlContext::matmul` at Stage 2 shapes (`[tokens, dim] @ [dim, hidden]`, odd token count)
    /// against an f64 host reference. Flex32 is not exercised: on CUDA it panics in cubecl-llvm
    /// 0.11.3 (see `MlContext::matmul`).
    #[test]
    fn matmul_matches_f64_reference_at_model_shapes() {
        let (m, k, n) = (961, 320, 800);
        let mut seed = 0x2545_f491_u32;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) - 0.5
        };
        let a_host: Vec<f32> = (0..m * k).map(|_| rand()).collect();
        let b_host: Vec<f32> = (0..k * n).map(|_| rand() * 0.1).collect();
        let mut want = vec![0.0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                let a = a_host[i * k + p] as f64;
                for j in 0..n {
                    want[i * n + j] += a * b_host[p * n + j] as f64;
                }
            }
        }
        let scale = want.iter().fold(0.0f64, |s, v| s.max(v.abs()));
        for target in ComputeTarget::available() {
            let ctx = MlContext::new(target).unwrap();
            let a = ctx.upload_tensor(&a_host, [m, k]).unwrap();
            let b = ctx.upload_tensor(&b_host, [k, n]).unwrap();
            let got: Vec<f32> = ctx.download_tensor(ctx.matmul(a, b)).unwrap();
            let max_err = got
                .iter()
                .zip(&want)
                .fold(0.0f64, |e, (&g, &w)| e.max((g as f64 - w).abs()));
            assert!(
                max_err / scale < 1e-5,
                "{target}: {max_err:.3e} vs {scale:.3e}"
            );
        }
    }
}
