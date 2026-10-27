//! Running work on a [`ComputeTarget`].
//!
//! Library code takes a [`Client`] (cubecl 0.11 resolves the runtime when the client is made, not
//! through a generic); the CLI and the wasm bindings pick a target once and get its client,
//! or run a [`ComputeTask`] on it.
//!
//! **Devices are shut down at exit.** CubeCL runs each device on a runner thread that nothing joins,
//! so the device is never destroyed and the GPU driver's own worker threads (NVIDIA's shader
//! compiler) were still running while the process exited: an intermittent crash or hang after all
//! the work was done (SIGSEGV in `libnvidia-glvkspirv`, "corrupted double-linked list" in
//! `libnvidia-gpucomp`, with the main thread inside `exit()`). Every device a client is made for is
//! recorded, and an exit handler, registered after each device is opened (so it runs before the
//! handlers of the libraries and layers loaded until then), deals with them. With glibc it ends the
//! process there with its exit status (`_exit`), before any teardown: shutting the devices down
//! raced the driver too, and autotuners keep clients alive so it could not finish anyway. Elsewhere
//! it shuts them down ([`shutdown_devices`]), waiting at most 2 s.

use std::sync::Mutex;

use cubecl::device::{AmdDevice, CpuDevice, CudaDevice, WgpuDevice};
use cubecl::device_handle::DeviceHandle;
use cubecl::prelude::*;
use cubecl::Device;

use crate::device::{ComputeError, ComputeTarget};

/// Work to run on whichever runtime a [`ComputeTarget`] selects.
pub trait ComputeTask {
    type Output;
    fn run(self, client: Client) -> Self::Output;
}

/// Turns off CubeCL's on-disk cache of compiled kernels when set to `0` (on by default).
pub const KERNEL_CACHE_ENV: &str = "CORRIDORKEY_KERNEL_CACHE";
/// Directory of CubeCL's caches (compiled kernels, autotune results) instead of the user cache.
pub const CACHE_DIR_ENV: &str = "CORRIDORKEY_CACHE_DIR";

/// CubeCL's configuration, set once before the first device opens: what CubeCL itself would load
/// (`cubecl.toml` / `burn.toml` up from the working directory, `CUBECL_*` variables), with the
/// compiled-kernel cache on. CubeCL keeps it off by default, so every process compiled every
/// kernel again: 8–16 s before the first frame on the RTX 2070 even with autotune results cached.
/// Kernels are cached per device and driver in CubeCL's environment, next to the autotune results.
/// That environment moves from CubeCL's default (a Cargo project's `target/`, found from the
/// working directory, so an installed command or a Python script started elsewhere began cold
/// every time) to the user cache (`~/.cache/cubecl`), or to [`CACHE_DIR_ENV`]; a `cubecl.toml`
/// that names a place keeps it. [`KERNEL_CACHE_ENV`]`=0` leaves kernel caching off. An
/// application that set CubeCL's configuration first keeps its own (`try_set`).
pub fn configure_runtime() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
            config.compilation.cache = std::env::var(KERNEL_CACHE_ENV).map_or(true, |v| v.trim() != "0");
            use cubecl::config::cache::CacheConfig;
            match std::env::var_os(CACHE_DIR_ENV) {
                Some(dir) if !dir.is_empty() => config.environment.path = CacheConfig::Directory(dir.into()),
                _ if matches!(config.environment.path, CacheConfig::Target) => config.environment.path = CacheConfig::Global,
                _ => {}
            }
            let _ = CubeClRuntimeConfig::try_set(config);
        });
    }
}

impl ComputeTarget {
    /// The default device of this runtime, if it is compiled in.
    pub fn device(self) -> Result<Device, ComputeError> {
        configure_runtime();
        Ok(match self.checked()? {
            Self::Wgpu => Device::Wgpu(WgpuDevice::default()),
            Self::Cpu => Device::Cpu(CpuDevice),
            Self::Cuda => Device::Cuda(CudaDevice::default()),
            Self::Hip => Device::Hip(AmdDevice::default()),
        })
    }

    /// A client for this runtime's default device (recorded for [`shutdown_devices`]).
    pub fn client(self) -> Result<Client, ComputeError> {
        Ok(open_device(self.device()?))
    }

    /// Runs `task` with a client for this runtime's default device.
    pub fn run<T: ComputeTask>(self, task: T) -> Result<T::Output, ComputeError> {
        Ok(task.run(self.client()?))
    }
}

/// Devices a client was made for, shut down at exit.
static OPENED: Mutex<Vec<Device>> = Mutex::new(Vec::new());

// No C exit handlers in the browser: the page owns the WebGPU device
#[cfg(not(target_arch = "wasm32"))]
unsafe extern "C" {
    /// The C library's exit-handler registration (handlers run in reverse order of registration).
    #[cfg_attr(all(target_os = "linux", target_env = "gnu"), allow(dead_code))]
    fn atexit(handler: extern "C" fn()) -> std::ffi::c_int;
}

// glibc's exit-handler registration that also hands the handler the exit status.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
unsafe extern "C" {
    fn on_exit(handler: extern "C" fn(std::ffi::c_int, *mut std::ffi::c_void), arg: *mut std::ffi::c_void) -> std::ffi::c_int;
    fn fflush(stream: *mut std::ffi::c_void) -> std::ffi::c_int;
}

/// How long the exit handler waits for the devices to shut down. A runner only stops once every
/// client is gone, and autotuners keep their tunable sets (which can hold a client) in statics
/// nothing clears: after a run that autotuned, CubeCL's own join would wait its full 30 s.
#[cfg(all(not(target_arch = "wasm32"), not(all(target_os = "linux", target_env = "gnu"))))]
const EXIT_SHUTDOWN_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// Shuts the devices down, waiting at most [`EXIT_SHUTDOWN_WAIT`]; whether they stopped in time.
#[cfg(all(not(target_arch = "wasm32"), not(all(target_os = "linux", target_env = "gnu"))))]
fn shutdown_within_wait() -> bool {
    let (tx, rx) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new().name("corridorkey-shutdown".into()).spawn(move || {
        shutdown_devices();
        let _ = tx.send(());
    });
    spawned.is_ok() && rx.recv_timeout(EXIT_SHUTDOWN_WAIT).is_ok()
}

#[cfg(all(not(target_arch = "wasm32"), not(all(target_os = "linux", target_env = "gnu"))))]
extern "C" fn shutdown_at_exit() {
    shutdown_within_wait();
}

/// With glibc the handler knows the exit status, so the process ends here with it: stdio flushed,
/// then `_exit`, without shutting the devices down or running the remaining C exit handlers and
/// library destructors. Both race NVIDIA's driver threads, which keep compiling and freeing on their
/// own: library teardown crashed at exit (SIGSEGV in `libnvidia-glvkspirv`), and so did a device
/// shutdown here (heap corruption in a `libnvidia-gpucomp` thread while CubeCL tore down the wgpu
/// device's services, a Python process that keyed on wgpu, 2026-10-27, 3 exits in 5). Whatever
/// hosts this library (the command line, Python) has finalized by then.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
extern "C" fn shutdown_on_exit(status: std::ffi::c_int, _arg: *mut std::ffi::c_void) {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: flushes every C stream (null means all), then ends the process with the status `exit`
    // was given; nothing is left to unwind
    unsafe {
        fflush(std::ptr::null_mut());
        c_exit_now(status)
    }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe extern "C" {
    #[link_name = "_exit"]
    fn c_exit_now(status: std::ffi::c_int) -> !;
}

/// Ends the process now, after flushing stdout and stderr: no exit handlers, no library teardown.
/// For a program whose work is done (outputs written and closed): the GPU drivers' teardown
/// racing their own worker threads is what crashed or hung at exit, and the device shutdown
/// above cannot finish while autotuners hold clients.
#[cfg(not(target_arch = "wasm32"))]
pub fn exit_now(code: i32) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: `_exit` takes a status and never returns; nothing is left to unwind
    unsafe { c_exit_now(code) }
}

/// Records `device`, newly opened, and registers [`shutdown_devices`] to run at exit. Called after
/// the client exists: exit handlers run in reverse order of registration, and opening a device
/// loads Vulkan layers that register their own (the validation layer destroys its device records
/// in one), so ours must come later to run first. Registered again for each newly opened device,
/// since a device reopened after a shutdown may load those layers again; running it more than once
/// is harmless (the list is empty after the first run).
fn opened(device: &Device) {
    let mut devices = OPENED.lock().unwrap_or_else(|e| e.into_inner());
    if devices.contains(device) {
        return;
    }
    devices.push(device.clone());
    drop(devices);
    // SAFETY: registers a plain `extern "C"` function; failure (no slot left) only means the
    // device is not shut down at exit, as before
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        on_exit(shutdown_on_exit, std::ptr::null_mut());
    }
    #[cfg(all(not(target_arch = "wasm32"), not(all(target_os = "linux", target_env = "gnu"))))]
    unsafe {
        atexit(shutdown_at_exit);
    }
}

/// The client of `device`, recorded so that [`shutdown_devices`] shuts it down at exit. Use this
/// instead of `Device::client` wherever a specific device is opened (e.g. one GPU of several).
pub fn open_device(device: Device) -> Client {
    configure_runtime();
    let client = device.client();
    opened(&device);
    client
}

/// Shuts down every device a client was made for: their runner threads stop (queued work runs
/// first) and the devices are destroyed, so the GPU driver's threads end before the process does.
/// Runs at exit by itself; call it earlier to release the GPUs. Clients still alive keep their
/// runner going (CubeCL waits up to 30 s for it, then leaves it), so drop them first. A later
/// [`ComputeTarget::client`] opens the device again.
pub fn shutdown_devices() {
    let devices = std::mem::take(&mut *OPENED.lock().unwrap_or_else(|e| e.into_inner()));
    for device in devices {
        // The runner is registered under the runtime's own id, not `Device`'s runtime-stamped one
        DeviceHandle::<()>::shutdown(cubecl::RuntimeId::strip(device.to_id()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RuntimeName;
    impl ComputeTask for RuntimeName {
        type Output = String;
        fn run(self, client: Client) -> String {
            client.name().to_string()
        }
    }

    /// Threads of this process (Linux).
    fn threads() -> usize {
        std::fs::read_dir("/proc/self/task").map(|d| d.count()).unwrap_or(0)
    }

    /// Shutting the devices down stops their runner (and driver) threads; a later client opens the
    /// device again.
    #[test]
    #[cfg(target_os = "linux")]
    fn shutdown_stops_the_device_threads() {
        let Some(target) = ComputeTarget::available().into_iter().find(|t| *t != ComputeTarget::Cpu) else { return };
        let client = target.client().unwrap();
        let data: Vec<u8> = (0..64u8).collect();
        let handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(data.clone()));
        assert_eq!(client.read_one_unchecked(handle).to_vec(), data);
        drop(client);
        let open = threads();
        shutdown_devices();
        let closed = threads();
        assert!(closed < open, "{open} threads with the device open, {closed} after shutdown");
        let again = target.client().unwrap();
        assert!(!again.name().is_empty());
    }

    #[test]
    fn runs_on_available_targets_and_refuses_missing_ones() {
        for target in ComputeTarget::available() {
            assert!(!target.run(RuntimeName).unwrap().is_empty());
        }
        for target in ComputeTarget::ALL.into_iter().filter(|t| !t.is_available()) {
            assert!(matches!(target.run(RuntimeName), Err(ComputeError::Unavailable(..))));
        }
    }
}
