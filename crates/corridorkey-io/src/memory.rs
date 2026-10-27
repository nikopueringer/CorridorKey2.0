//! GPU memory as the NVIDIA driver reports it, through NVML (NVIDIA's management library, loaded
//! at run time): free memory for choosing the window size, and this process's own usage (context,
//! allocator pools and all) for benchmarks. Covers CUDA and Vulkan (wgpu) processes alike. Without
//! NVML (other vendors, no driver) every query answers `None`.
//!
//! The device is NVML's index 0, the default device of both runtimes on a single-GPU machine.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use nvml_wrapper::Nvml;
use nvml_wrapper::enums::device::UsedGpuMemory;

/// NVML's index of the device queried.
const DEVICE_INDEX: u32 = 0;

/// Free and total memory of the device, in bytes.
pub fn free_and_total() -> Option<(u64, u64)> {
    let nvml = Nvml::init().ok()?;
    let info = nvml.device_by_index(DEVICE_INDEX).ok()?.memory_info().ok()?;
    Some((info.free, info.total))
}

/// Device memory used by process `pid` (compute and graphics contexts), in bytes.
fn process_used(nvml: &Nvml, pid: u32) -> Option<u64> {
    let device = nvml.device_by_index(DEVICE_INDEX).ok()?;
    let mut procs = device.running_compute_processes().unwrap_or_default();
    procs.extend(device.running_graphics_processes().unwrap_or_default());
    // A process with both contexts is listed twice with the same usage: take the largest entry.
    procs
        .iter()
        .filter(|p| p.pid == pid)
        .filter_map(|p| match p.used_gpu_memory {
            UsedGpuMemory::Used(bytes) => Some(bytes),
            UsedGpuMemory::Unavailable => None,
        })
        .max()
}

/// Samples this process's device memory on a thread until [`PeakSampler::stop`].
pub struct PeakSampler {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<Option<u64>>,
}

impl PeakSampler {
    /// Starts sampling every `period`; `None` without NVML.
    pub fn start(period: Duration) -> Option<Self> {
        let nvml = Nvml::init().ok()?;
        let pid = std::process::id();
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let thread = std::thread::Builder::new()
            .name("corridorkey-vram".into())
            .spawn(move || {
                let mut peak = None;
                loop {
                    if let Some(used) = process_used(&nvml, pid) {
                        peak = Some(peak.map_or(used, |p: u64| p.max(used)));
                    }
                    if flag.load(Ordering::Relaxed) {
                        return peak;
                    }
                    std::thread::sleep(period);
                }
            })
            .ok()?;
        Some(Self { stop, thread })
    }

    /// Stops sampling; the peak seen in bytes (`None` if the driver never listed the process).
    pub fn stop(self) -> Option<u64> {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.join().ok().flatten()
    }
}
