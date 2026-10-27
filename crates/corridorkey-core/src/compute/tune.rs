//! Helpers for CubeCL autotuning of launch settings.
//!
//! Settings that trade work per unit against parallelism (samples scanned per unit, time blocks of
//! a recurrence) have no device-independent best value. Kernels offer a range of candidates and
//! CubeCL's autotuner benchmarks them once per device and problem-size key, then reuses the
//! fastest (`cubecl::tune::LocalTuner`).

use cubecl::prelude::*;

/// Autotune id of the device behind `client`: runtime name plus the hardware properties that shape
/// launches, so different devices keep separate tuning results.
pub fn tune_id(client: &Client) -> String {
    let hw = &client.properties().hardware;
    format!(
        "{}-plane{}-units{}-cores{}-sm{}",
        client.name(),
        hw.plane_size_max,
        hw.max_units_per_cube,
        hw.num_cpu_cores.unwrap_or(0),
        hw.num_streaming_multiprocessors.unwrap_or(0)
    )
}

/// Problem sizes grouped by power of two, so similar sizes share one tuning result.
pub fn size_class(n: usize) -> usize {
    n.max(1).next_power_of_two()
}
