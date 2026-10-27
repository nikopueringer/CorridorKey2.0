//! Device-synchronised timing for kernel benchmarks.
//!
//! Kernel launches are asynchronous; a timing is only meaningful once the client's queue has
//! drained. [`time_device`] syncs before starting the clock and after the timed closure.

use std::time::{Duration, Instant};

use cubecl::prelude::*;

/// Waits until every operation queued on `client` has finished.
pub fn sync(client: &Client) {
    cubecl::future::block_on(client.sync()).expect("device sync");
}

/// Runs `f` once as warm-up (kernel compilation, allocation), then `iters` times, and returns the
/// median wall time per iteration including device completion.
pub fn time_device(client: &Client, iters: usize, mut f: impl FnMut()) -> Duration {
    f();
    sync(client);
    let mut times: Vec<Duration> = (0..iters.max(1))
        .map(|_| {
            let start = Instant::now();
            f();
            sync(client);
            start.elapsed()
        })
        .collect();
    times.sort();
    times[times.len() / 2]
}
