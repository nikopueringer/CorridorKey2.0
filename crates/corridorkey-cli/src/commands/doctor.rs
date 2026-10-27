//! `doctor`: what this build can do on this machine, as facts (no fixed text).

use std::path::Path;

use cubecl::prelude::*;
use corridorkey_core::compute::{ComputeTarget, ComputeTask};

use super::assets::AssetPaths;

/// The device a runtime opens, described from its reported properties.
struct Describe;

impl ComputeTask for Describe {
    type Output = String;
    fn run(self, client: Client) -> String {
        let hw = &client.properties().hardware;
        let mut parts = vec![
            format!("plane {}", hw.plane_size_max),
            format!("≤ {} units per cube", hw.max_units_per_cube),
        ];
        parts.extend(hw.num_streaming_multiprocessors.map(|n| format!("{n} multiprocessors")));
        parts.extend(hw.num_cpu_cores.map(|n| format!("{n} cores")));
        format!("{} ({})", client.name(), parts.join(", "))
    }
}

/// Cargo features of this build that change what it can do.
const FEATURES: [(&str, bool); 4] = [
    ("wgpu", cfg!(feature = "wgpu")),
    ("cpu", cfg!(feature = "cpu")),
    ("cuda", cfg!(feature = "cuda")),
    ("hip", cfg!(feature = "hip")),
];

pub fn run(root: &Path, selected: Option<ComputeTarget>) -> anyhow::Result<()> {
    println!("corridorkey {} ({} / {})", env!("CARGO_PKG_VERSION"), std::env::consts::OS, std::env::consts::ARCH);
    let on: Vec<&str> = FEATURES.iter().filter(|(_, on)| *on).map(|(name, _)| *name).collect();
    println!("Features:   {}", on.join(", "));

    println!("Runtimes (compiled in, in preference order):");
    for target in ComputeTarget::available() {
        // A runtime without a device (no GPU, no driver) panics when opening its client
        let opened = std::panic::catch_unwind(|| target.run(Describe));
        match opened {
            Ok(Ok(device)) => println!("  {:<5} usable: {device}", target.name()),
            Ok(Err(e)) => println!("  {:<5} not usable: {e}", target.name()),
            Err(_) => println!("  {:<5} not usable: no device", target.name()),
        }
    }
    if let Some(selected) = selected {
        println!("Selected:   {selected} (--runtime, {} or the first compiled in)", corridorkey_core::compute::RUNTIME_ENV);
    }

    println!("Package:    {}", root.display());
    let paths = AssetPaths::new(root);
    for (name, path) in paths.all() {
        println!("  {:<20} {}", name, if path.exists() { "present" } else { "missing" });
    }
    let env = [".venv", "venv"].into_iter().find(|v| root.join(v).is_dir());
    println!("Python env: {}", env.unwrap_or("none (run `uv sync`)"));
    Ok(())
}
