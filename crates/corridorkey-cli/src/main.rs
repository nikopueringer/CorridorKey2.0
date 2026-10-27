//! `corridorkey`: a thin command line over the corridorkey crates. Each command lives in its own
//! module with its own arguments; this file only parses and dispatches.

mod commands;
mod progress;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use corridorkey_core::ComputeTarget;

/// Environment variable naming the package root (where `weights/` and `vfms/` live).
pub const ROOT_ENV: &str = "CORRIDORKEY_ROOT";

#[derive(Parser, Debug)]
#[command(name = "corridorkey", version, about = "CorridorKey 2.0 from the command line: devices, model assets, inference")]
struct Cli {
    /// Package root holding `weights/`, `vfms/` and the Python runtime
    #[arg(long, global = true, env = ROOT_ENV, default_value = ".")]
    root: PathBuf,

    /// Compute runtime: wgpu, cpu, cuda or hip, among those compiled in (default: the first
    /// compiled in, GPUs first; `doctor` shows which work on this machine)
    #[arg(long, global = true, env = corridorkey_core::compute::RUNTIME_ENV)]
    runtime: Option<String>,

    /// Show debug logs (runtimes, downloads)
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// What this build can do here: version, features, runtimes, model assets
    Doctor,
    /// Model assets: download, check, convert
    Weights(commands::weights::WeightsArgs),
    /// Key a clip (video file or frame directory) and write alpha / foreground / previews
    Infer(commands::infer::InferArgs),
    /// Time the Rust engine and read its peak GPU memory (one configuration, or a sweep)
    Bench(commands::bench::BenchArgs),
    /// Guided run: asks for the clip and options in the terminal
    Interactive(commands::interactive::InteractiveArgs),
}

impl Cli {
    /// The runtime asked for (checked to be compiled in), else the first compiled-in one.
    fn target(&self) -> anyhow::Result<ComputeTarget> {
        Ok(match &self.runtime {
            Some(name) => ComputeTarget::parse(name)?.checked()?,
            None => ComputeTarget::from_env()?,
        })
    }
}

/// Log level when `--verbose` is not given: warnings only, so runtime chatter (e.g. GPU adapter
/// details) does not bury the command's output.
const DEFAULT_LOG_LEVEL: tracing::Level = tracing::Level::WARN;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt().with_max_level(if cli.verbose { tracing::Level::DEBUG } else { DEFAULT_LOG_LEVEL }).init();
    let root = commands::assets::package_root(&cli.root)?;
    match &cli.command {
        Command::Doctor => commands::doctor::run(&root, cli.target().ok()),
        Command::Weights(args) => commands::weights::run(&root, args),
        Command::Infer(_) | Command::Bench(_) => {
            // Work is done and outputs closed either way: end the process now rather than tearing
            // the GPU runtimes down (driver threads race process teardown, dsp-kitchen lessons §7;
            // autotuners keep clients alive, so a device shutdown would wait out its timeout).
            let result = match &cli.command {
                Command::Infer(args) => commands::infer::run(&root, cli.target(), args),
                Command::Bench(args) => cli.target().and_then(|t| commands::bench::run(&root, t, args)),
                _ => unreachable!(),
            };
            let code = match result {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("Error: {e:?}");
                    1
                }
            };
            corridorkey_core::compute::exit_now(code)
        }
        Command::Interactive(args) => commands::interactive::run(&root, args),
    }
}
