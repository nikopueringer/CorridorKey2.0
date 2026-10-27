//! `weights`: fetch, check and convert the model assets (`scripts/download_weights.py`,
//! `scripts/smoke_test_runtime.py`, `scripts/convert_to_safetensors.py`).

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use serde_json::Value;

use super::assets::*;
use super::download::{Expected, download, hub_files, hub_url, http_client, is_present};
use crate::progress::TerminalProgress;

#[derive(Args, Debug)]
pub struct WeightsArgs {
    #[command(subcommand)]
    command: WeightsCommand,
}

#[derive(Subcommand, Debug)]
enum WeightsCommand {
    /// Download CorridorKey's weights and the foundation models into `weights/` and `vfms/`
    /// (files already present are skipped)
    Download(DownloadArgs),
    /// Check that every asset is present and the checkpoint is a CorridorKey v2 runtime
    /// checkpoint; with --hash, also verify each file against its published checksum
    Check(CheckArgs),
    /// Convert legacy `.pt` assets to `.safetensors` (runs the Python converter: needs torch)
    Convert(ConvertArgs),
}

#[derive(Args, Debug)]
struct DownloadArgs {
    /// Hugging Face repository of CorridorKey's weights
    #[arg(long, default_value = CORRIDORKEY_REPO)]
    repo: String,
    /// Revision (branch, tag or commit) of CorridorKey's weights
    #[arg(long, default_value = "main")]
    revision: String,
    /// Only fetch the third-party foundation models
    #[arg(long)]
    skip_corridorkey: bool,
    /// Hash files already present and download again those that do not match
    #[arg(long)]
    hash: bool,
}

#[derive(Args, Debug)]
struct CheckArgs {
    /// Also verify every file against the checksum its publisher lists (network access; reads
    /// about 5 GB)
    #[arg(long)]
    hash: bool,
    /// Revision of CorridorKey's weights to verify against
    #[arg(long, default_value = "main")]
    revision: String,
}

#[derive(Args, Debug)]
struct ConvertArgs {
    /// Directory holding `corridorkey_v2.pt` and `foundation_stats.pt` (default: `<root>/weights`)
    #[arg(long)]
    weights_dir: Option<PathBuf>,
    /// Where to write the `.safetensors` files (default: the weights directory)
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

pub fn run(root: &Path, args: &WeightsArgs) -> Result<()> {
    match &args.command {
        WeightsCommand::Download(a) => run_download(root, a),
        WeightsCommand::Check(a) => run_check(root, a),
        WeightsCommand::Convert(a) => run_convert(root, a),
    }
}

/// One file to fetch.
struct Asset {
    label: String,
    url: String,
    dest: PathBuf,
    expected: Expected,
}

/// Every file of every asset, with what it must match.
fn plan(client: &reqwest::blocking::Client, root: &Path, repo: &str, revision: &str, corridorkey: bool) -> Result<Vec<Asset>> {
    let mut assets = vec![];
    if corridorkey {
        let files = hub_files(client, repo, revision)?;
        for name in CORRIDORKEY_FILES {
            let file = files.iter().find(|f| f.path == name).with_context(|| format!("{repo}@{revision} has no {name}"))?;
            assets.push(Asset {
                label: name.to_string(),
                url: hub_url(repo, revision, name),
                dest: root.join("weights").join(name),
                expected: file.expected.clone(),
            });
        }
    }
    let paths = AssetPaths::new(root);
    for file in hub_files(client, CRADIO_REPO, CRADIO_REVISION)? {
        if file.path.ends_with(CRADIO_SKIP_SUFFIX) {
            continue;
        }
        assets.push(Asset {
            label: format!("c-radio/{}", file.path),
            url: hub_url(CRADIO_REPO, CRADIO_REVISION, &file.path),
            dest: paths.cradio_dir.join(&file.path),
            expected: file.expected,
        });
    }
    let moge = hub_files(client, MOGE_REPO, MOGE_REVISION)?
        .into_iter()
        .find(|f| f.path == MOGE_FILE)
        .with_context(|| format!("{MOGE_REPO}@{MOGE_REVISION} has no {MOGE_FILE}"))?;
    assets.push(Asset {
        label: format!("moge/{MOGE_FILE}"),
        url: hub_url(MOGE_REPO, MOGE_REVISION, MOGE_FILE),
        dest: paths.moge,
        expected: moge.expected,
    });
    assets.push(Asset {
        label: format!("rvm/{RVM_FILE}"),
        url: RVM_URL.to_string(),
        dest: paths.rvm,
        expected: Expected { md5_b64: Some(RVM_MD5_B64.to_string()), ..Default::default() },
    });
    Ok(assets)
}

fn run_download(root: &Path, args: &DownloadArgs) -> Result<()> {
    let client = http_client()?;
    let assets = plan(&client, root, &args.repo, &args.revision, !args.skip_corridorkey)?;
    let mut todo = vec![];
    for asset in assets {
        if is_present(&asset.dest, &asset.expected, args.hash)? {
            println!("  present  {}", asset.label);
        } else {
            todo.push(asset);
        }
    }
    let progress = TerminalProgress::default();
    for (i, asset) in todo.iter().enumerate() {
        download(&client, &asset.url, &asset.dest, &asset.expected, (i + 1, todo.len(), &asset.label), &progress)?;
    }
    println!("All assets are in place under {}", root.display());
    Ok(())
}

/// Header of a `.safetensors` file: the tensors (name → dtype, shape) and the string metadata.
/// Only the header is read, not the tensors.
fn safetensors_header(path: &Path) -> Result<(BTreeMap<String, (String, Vec<usize>)>, BTreeMap<String, String>)> {
    let mut file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)?;
    let mut header = vec![0u8; u64::from_le_bytes(len) as usize];
    file.read_exact(&mut header).with_context(|| format!("{} has a truncated header", path.display()))?;
    let json: BTreeMap<String, Value> = serde_json::from_slice(&header).with_context(|| format!("{} has no safetensors header", path.display()))?;
    let mut tensors = BTreeMap::new();
    let mut metadata = BTreeMap::new();
    for (name, value) in json {
        if name == "__metadata__" {
            metadata = serde_json::from_value(value)?;
        } else {
            let dtype = value["dtype"].as_str().unwrap_or("?").to_string();
            let shape = serde_json::from_value(value["shape"].clone())?;
            tensors.insert(name, (dtype, shape));
        }
    }
    Ok((tensors, metadata))
}

fn run_check(root: &Path, args: &CheckArgs) -> Result<()> {
    let paths = AssetPaths::new(root);
    for (name, path) in paths.all() {
        println!("  {:<20} {} {}", name, if path.exists() { "ok     " } else { "MISSING" }, path.display());
    }
    let missing = paths.missing();
    if !missing.is_empty() {
        bail!("missing asset(s): {}; run `corridorkey weights download`", missing.join(", "));
    }

    let (tensors, metadata) = safetensors_header(&paths.weights)?;
    let format = metadata.get("format").map(String::as_str).unwrap_or("none");
    if format != CHECKPOINT_FORMAT {
        bail!("{} has format {format:?}, not {CHECKPOINT_FORMAT:?}", paths.weights.display());
    }
    if !metadata.contains_key("model_config") {
        bail!("{} has no model_config in its header", paths.weights.display());
    }
    let params: usize = tensors.values().map(|(_, shape)| shape.iter().product::<usize>()).sum();
    let mut dtypes: BTreeMap<&str, usize> = BTreeMap::new();
    for (dtype, _) in tensors.values() {
        *dtypes.entry(dtype).or_default() += 1;
    }
    println!(
        "checkpoint ok: {} tensors, {:.1} M parameters, dtypes {dtypes:?}, version {}",
        tensors.len(),
        params as f64 / 1e6,
        metadata.get("format_version").map(String::as_str).unwrap_or("?")
    );
    if let Some(meta) = metadata.get("metadata") {
        println!("  metadata: {meta}");
    }

    let (stats, _) = safetensors_header(&paths.foundation_stats)?;
    let absent: Vec<&str> = STATS_KEYS.into_iter().filter(|k| !stats.contains_key(*k)).collect();
    if !absent.is_empty() {
        bail!("{} lacks {}", paths.foundation_stats.display(), absent.join(", "));
    }
    let width = |k: &str| stats[k].1.iter().product::<usize>();
    println!("foundation stats ok: cradio={}, moge={}, rvm={}", width("cradio_mean"), width("moge_mean"), width("rvm_mean"));

    if args.hash {
        let client = http_client()?;
        let mut bad = vec![];
        for asset in plan(&client, root, CORRIDORKEY_REPO, &args.revision, true)? {
            let ok = is_present(&asset.dest, &asset.expected, true)?;
            println!("  {} {}", if ok { "verified" } else { "MISMATCH" }, asset.label);
            if !ok {
                bad.push(asset.label);
            }
        }
        if !bad.is_empty() {
            bail!("{} file(s) do not match their published checksum: {}", bad.len(), bad.join(", "));
        }
    }
    Ok(())
}

fn run_convert(root: &Path, args: &ConvertArgs) -> Result<()> {
    let weights_dir = args.weights_dir.clone().unwrap_or_else(|| root.join("weights"));
    let mut argv = vec!["--weights_dir".into(), weights_dir.display().to_string()];
    if let Some(out) = &args.output_dir {
        argv.extend(["--output_dir".into(), out.display().to_string()]);
    }
    super::python::run_script(root, "scripts/convert_to_safetensors.py", &argv)
}
