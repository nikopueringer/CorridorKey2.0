//! The Python runtime, for what is not ported to Rust yet (inference, the interactive guide, the
//! `.pt` converter). Runs a script of the package with `uv run` (the project's locked
//! environment), else `.venv` or `venv`'s Python, else `python3`.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

/// The command that runs Python in the package's environment.
fn python(root: &Path) -> Command {
    let on_path = |name: &str| std::env::var_os("PATH").is_some_and(|p| std::env::split_paths(&p).any(|d| d.join(name).is_file()));
    if on_path("uv") && root.join("pyproject.toml").is_file() {
        let mut cmd = Command::new("uv");
        cmd.arg("run").arg("--project").arg(root).arg("python");
        return cmd;
    }
    let venv = [".venv", "venv"]
        .iter()
        .map(|v| root.join(v).join(if cfg!(windows) { "Scripts/python.exe" } else { "bin/python" }))
        .find(|p: &PathBuf| p.is_file());
    Command::new(venv.unwrap_or_else(|| PathBuf::from("python3")))
}

/// Runs `script` (relative to the package root) with `args`, from the package root, and fails if
/// it fails.
pub fn run_script(root: &Path, script: &str, args: &[String]) -> Result<()> {
    let path = root.join(script);
    if !path.is_file() {
        bail!("{} not found: is --root the CorridorKey package root?", path.display());
    }
    let mut cmd = python(root);
    cmd.arg(&path).args(args).current_dir(root);
    tracing::debug!(?cmd, "running the Python runtime");
    let status = cmd.status().with_context(|| format!("starting {cmd:?}"))?;
    if !status.success() {
        bail!("{script} failed ({status})");
    }
    Ok(())
}
