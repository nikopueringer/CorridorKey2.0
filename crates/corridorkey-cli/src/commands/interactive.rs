//! `interactive`: the guided run of `run_interactive.py` (asks for the clip and options in the
//! terminal; the drag-and-drop launchers call it).

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Args;

#[derive(Args, Debug)]
pub struct InteractiveArgs {
    /// Clip or frame directory (asked for when omitted)
    pub clip: Option<PathBuf>,
}

pub fn run(root: &Path, args: &InteractiveArgs) -> Result<()> {
    let argv: Vec<String> = args.clip.iter().map(|p| p.display().to_string()).collect();
    super::python::run_script(root, "run_interactive.py", &argv)
}
