//! Where the model assets come from and where they live (`scripts/download_weights.py`,
//! `RuntimePaths` in `corridorkey/runtime.py`).
//!
//! CorridorKey's own weights come from its Hugging Face repository; the three frozen foundation
//! models from their publishers at pinned revisions, under their own licenses (see
//! `THIRD_PARTY_NOTICES.md`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub const CORRIDORKEY_REPO: &str = "nikopueringer/CorridorKey2.0";
pub const CORRIDORKEY_FILES: [&str; 2] = ["corridorkey_v2.safetensors", "foundation_stats.safetensors"];

/// C-RADIOv4-SO400M (NVIDIA, NVIDIA Open Model License).
pub const CRADIO_REPO: &str = "nvidia/C-RADIOv4-SO400M";
pub const CRADIO_REVISION: &str = "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8";
/// The half-precision `.pth.tar` duplicates `model.safetensors` and is unused.
pub const CRADIO_SKIP_SUFFIX: &str = ".pth.tar";

/// MoGe-2 ViT-B (Microsoft, MIT).
pub const MOGE_REPO: &str = "Ruicheng/moge-2-vitb-normal";
pub const MOGE_REVISION: &str = "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368";
pub const MOGE_FILE: &str = "model.pt";

/// RVM small (Google DeepMind, CC-BY 4.0): not on Hugging Face, checked by the MD5 its bucket
/// publishes.
pub const RVM_FILE: &str = "pretrain_rvm_small16_256_204031069.npz";
pub const RVM_URL: &str = "https://storage.googleapis.com/representations4d/checkpoints/pretrain_rvm_small16_256_204031069.npz";
pub const RVM_MD5_B64: &str = "ZJb6kj8+tasiJbVeplXu7Q==";

/// Header `format` of a CorridorKey v2 runtime checkpoint.
pub const CHECKPOINT_FORMAT: &str = "corridorkey_v2_runtime_checkpoint";
/// Normalisation vectors in `foundation_stats.safetensors`.
pub const STATS_KEYS: [&str; 6] = ["moge_mean", "moge_std", "rvm_mean", "rvm_std", "cradio_mean", "cradio_std"];

/// The package root, checked to exist, as an absolute path.
pub fn package_root(root: &Path) -> Result<PathBuf> {
    root.canonicalize().with_context(|| format!("package root {} does not exist", root.display()))
}

/// Paths of every runtime asset under a package root (`RuntimePaths.from_package`).
#[derive(Debug, Clone)]
pub struct AssetPaths {
    pub weights: PathBuf,
    pub foundation_stats: PathBuf,
    pub rvm: PathBuf,
    pub moge: PathBuf,
    pub cradio_dir: PathBuf,
}

impl AssetPaths {
    pub fn new(root: &Path) -> Self {
        Self {
            weights: root.join("weights").join(CORRIDORKEY_FILES[0]),
            foundation_stats: root.join("weights").join(CORRIDORKEY_FILES[1]),
            rvm: root.join("vfms/rvm").join(RVM_FILE),
            moge: root.join("vfms/moge").join(MOGE_FILE),
            cradio_dir: root.join("vfms/cradio/C-RADIOv4-SO400M"),
        }
    }

    /// `(name, path)` of each asset.
    pub fn all(&self) -> [(&'static str, &Path); 5] {
        [
            ("CorridorKey weights", &self.weights),
            ("foundation stats", &self.foundation_stats),
            ("RVM", &self.rvm),
            ("MoGe-2", &self.moge),
            ("C-RADIOv4", &self.cradio_dir),
        ]
    }

    /// Assets not on disk.
    pub fn missing(&self) -> Vec<&'static str> {
        self.all().into_iter().filter(|(_, p)| !p.exists()).map(|(n, _)| n).collect()
    }
}
