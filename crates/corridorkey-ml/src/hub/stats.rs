//! Per-channel normalization statistics for the three frozen foundation backbones
//! (`foundation_stats.safetensors`).

use std::path::Path;

use corridorkey_core::{CkError, CkResult};

use super::safetensors::SafetensorsMap;

/// Channel-wise `(mean, std)` vectors for normalizing MoGe, RVM and C-RADIO features at `/64`.
#[derive(Debug, Clone, PartialEq)]
pub struct FoundationStats {
    pub moge_mean: Vec<f32>,
    pub moge_std: Vec<f32>,
    pub rvm_mean: Vec<f32>,
    pub rvm_std: Vec<f32>,
    pub cradio_mean: Vec<f32>,
    pub cradio_std: Vec<f32>,
}

impl FoundationStats {
    /// Loads the six normalization vectors from a `SafetensorsMap`.
    pub fn from_map(map: &SafetensorsMap) -> CkResult<Self> {
        let vec1d = |key: &str| -> CkResult<Vec<f32>> {
            let (_, data) = map.get(key)?;
            if data.is_empty() {
                return Err(CkError::Weights(format!("foundation stat '{key}' is empty")));
            }
            Ok(data.to_vec())
        };
        let stats = Self {
            moge_mean: vec1d("moge_mean")?,
            moge_std: vec1d("moge_std")?,
            rvm_mean: vec1d("rvm_mean")?,
            rvm_std: vec1d("rvm_std")?,
            cradio_mean: vec1d("cradio_mean")?,
            cradio_std: vec1d("cradio_std")?,
        };
        if stats.moge_mean.len() != stats.moge_std.len()
            || stats.rvm_mean.len() != stats.rvm_std.len()
            || stats.cradio_mean.len() != stats.cradio_std.len()
        {
            return Err(CkError::Weights("foundation stats mean/std length mismatch".into()));
        }
        Ok(stats)
    }

    /// Loads `foundation_stats.safetensors` from disk.
    pub fn from_file(path: impl AsRef<Path>) -> CkResult<Self> {
        Self::from_map(&SafetensorsMap::from_file(path)?)
    }

    pub fn moge_dim(&self) -> usize {
        self.moge_mean.len()
    }

    pub fn rvm_dim(&self) -> usize {
        self.rvm_mean.len()
    }

    pub fn cradio_dim(&self) -> usize {
        self.cradio_mean.len()
    }
}
