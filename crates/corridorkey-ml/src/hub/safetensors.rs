//! Streaming and in-memory `.safetensors` reader/writer.
//!
//! [`SafetensorsFile`] reads only the JSON header on open (~80 KiB for `corridorkey_v2.safetensors`)
//! and seeks directly to requested tensor ranges on demand, so loading a single stage or block
//! never reads the full 2.66 GB checkpoint into host memory. [`SafetensorsMap`] provides an
//! in-memory representation for small files (`foundation_stats.safetensors`) and wasm buffers.

use std::any::Any;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use corridorkey_base::core::{CkFloat, cast_f32};
use corridorkey_core::{CkError, CkResult};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::config::CheckpointHeader;

/// Metadata for a single tensor in a `.safetensors` header.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

impl TensorEntry {
    /// Total number of elements (`shape.iter().product()`).
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Byte length (`data_offsets[1] - data_offsets[0]`).
    pub fn byte_len(&self) -> usize {
        self.data_offsets[1].saturating_sub(self.data_offsets[0])
    }
}

/// On-disk `.safetensors` archive that parses its header up front and reads tensor payloads by
/// offset on demand.
#[derive(Debug, Clone)]
pub struct SafetensorsFile {
    path: PathBuf,
    data_start: u64,
    tensors: BTreeMap<String, TensorEntry>,
    metadata: BTreeMap<String, String>,
}

impl SafetensorsFile {
    /// Opens `path` and parses only its header.
    pub fn open(path: impl AsRef<Path>) -> CkResult<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file =
            File::open(&path).map_err(|e| CkError::Io(format!("{}: {e}", path.display())))?;
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)
            .map_err(|e| CkError::Io(format!("{}: {e}", path.display())))?;
        let header_len = u64::from_le_bytes(len_bytes);
        let mut header_buf = vec![0u8; header_len as usize];
        file.read_exact(&mut header_buf)
            .map_err(|e| CkError::Weights(format!("{}: truncated header ({e})", path.display())))?;
        let raw_json: BTreeMap<String, Value> =
            serde_json::from_slice(&header_buf).map_err(|e| {
                CkError::Weights(format!(
                    "{}: invalid safetensors header: {e}",
                    path.display()
                ))
            })?;

        let mut tensors = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        for (name, val) in raw_json {
            if name == "__metadata__" {
                metadata = serde_json::from_value(val).map_err(|e| {
                    CkError::Weights(format!("{}: invalid __metadata__: {e}", path.display()))
                })?;
            } else {
                let entry: TensorEntry = serde_json::from_value(val).map_err(|e| {
                    CkError::Weights(format!("{}: invalid entry '{name}': {e}", path.display()))
                })?;
                tensors.insert(name, entry);
            }
        }

        Ok(Self {
            path,
            data_start: 8 + header_len,
            tensors,
            metadata,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorEntry> {
        &self.tensors
    }

    /// Parses the CorridorKey v2 checkpoint header (`model_config` and `metadata`).
    pub fn checkpoint_header(&self) -> CkResult<CheckpointHeader> {
        CheckpointHeader::from_raw_metadata(&self.metadata)
    }

    /// Looks up a tensor's header entry by name.
    pub fn entry(&self, name: &str) -> CkResult<&TensorEntry> {
        self.tensors.get(name).ok_or_else(|| {
            CkError::Weights(format!(
                "tensor '{name}' not found in {}",
                self.path.display()
            ))
        })
    }

    /// Reads the raw little-endian bytes of tensor `name` by seeking directly to its offset.
    pub fn read_bytes(&self, name: &str) -> CkResult<(Vec<usize>, Vec<u8>)> {
        let entry = self.entry(name)?;
        let mut file = File::open(&self.path)
            .map_err(|e| CkError::Io(format!("{}: {e}", self.path.display())))?;
        let offset = self.data_start + entry.data_offsets[0] as u64;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| CkError::Io(format!("{}: seek failed: {e}", self.path.display())))?;
        let mut buf = vec![0u8; entry.byte_len()];
        file.read_exact(&mut buf).map_err(|e| {
            CkError::Io(format!(
                "{}: read '{name}' failed: {e}",
                self.path.display()
            ))
        })?;
        Ok((entry.shape.clone(), buf))
    }

    /// Reads an `F32` tensor by name as the model's compute type `F` (rounded once, on the host).
    pub fn read_as<F: CkFloat>(&self, name: &str) -> CkResult<(Vec<usize>, Vec<F>)> {
        let (shape, values) = self.read_f32(name)?;
        let values = match (Box::new(values) as Box<dyn Any>).downcast::<Vec<F>>() {
            // `F` is f32: the buffer as read, no per-element pass.
            Ok(same) => *same,
            Err(other) => cast_f32(
                &other
                    .downcast::<Vec<f32>>()
                    .expect("read_f32 returns Vec<f32>"),
            ),
        };
        Ok((shape, values))
    }

    /// Reads an `F32` tensor by name into `(shape, Vec<f32>)`.
    pub fn read_f32(&self, name: &str) -> CkResult<(Vec<usize>, Vec<f32>)> {
        let entry = self.entry(name)?;
        if entry.dtype != "F32" {
            return Err(CkError::Weights(format!(
                "tensor '{name}' in {} has dtype {}, expected F32",
                self.path.display(),
                entry.dtype
            )));
        }
        #[cfg(target_endian = "little")]
        {
            let mut file = File::open(&self.path)
                .map_err(|e| CkError::Io(format!("{}: {e}", self.path.display())))?;
            let offset = self.data_start + entry.data_offsets[0] as u64;
            file.seek(SeekFrom::Start(offset))
                .map_err(|e| CkError::Io(format!("{}: seek failed: {e}", self.path.display())))?;
            let n_floats = entry.byte_len() / 4;
            let mut values = vec![0.0f32; n_floats];
            let byte_slice = unsafe {
                std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, n_floats * 4)
            };
            file.read_exact(byte_slice).map_err(|e| {
                CkError::Io(format!(
                    "{}: read '{name}' failed: {e}",
                    self.path.display()
                ))
            })?;
            Ok((entry.shape.clone(), values))
        }
        #[cfg(not(target_endian = "little"))]
        {
            let (shape, bytes) = self.read_bytes(name)?;
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok((shape, values))
        }
    }
}

/// In-memory collection of named `(shape, Vec<f32>)` tensors for small `.safetensors` files and
/// unit test fixtures.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SafetensorsMap {
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}

impl SafetensorsMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn insert(&mut self, name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "shape {shape:?} expects {expected} elements, got {}",
            data.len()
        );
        self.tensors.insert(name.into(), (shape, data));
    }

    pub fn get(&self, name: &str) -> CkResult<(&[usize], &[f32])> {
        let (shape, data) = self.tensors.get(name).ok_or_else(|| {
            CkError::Weights(format!("tensor '{name}' not found in safetensors map"))
        })?;
        Ok((shape.as_slice(), data.as_slice()))
    }

    pub fn to_bytes(&self) -> CkResult<Vec<u8>> {
        let mut views = Vec::with_capacity(self.tensors.len());
        for (name, (shape, data)) in &self.tensors {
            let raw_bytes: &[u8] = bytemuck::cast_slice(data.as_slice());
            let view = TensorView::new(Dtype::F32, shape.clone(), raw_bytes)
                .map_err(|e| CkError::Weights(format!("TensorView '{name}': {e}")))?;
            views.push((name.as_str(), view));
        }
        safetensors::serialize(views, None)
            .map_err(|e| CkError::Weights(format!("serialize safetensors: {e}")))
    }

    pub fn from_bytes(bytes: &[u8]) -> CkResult<Self> {
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| CkError::Weights(format!("deserialize safetensors: {e}")))?;
        let mut tensors = BTreeMap::new();
        for (key, view) in st.tensors() {
            if view.dtype() != Dtype::F32 {
                return Err(CkError::Weights(format!(
                    "unsupported dtype {:?} for tensor '{key}', expected F32",
                    view.dtype()
                )));
            }
            let values: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            tensors.insert(key, (view.shape().to_vec(), values));
        }
        Ok(Self { tensors })
    }

    pub fn from_file(path: impl AsRef<Path>) -> CkResult<Self> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| CkError::Io(format!("{}: {e}", path.as_ref().display())))?;
        Self::from_bytes(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safetensors_map_round_trip() {
        let mut map = SafetensorsMap::new();
        map.insert("w", vec![2, 3], vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
        let bytes = map.to_bytes().unwrap();
        let back = SafetensorsMap::from_bytes(&bytes).unwrap();
        let (shape, data) = back.get("w").unwrap();
        assert_eq!(shape, &[2, 3]);
        assert_eq!(data, &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
    }
}
