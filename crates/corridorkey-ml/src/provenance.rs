//! Provenance of CorridorKey v2 and its three frozen foundation models (`THIRD_PARTY_NOTICES.md`).
//!
//! Exposed by every model through [`Attributed`] so weights, code licenses and upstream citations
//! stay attached to the runtime objects that execute them.

use serde::{Deserialize, Serialize};

/// How a model relates to its upstream publication and weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceKind {
    /// Published weights and architecture ported from the upstream repository.
    PortedFromCode,
    /// Pretrained foundation weights run frozen as released.
    UpstreamWeights,
}

/// The publication to cite, when available.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Paper {
    pub title: String,
    pub authors: Vec<String>,
    pub venue: String,
    pub year: u16,
    pub url: String,
}

/// The upstream source code repository.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpstreamCode {
    pub publisher: String,
    pub url: String,
    pub license: String,
    pub revision: String,
}

/// A weight or statistics artifact required at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactSource {
    pub name: String,
    pub url: String,
    pub license: String,
}

/// Complete attribution record for a model component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub name: String,
    pub kind: ProvenanceKind,
    pub paper: Option<Paper>,
    pub code: UpstreamCode,
    #[serde(default)]
    pub artifacts: Vec<ArtifactSource>,
}

impl Provenance {
    /// One-line human-readable citation string.
    pub fn citation(&self) -> String {
        let code = format!(
            "{} ({}, {}, {})",
            self.code.url, self.code.publisher, self.code.revision, self.code.license
        );
        match &self.paper {
            Some(p) => format!("{} ({}). {}. {code}", p.title, p.year, p.venue),
            None => format!("{}: {code}", self.name),
        }
    }
}

/// Any model or stage whose upstream origin can be queried.
pub trait Attributed {
    fn provenance(&self) -> Provenance;
}

/// Provenance for the CorridorKey v2 two-stage video matting network.
pub fn corridorkey_v2_provenance() -> Provenance {
    Provenance {
        name: "CorridorKey v2".into(),
        kind: ProvenanceKind::PortedFromCode,
        paper: None,
        code: UpstreamCode {
            publisher: "Zachary Harris & Corridor Digital LLC".into(),
            url: "https://github.com/nikopueringer/CorridorKey2.0".into(),
            license: "Corridor Key Licence".into(),
            revision: "485f5e9".into(),
        },
        artifacts: vec![
            ArtifactSource {
                name: "corridorkey_v2.safetensors".into(),
                url: "https://huggingface.co/nikopueringer/CorridorKey2.0".into(),
                license: "Corridor Key Licence".into(),
            },
            ArtifactSource {
                name: "foundation_stats.safetensors".into(),
                url: "https://huggingface.co/nikopueringer/CorridorKey2.0".into(),
                license: "Corridor Key Licence".into(),
            },
        ],
    }
}

/// Provenance for NVIDIA C-RADIOv4-SO400M.
pub fn cradio_v4_provenance() -> Provenance {
    Provenance {
        name: "C-RADIOv4-SO400M".into(),
        kind: ProvenanceKind::UpstreamWeights,
        paper: None,
        code: UpstreamCode {
            publisher: "NVIDIA Corporation".into(),
            url: "https://huggingface.co/nvidia/C-RADIOv4-SO400M".into(),
            license: "NVIDIA Open Model License".into(),
            revision: "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8".into(),
        },
        artifacts: vec![ArtifactSource {
            name: "model.safetensors".into(),
            url: "https://huggingface.co/nvidia/C-RADIOv4-SO400M".into(),
            license: "NVIDIA Open Model License".into(),
        }],
    }
}

/// Provenance for Microsoft MoGe-2 ViT-B.
pub fn moge2_vitb_provenance() -> Provenance {
    Provenance {
        name: "MoGe-2-ViT-B".into(),
        kind: ProvenanceKind::UpstreamWeights,
        paper: None,
        code: UpstreamCode {
            publisher: "Microsoft Corporation".into(),
            url: "https://github.com/microsoft/MoGe".into(),
            license: "MIT".into(),
            revision: "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368".into(),
        },
        artifacts: vec![ArtifactSource {
            name: "model.pt".into(),
            url: "https://huggingface.co/Ruicheng/moge-2-vitb-normal".into(),
            license: "MIT".into(),
        }],
    }
}

/// Provenance for Google DeepMind RVM (Recurrent Video MAE) small.
pub fn rvm_small_provenance() -> Provenance {
    Provenance {
        name: "RVM-small".into(),
        kind: ProvenanceKind::UpstreamWeights,
        paper: None,
        code: UpstreamCode {
            publisher: "Google DeepMind".into(),
            url: "https://github.com/google-deepmind/representations4d".into(),
            license: "Apache-2.0".into(),
            revision: "204031069".into(),
        },
        artifacts: vec![ArtifactSource {
            name: "pretrain_rvm_small16_256_204031069.npz".into(),
            url: "https://storage.googleapis.com/representations4d/checkpoints/pretrain_rvm_small16_256_204031069.npz".into(),
            license: "CC-BY-4.0".into(),
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenances_format_cleanly() {
        for p in [
            corridorkey_v2_provenance(),
            cradio_v4_provenance(),
            moge2_vitb_provenance(),
            rvm_small_provenance(),
        ] {
            let c = p.citation();
            assert!(c.contains(&p.code.url), "{c}");
            assert!(!p.artifacts.is_empty());
        }
    }
}
