//! The whole clip pipeline against the product's own `runtime.run_inference` in fp32
//! (`scripts/parity/pipeline_reference.py`): 12 frames, two overlapping Stage 1 windows (Hann blend,
//! carried hints), every output as f32 EXR frames.
//!
//! ```text
//! uv run python scripts/convert_foundations.py
//! uv run python scripts/parity/pipeline_reference.py /tmp/corridorkey-e2e   # also writes frames_rgb8.u8
//! CORRIDORKEY_RUNTIME=cuda cargo test --release -p corridorkey-ml --test parity_pipeline -- --ignored --nocapture
//! CORRIDORKEY_RUNTIME=wgpu cargo test --release -p corridorkey-ml --test parity_pipeline -- --ignored --nocapture
//! ```
//!
//! One runtime per process: each loads the whole keyer twice (f32, f16), and CubeCL keeps a
//! runtime's memory pool for the life of the process, so all runtimes in one process exceed 8 GB.

use std::collections::HashMap;
use std::path::Path;

use corridorkey_base::core::CkFloat;
use corridorkey_base::frame::RenderOutput;
use corridorkey_core::{CkResult, ComputeTarget, FrameMeta};
use corridorkey_ml::MlContext;
use corridorkey_ml::pipeline::{FrameReader, FrameSink, Keyer, ModelPaths, PipelineConfig, RenderedFrames};
use half::f16;

const REF_DIR: &str = "/tmp/corridorkey-e2e";
const OUTPUTS: [(RenderOutput, &str); 4] = [
    (RenderOutput::Alpha, "alpha"),
    (RenderOutput::Fg, "fg"),
    (RenderOutput::Cutout, "cutout"),
    (RenderOutput::Checker, "checker_comp_despill050"),
];

struct Bytes {
    frames: Vec<u8>,
    frame_len: usize,
}

impl FrameReader for Bytes {
    fn read_rgb8(&mut self, start: usize, count: usize) -> CkResult<Vec<u8>> {
        Ok(self.frames[start * self.frame_len..(start + count) * self.frame_len].to_vec())
    }
}

#[derive(Default)]
struct Collect {
    frames: HashMap<RenderOutput, Vec<(usize, Vec<f32>)>>,
}

impl FrameSink for Collect {
    fn write(&mut self, output: RenderOutput, first: usize, _n: usize, data: RenderedFrames) -> CkResult<()> {
        let RenderedFrames::Exr(v) = data else { panic!("EXR run") };
        self.frames.entry(output).or_default().push((first, v));
        Ok(())
    }
}

fn read_f32(name: &str) -> Vec<f32> {
    let path = Path::new(REF_DIR).join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e} (run scripts/parity/pipeline_reference.py)", path.display()));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

/// Runs the pipeline as `F`; returns `(output, max |error| / max(1, |ref|), mean |error|)`.
fn run<F: CkFloat>(target: ComputeTarget) -> Vec<(String, f64, f64)> {
    let ctx = MlContext::new(target).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let keyer = Keyer::load::<F>(&ctx, &ModelPaths::from_root(&root)).unwrap();
    let (frames_n, h0, w0) = (12, 420, 664);
    let meta = FrameMeta::new((h0, w0), 1.0, 128, 24.0, frames_n).unwrap();
    let mut reader = Bytes { frames: std::fs::read(Path::new(REF_DIR).join("frames_rgb8.u8")).unwrap(), frame_len: h0 * w0 * 3 };
    let mut sink = Collect::default();
    let cfg = PipelineConfig {
        chunk: 8,
        stride: 4,
        outputs: OUTPUTS.iter().map(|(o, _)| *o).collect(),
        exr: true,
        ..PipelineConfig::default()
    };
    keyer.run::<F>(&ctx, &meta, &mut reader, &mut sink, &cfg).unwrap();

    OUTPUTS
        .iter()
        .map(|(out, name)| {
            let mut parts = sink.frames.remove(out).expect("output written");
            parts.sort_by_key(|(t, _)| *t);
            let got: Vec<f32> = parts.into_iter().flat_map(|(_, v)| v).collect();
            let want = read_f32(&format!("{name}.f32"));
            assert_eq!(got.len(), want.len(), "{name}");
            let (mut max, mut sum) = (0f64, 0f64);
            for (&g, &w) in got.iter().zip(&want) {
                let e = (g as f64 - w as f64).abs();
                max = max.max(e / (w.abs() as f64).max(1.0));
                sum += e;
            }
            (name.to_string(), max, sum / want.len() as f64)
        })
        .collect()
}

#[test]
#[ignore = "needs converted backbones and scripts/parity/pipeline_reference.py output"]
fn pipeline_matches_the_product() {
    let targets = match std::env::var(corridorkey_core::compute::RUNTIME_ENV) {
        Ok(_) => vec![ComputeTarget::from_env().unwrap()],
        Err(_) => ComputeTarget::available(),
    };
    for target in targets {
        for (precision, errs) in [("f32", run::<f32>(target)), ("f16", run::<f16>(target))] {
            for (name, max, mean) in &errs {
                println!("{target} {precision} {name:24}: max error {max:.3e} (relative where |v| > 1), mean {mean:.3e}");
            }
            if precision == "f32" {
                for (name, max, mean) in &errs {
                    assert!(*max < 1e-2 && *mean < 1e-4, "{target} f32 {name}: {max:.3e} / {mean:.3e}");
                }
            }
        }
    }
}
