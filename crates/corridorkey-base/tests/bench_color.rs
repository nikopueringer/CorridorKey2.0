//! Colour transfer kernels against the Python implementation: accuracy and latency.
//!
//! Needs the data written by `scripts/parity/color_reference.py --out <dir>`; run with
//! `CORRIDORKEY_PARITY_DIR=<dir> cargo test --release -p corridorkey-base --test bench_color -- --ignored --nocapture`.
//! Errors are measured against the f64 NumPy reference (and PyTorch's f32 result for scale);
//! times are the median of [`ITERS`] runs synced with the device, with the frame already on the
//! device ("resident") and including upload and download into buffers allocated once ("end to end").

use std::path::{Path, PathBuf};

use corridorkey_base::color::{convert, vector_size};
use corridorkey_base::core::buffer::{download, empty, upload, write};
use corridorkey_core::compute::bench::time_device;
use corridorkey_core::compute::{ComputeTarget, ComputeTask};
use corridorkey_core::{ColorSpace, LinearAdjust};
use cubecl::prelude::*;
use serde_json::Value;

const ITERS: usize = 20;
const PARITY_DIR_ENV: &str = "CORRIDORKEY_PARITY_DIR";

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

/// `(max |error|, max relative error over |reference| ≥ 1e-3, RMS error)` of `got` against `want`.
fn errors(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    let (mut max_abs, mut max_rel, mut sq) = (0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let e = (g as f64 - w as f64).abs();
        max_abs = max_abs.max(e);
        if (w as f64).abs() >= 1e-3 {
            max_rel = max_rel.max(e / (w as f64).abs());
        }
        sq += e * e;
    }
    (max_abs, max_rel, (sq / got.len() as f64).sqrt())
}

fn space(name: &str) -> ColorSpace {
    match name {
        "srgb" => ColorSpace::Srgb,
        "linear" => ColorSpace::Linear,
        "asinh" => ColorSpace::Asinh,
        other => panic!("unknown encoding {other}"),
    }
}

/// `(from, to, adjust)` of each conversion the Python script ran.
fn conversion(name: &str, meta: &Value) -> (ColorSpace, ColorSpace, LinearAdjust) {
    let a = &meta["adjust"];
    let adjust = LinearAdjust { gain: a["gain"].as_f64().unwrap(), contrast: a["contrast"].as_f64().unwrap(), pivot: a["pivot"].as_f64().unwrap() };
    let none = LinearAdjust::default();
    use ColorSpace::*;
    match name {
        "srgb_to_linear" => (Srgb, Linear, none),
        "linear_to_srgb" => (Linear, Srgb, none),
        "linear_to_asinh" => (Linear, Asinh, none),
        "asinh_to_linear" => (Asinh, Linear, none),
        "srgb_adjust_asinh" => (Srgb, Asinh, adjust),
        other => panic!("unknown conversion {other}"),
    }
}

struct Compare {
    dir: PathBuf,
    meta: Value,
}

impl ComputeTask for Compare {
    type Output = ();
    fn run(self, client: Client) {
        let torch = self.meta["cases"].as_object().unwrap();
        println!("\n{} vs PyTorch {} on {}", client.name(), self.meta["torch"], self.meta["device"]);
        println!(
            "{:6} {:18} | {:>9} {:>9} {:>9} | {:>9} | {:>8} {:>8} {:>6} | {:>8} {:>8} {:>6}",
            "size", "conversion", "max abs", "max rel", "rms", "torch abs", "res ms", "torch", "×", "e2e ms", "torch", "×"
        );
        let mut transfers = vec![];
        for size in ["1080p", "4k"] {
            for (name, from) in self.meta["conversions"].as_object().unwrap() {
                let (from_space, to, adjust) = conversion(name, &self.meta);
                assert_eq!(from_space, space(from.as_str().unwrap()));
                let input = read_f32(&self.dir.join(format!("{size}_in_{}.f32", from.as_str().unwrap())));
                let reference = read_f32(&self.dir.join(format!("{size}_{name}_ref.f32")));
                let torch_out = read_f32(&self.dir.join(format!("{size}_{name}_torch.f32")));
                let n = input.len();

                let x = upload(&client, &input);
                let y = empty::<f32>(&client, n);
                convert::<f32>(&client, &x, &y, n, from_space, to, adjust);
                let got: Vec<f32> = download(&client, y.clone());
                let (abs, rel, rms) = errors(&got, &reference);
                let (torch_abs, _, _) = errors(&torch_out, &reference);

                let resident = time_device(&client, ITERS, || convert::<f32>(&client, &x, &y, n, from_space, to, adjust));
                // Buffers allocated once and reused, as a streaming pipeline does
                let upload_only = time_device(&client, ITERS, || write(&client, &x, &input));
                let download_only = time_device(&client, ITERS, || {
                    let _: Vec<f32> = download(&client, y.clone());
                });
                let end_to_end = time_device(&client, ITERS, || {
                    write(&client, &x, &input);
                    convert::<f32>(&client, &x, &y, n, from_space, to, adjust);
                    let _: Vec<f32> = download(&client, y.clone());
                });
                let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
                transfers.push(format!("{size:6} {name:18} vector {} | upload {:7.2} ms  download {:7.2} ms", vector_size::<f32>(&client, n), ms(upload_only), ms(download_only)));
                let t = &torch[&format!("{size}/{name}")];
                let (t_res, t_e2e) = (t["resident_ms"].as_f64().unwrap(), t["end_to_end_ms"].as_f64().unwrap());
                let (res, e2e) = (resident.as_secs_f64() * 1e3, end_to_end.as_secs_f64() * 1e3);
                println!(
                    "{size:6} {name:18} | {abs:9.2e} {rel:9.2e} {rms:9.2e} | {torch_abs:9.2e} | {res:8.3} {t_res:8.3} {:6.1} | {e2e:8.3} {t_e2e:8.3} {:6.2}",
                    t_res / res,
                    t_e2e / e2e
                );
                assert!(rel < 1e-5, "{size} {name}: relative error {rel:e}");
            }
        }
        println!("\nTransfers (reused buffers):");
        for line in transfers {
            println!("{line}");
        }
    }
}

#[test]
#[ignore = "needs the data of scripts/parity/color_reference.py (CORRIDORKEY_PARITY_DIR)"]
fn compare_with_python() {
    let dir = PathBuf::from(std::env::var(PARITY_DIR_ENV).expect("set CORRIDORKEY_PARITY_DIR"));
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("torch_timings.json")).unwrap()).unwrap();
    for target in ComputeTarget::available() {
        target.run(Compare { dir: dir.clone(), meta: meta.clone() }).unwrap();
    }
}
