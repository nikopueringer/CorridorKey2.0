//! `resize.rs` against OpenCV (`scripts/parity/resize_reference.py`).
//!
//! ```text
//! uv run python scripts/parity/resize_reference.py /tmp/corridorkey-resize
//! cargo test --release -p corridorkey-io --test parity_resize -- --ignored --nocapture
//! ```

use std::path::Path;

use corridorkey_io::resize::{bgr_to_gray_f32, resize_lanczos4};
use serde_json::Value;

const REF_DIR: &str = "/tmp/corridorkey-resize";

fn read(name: &str) -> Vec<f32> {
    let p = Path::new(REF_DIR).join(name);
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e} (run scripts/parity/resize_reference.py)", p.display()));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

/// Largest difference, and how many outputs differ at all.
fn compare(got: &[f32], want: &[f32]) -> (f32, usize) {
    assert_eq!(got.len(), want.len());
    got.iter().zip(want).fold((0f32, 0usize), |(m, n), (g, w)| ((g - w).abs().max(m), n + usize::from(g != w)))
}

#[test]
#[ignore = "needs scripts/parity/resize_reference.py output"]
fn matches_opencv() {
    let cases: Value = serde_json::from_slice(&std::fs::read(Path::new(REF_DIR).join("cases.json")).unwrap()).unwrap();
    println!("OpenCV {}", cases["opencv"]);
    for c in cases["cases"].as_array().unwrap() {
        let id = c["id"].as_u64().unwrap();
        let hw = |v: &Value| (v[0].as_u64().unwrap() as usize, v[1].as_u64().unwrap() as usize);
        let (max, differing, total) = match c["kind"].as_str().unwrap() {
            "resize" => {
                let (h, w) = hw(&c["hw"]);
                let (oh, ow) = hw(&c["out_hw"]);
                let got = resize_lanczos4(&read(&format!("resize{id}_in.f32")), (h, w), (oh, ow));
                let (m, n) = compare(&got, &read(&format!("resize{id}_out.f32")));
                println!("resize {h}x{w} -> {oh}x{ow}: max |diff| {m:.3e}, {n} of {} differ", got.len());
                (m, n, got.len())
            }
            _ => {
                let ch = c["channels"].as_u64().unwrap() as usize;
                let got = bgr_to_gray_f32(&read(&format!("gray{id}_in.f32")), ch);
                let (m, n) = compare(&got, &read(&format!("gray{id}_out.f32")));
                println!("gray from {ch} channels: max |diff| {m:.3e}, {n} of {} differ", got.len());
                (m, n, got.len())
            }
        };
        // Within float rounding of OpenCV's own arithmetic (SIMD paths may fuse a multiply-add)
        assert!(max <= 2e-6, "case {id}: {max:.3e} ({differing} of {total} differ)");
    }
}
