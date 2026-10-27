//! OpenCV's host-side image operations the product applies to hints, ported for single-channel f32:
//! `cv2.resize(..., interpolation=cv2.INTER_LANCZOS4)` (`preprocess_hint_frame` resizes hint frames
//! of another size to the model size) and `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)` of float pixels
//! (EXR hint frames). Checked against OpenCV 5.0 by `tests/parity_resize.rs`: within 3.6e-7 of
//! it (1–3 float units in the last place, where OpenCV's SIMD paths sum in another order; neither
//! plain nor fused multiply-adds reproduce it exactly), far below the bf16 rounding hints then get.

use std::f64::consts::PI;

/// `interpolateLanczos4` (`imgproc/src/resize.cpp`): the 8 weights of the source pixels at
/// offsets -3..=4 for fractional position `x`, normalized to sum 1.
fn lanczos4_weights(x: f32) -> [f32; 8] {
    const S45: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const CS: [[f64; 2]; 8] = [[1.0, 0.0], [-S45, -S45], [0.0, 1.0], [S45, -S45], [-1.0, 0.0], [S45, S45], [0.0, -1.0], [-S45, S45]];
    let y0 = -((x + 3.0) as f64) * PI * 0.25;
    let (s0, c0) = (y0.sin(), y0.cos());
    let mut w = [0f32; 8];
    let mut sum = 0f32;
    for (i, wi) in w.iter_mut().enumerate() {
        let d = x + 3.0 - i as f32;
        *wi = if d.abs() >= 1e-6 {
            let y = -(d as f64) * PI * 0.25;
            ((CS[i][0] * s0 + CS[i][1] * c0) / (y * y)) as f32
        } else {
            1e30
        };
        sum += *wi;
    }
    let inv = 1.0 / sum;
    w.iter_mut().for_each(|v| *v *= inv);
    w
}

/// Source index and weights of each output position along one axis of `src` → `dst` samples
/// (`resizeGeneric`: `f = (float)((d + 0.5)·scale − 0.5)`, `s = floor(f)`).
fn axis(src: usize, dst: usize) -> Vec<(isize, [f32; 8])> {
    let scale = 1.0 / (dst as f64 / src as f64);
    (0..dst)
        .map(|d| {
            let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let s = f.floor();
            (s as isize, lanczos4_weights(f - s))
        })
        .collect()
}

/// `cv2.resize(src, (ow, oh), interpolation=cv2.INTER_LANCZOS4)` of a single-channel f32 image
/// `[h, w]`: separable, horizontal pass first, borders repeating the edge pixel.
pub fn resize_lanczos4(src: &[f32], (h, w): (usize, usize), (oh, ow): (usize, usize)) -> Vec<f32> {
    assert_eq!(src.len(), h * w, "image of {h}x{w} expected");
    if (h, w) == (oh, ow) {
        return src.to_vec();
    }
    let clamp = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
    let xs = axis(w, ow);
    let ys = axis(h, oh);
    // HResizeLanczos4: every source row, each output column a sum over 8 neighbours in order
    let hrow = |y: usize| -> Vec<f32> {
        let row = &src[y * w..(y + 1) * w];
        xs.iter()
            .map(|&(sx, wt)| {
                let mut v = 0f32;
                for (k, wk) in wt.iter().enumerate() {
                    v += row[clamp(sx - 3 + k as isize, w)] * wk;
                }
                v
            })
            .collect()
    };
    let rows: Vec<Vec<f32>> = (0..h).map(hrow).collect();
    // VResizeLanczos4: each output row from 8 horizontally resized rows, accumulated in order
    let mut out = vec![0f32; oh * ow];
    for (dy, &(sy, wt)) in ys.iter().enumerate() {
        let picked: [&[f32]; 8] = std::array::from_fn(|k| rows[clamp(sy - 3 + k as isize, h)].as_slice());
        let dst = &mut out[dy * ow..(dy + 1) * ow];
        for (x, d) in dst.iter_mut().enumerate() {
            let mut v = picked[0][x] * wt[0];
            for k in 1..8 {
                v += picked[k][x] * wt[k];
            }
            *d = v;
        }
    }
    out
}

/// `cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)` of float pixels `[.., channels]` in B G R (A) order:
/// `0.114·B + 0.587·G + 0.299·R` (alpha ignored).
pub fn bgr_to_gray_f32(src: &[f32], channels: usize) -> Vec<f32> {
    assert!(channels == 3 || channels == 4, "BGR or BGRA expected, got {channels} channels");
    src.chunks_exact(channels).map(|p| p[0] * 0.114 + p[1] * 0.587 + p[2] * 0.299).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one_and_hit_integer_positions() {
        for x in [0.0f32, 0.25, 0.5, 0.999] {
            let s: f32 = lanczos4_weights(x).iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "{x}: {s}");
        }
        let w = lanczos4_weights(0.0);
        assert!((w[3] - 1.0).abs() < 1e-6 && w.iter().enumerate().all(|(i, v)| i == 3 || v.abs() < 1e-6));
    }

    #[test]
    fn same_size_is_a_copy() {
        let src: Vec<f32> = (0..12).map(|v| v as f32).collect();
        assert_eq!(resize_lanczos4(&src, (3, 4), (3, 4)), src);
    }
}
