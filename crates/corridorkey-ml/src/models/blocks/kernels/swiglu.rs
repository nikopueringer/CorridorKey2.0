//! CubeCL kernels for SwiGLU gating (`silu(gate + b_gate) * (up + b_up)`) and row-bias /
//! residual epilogues.

use cubecl::prelude::*;

use corridorkey_base::core::CkFloat;

/// `silu(x) = x / (1 + exp(-x))`.
#[cube]
pub fn silu<F: Float>(x: F) -> F {
    x / (F::new(1.0f32) + F::exp(-x))
}

/// Fused elementwise bias addition and SwiGLU activation over split `gate` and `up` `[num_rows, hidden]`:
/// `output[i] = silu(gate[i] + bias_gate[i % hidden]) * (up[i] + bias_up[i % hidden])`.
#[cube(launch)]
pub fn swiglu_gate_kernel<F: CkFloat>(
    gate: &[F],
    up: &[F],
    bias_gate: &[F],
    bias_up: &[F],
    output: &mut [F],
    total: u32,
    hidden: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let c = (i % hidden) as usize;
        let idx = i as usize;
        let g = gate[idx] + bias_gate[c];
        let u = up[idx] + bias_up[c];
        output[idx] = silu::<F>(g) * u;
    }
}

/// Fused elementwise bias addition and SwiGLU activation over horizontally-concatenated
/// `gate_up` (`[num_rows, 2 * hidden]`):
/// `output[r, c] = silu(gate_up[r, c] + bias_gate[c]) * (gate_up[r, hidden + c] + bias_up[c])`.
#[cube(launch)]
pub fn swiglu_gate_fused_kernel<F: CkFloat>(
    gate_up: &[F],
    bias_gate: &[F],
    bias_up: &[F],
    output: &mut [F],
    total: u32,
    hidden: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let r = (i / hidden) as usize;
        let c = (i % hidden) as usize;
        let h = hidden as usize;
        let base = r * (2 * h);
        let g = gate_up[base + c] + bias_gate[c];
        let u = gate_up[base + h + c] + bias_up[c];
        output[i as usize] = silu::<F>(g) * u;
    }
}

/// Elementwise row bias addition: `inout[i] += bias[i % cols]`.
#[cube(launch)]
pub fn add_row_bias_kernel<F: CkFloat>(
    inout: &mut [F],
    bias: &[F],
    total: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let c = (i % cols) as usize;
        let idx = i as usize;
        inout[idx] += bias[c];
    }
}

/// Fused row-bias addition + residual add: `inout[i] = residual[i] + inout[i] + bias[i % cols]`.
#[cube(launch)]
pub fn add_row_bias_residual_kernel<F: CkFloat>(
    inout: &mut [F],
    bias: &[F],
    residual: &[F],
    total: u32,
    cols: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < total {
        let c = (i % cols) as usize;
        let idx = i as usize;
        inout[idx] = residual[idx] + inout[idx] + bias[c];
    }
}
