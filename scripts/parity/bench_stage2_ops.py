#!/usr/bin/env python3
"""Stage 2 block operators at model shapes on PyTorch (cuBLAS GEMM, SDPA), fp32 and bf16.
Rust counterpart: crates/corridorkey-ml/tests/bench_stage2_ops.rs."""

import torch
import torch.nn.functional as F

TOKENS, DIM, HIDDEN, HEADS, HEAD_DIM = 1536, 1280, 3200, 16, 80
GEMMS = [
    ("qkv", TOKENS, DIM, 3 * DIM),
    ("attn gate", TOKENS, DIM, HEADS),
    ("proj", TOKENS, DIM, DIM),
    ("mlp gate|up", TOKENS, DIM, 2 * HIDDEN),
    ("mlp down", TOKENS, HIDDEN, DIM),
]


def time_ms(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
    return sorted(times)[len(times) // 2]


def main():
    print(torch.cuda.get_device_name(0), "tf32:", torch.backends.cuda.matmul.allow_tf32)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        print(f"\n{dtype}:")
        total = 0.0
        for name, m, k, n in GEMMS:
            a = torch.randn(m, k, device="cuda", dtype=dtype)
            b = torch.randn(k, n, device="cuda", dtype=dtype)
            ms = time_ms(lambda: a @ b)
            total += ms
            print(f"  {name:12} [{m}x{k}]·[{k}x{n}]  {ms:7.3} ms  {2 * m * k * n / ms / 1e9:5.2f} TFLOP/s")
        q, k, v = (torch.randn(1, HEADS, TOKENS, HEAD_DIM, device="cuda", dtype=dtype) for _ in range(3))
        ms = time_ms(lambda: F.scaled_dot_product_attention(q, k, v))
        print(f"  {'full sdpa':12} [{HEADS}x{TOKENS}x{HEAD_DIM}]       {ms:7.3} ms  "
              f"{4 * HEADS * TOKENS * TOKENS * HEAD_DIM / ms / 1e9:5.2f} TFLOP/s")
        print(f"  GEMMs per block: {total:.2f} ms")


if __name__ == "__main__":
    main()
