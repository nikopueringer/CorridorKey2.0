#!/usr/bin/env python3
"""CorridorKey 2.0: PyTorch vs the Rust runner (CubeCL / Burn), latency, memory and parity.

Each number is taken the same way for every side:

- Inputs are prepared like the product (`scripts/parity/production_reference.py`: asinh(linear)
  model input, no-hint sentinel, centre reflect-101 pad), for a window of `--frames` frames.
- PyTorch runs every precision mode worth comparing on this GPU (fp32 reference, the product's
  bf16 autocast, and the GPU's fastest native mode: bf16 on sm >= 8.0, fp16 on 7.x).
- Rust runs each `--configs` entry (`runtime:dtype`, dtype `auto|f32|f16`): one launch to autotune
  (persisted), then `--launches` fresh processes; the reported time is their median. Warm timings
  from the process that tuned are never used (they read up to 30 % off on a laptop GPU).
- Peak device memory comes from the driver for every process (`scripts/parity/peak_vram.py`), with
  the PyTorch backbones freed first so both sides hold Stage 1 + Stage 2 only.
- Errors are in product units (`scripts/parity/compare_outputs.py`).

Writes `<out>/benchmark_report.json` and `<out>/benchmark_summary.md`.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARITY = ROOT / "scripts" / "parity"
RUNNER = ROOT / "target" / "release" / "run_stage1_stage2"


def run(cmd: list[str], timeout: int = 3600) -> str:
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        sys.stderr.write(out.stdout + out.stderr)
        raise RuntimeError(f"failed: {' '.join(cmd)}")
    return out.stdout


def peak_mib(text: str) -> float | None:
    m = re.search(r"PEAK_VRAM_MIB (\d+)", text)
    return float(m.group(1)) if m else None


def gpu_state() -> str:
    q = "name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw"
    return run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"]).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "data" / "green_screen_portrait.png"))
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--configs", default="cuda:auto,cuda:f32,wgpu:auto,wgpu:f32")
    ap.add_argument("--launches", type=int, default=3, help="fresh timed launches per Rust config")
    args = ap.parse_args()
    out = Path(args.out)
    ref = out / f"reference_t{args.frames}"
    ref.mkdir(parents=True, exist_ok=True)
    py = [sys.executable]

    print(f"[1/3] PyTorch reference, T={args.frames}")
    run(py + [str(PARITY / "production_reference.py"), args.input, str(ref), "--frames", str(args.frames)])
    torch_ref = json.loads((ref / "torch_ref.json").read_text())
    torch_vram = {}
    for mode in torch_ref["modes"]:
        text = run(py + [str(PARITY / "peak_vram.py")] + py + [
            str(PARITY / "production_reference.py"), args.input, str(out / "vram_scratch"),
            "--frames", str(args.frames), "--modes", mode, "--free-foundations"])
        torch_vram[mode] = peak_mib(text)

    print("[2/3] Rust runner")
    rust = {}
    for cfg in args.configs.split(","):
        runtime, dtype = cfg.split(":")
        dest = ref / f"rust_{runtime}_{dtype}"
        run([str(RUNNER), str(ref), str(out / "tune_scratch"), "--runtime", runtime, "--dtype", dtype])
        timings, peaks = [], []
        for _ in range(args.launches):
            text = run(py + [str(PARITY / "peak_vram.py"), str(RUNNER), str(ref), str(dest),
                             "--runtime", runtime, "--dtype", dtype])
            timings.append(json.loads((dest / "rust_timing.json").read_text()))
            peaks.append(peak_mib(text))
        rust[cfg] = {
            "dtype": timings[0].get("dtype"),
            "stage1_ms": statistics.median(t["warm_stage1_ms"] for t in timings),
            "stage2_ms": statistics.median(t["warm_stage2_ms"] for t in timings),
            "total_ms": statistics.median(t["warm_total_ms"] for t in timings),
            "total_ms_range": [min(t["warm_total_ms"] for t in timings), max(t["warm_total_ms"] for t in timings)],
            "load_ms": statistics.median(t["load_ms"] for t in timings),
            "cold_ms": statistics.median(t["cold_ms"] for t in timings),
            "peak_vram_mib": max(p for p in peaks if p is not None) if any(peaks) else None,
        }

    print("[3/3] Parity")
    dirs = [str(ref / f"rust_{c.replace(':', '_')}") for c in rust]
    parity = run(py + [str(PARITY / "compare_outputs.py"), str(ref)] + dirs)

    sha = run(["git", "rev-parse", "--short", "HEAD"]).strip()
    dirty = bool(run(["git", "status", "--porcelain"]).strip())
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": sha + ("-dirty" if dirty else ""),
        "gpu_state_after": gpu_state(),
        "torch": {**torch_ref, "peak_vram_mib": torch_vram},
        "rust": rust,
        "parity": parity,
    }
    (out / "benchmark_report.json").write_text(json.dumps(report, indent=2))

    h, w = torch_ref["native_hw"]
    lines = [
        "# CorridorKey 2.0: Benchmark Report",
        "",
        f"- **Date**: `{report['date']}`, commit `{report['git']}`",
        f"- **GPU**: {torch_ref['device']} (sm_{''.join(map(str, torch_ref['compute_capability']))}), "
        f"torch {torch_ref['torch']}",
        f"- **Input**: `{w}x{h}` padded to `{torch_ref['w']}x{torch_ref['h']}`, window **T = {torch_ref['t']}**",
        "- Rust times: median of fresh launches after a tuning launch. VRAM: driver peak per process, "
        "Stage 1 + Stage 2 resident.",
        "",
        "| Runtime | Stage 1 | Stage 2 | Total | Peak VRAM |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for mode, r in torch_ref["modes"].items():
        if "oom" in r:
            lines.append(f"| PyTorch {mode} ({r['dtype']}) | — | — | out of memory | — |")
        else:
            lines.append(f"| PyTorch {mode} ({r['dtype']}) | {r['stage1_ms']:.1f} ms | {r['stage2_ms']:.1f} ms | "
                         f"{r['total_ms']:.1f} ms | {torch_vram.get(mode) or 0:.0f} MiB |")
    for cfg, r in rust.items():
        lo, hi = r["total_ms_range"]
        lines.append(f"| Rust {cfg} ({r['dtype']}) | {r['stage1_ms']:.1f} ms | {r['stage2_ms']:.1f} ms | "
                     f"{r['total_ms']:.1f} ms ({lo:.0f}–{hi:.0f}) | {r['peak_vram_mib'] or 0:.0f} MiB |")
    lines += ["", "## Error vs PyTorch fp32 (product units)", "", "```", parity.strip(), "```", ""]
    (out / "benchmark_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
