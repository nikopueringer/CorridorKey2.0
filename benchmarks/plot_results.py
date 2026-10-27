#!/usr/bin/env python3
"""Charts of the Rust engine vs PyTorch benchmark, as PNGs for the README and the pull request.

Reads `docs/benchmarks/data/*.json` (throughput and peak memory per engine, precision and frame
size; alpha accuracy against PyTorch fp32; the 1080p clip end to end) and writes
`docs/benchmarks/{throughput,memory,accuracy,end_to_end}.png`.

Colors follow the configuration, the same in every chart (a categorical palette in fixed order,
validated for color-vision deficiency). Every bar carries its value, since three of the colors are
light against the background.

usage: uv run --with matplotlib python benchmarks/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "benchmarks" / "data"
OUT = ROOT / "docs" / "benchmarks"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# One color per configuration, in the palette's fixed order
CONFIGS = [
    ("Rust CUDA f16", "#2a78d6"),
    ("Rust CUDA f32", "#eb6834"),
    ("Rust WGPU f16", "#1baf7a"),
    ("PyTorch fp16 (fastest on this GPU)", "#eda100"),
    ("PyTorch fp32", "#e87ba4"),
    ("PyTorch bf16 (as shipped)", "#008300"),
    ("Rust WGPU f32", "#6250d6"),
]
COLOR = dict(CONFIGS)
SHORT = {"PyTorch fp16 (fastest on this GPU)": "PyTorch fp16 (fastest here)"}
SIZES = ["768x512", "1280x768", "1920x1080"]
MACHINE = "RTX 2070 Max-Q, 8 GB (Turing), whole pipeline, one 8-frame window"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def load(name: str):
    return json.loads((DATA / f"{name}.json").read_text())


def frame(ax, title: str, note: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="medium", pad=26)
    ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def grouped(rows, value: str, fmt: str, title: str, note: str, unit: str, path: Path, ref=None) -> None:
    configs = [c for c, _ in CONFIGS[:6]]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    width = 0.8 / len(configs)
    top = max((r[value] or 0) for r in rows if r["config"] in configs)
    top = max(top, ref[0] if ref else 0) * 1.18
    for j, c in enumerate(configs):
        for i, s in enumerate(SIZES):
            r = next((r for r in rows if r["config"] == c and r["size"] == s), None)
            x = i - 0.4 + width * (j + 0.5)
            if r is None:
                continue
            v = r[value]
            if v is None:
                ax.text(x, top * 0.012, "✕", ha="center", va="bottom", fontsize=11, color=INK2)
                continue
            ax.bar(x, v, width * 0.94, color=COLOR[c], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(x, v + top * 0.012, format(v, fmt), ha="center", va="bottom", fontsize=7.5, color=INK, zorder=4)
    if ref:
        ax.axhline(ref[0], color=INK2, linestyle=(0, (4, 4)), linewidth=1, zorder=2)
        ax.text(len(SIZES) - 0.42, ref[0], f" {ref[1]}", va="bottom", ha="right", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(SIZES)), [s.replace("x", " × ") for s in SIZES])
    ax.set_ylim(0, top)
    ax.set_ylabel(unit)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    frame(ax, title, note)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[c]) for c in configs]
    ax.legend(handles, [SHORT.get(c, c) for c in configs], ncol=3, frameon=False, fontsize=8.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def accuracy(rows, path: Path) -> None:
    rows = sorted(rows, key=lambda r: r["mean_codes16"])
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=160)
    labels = [SHORT.get(r["config"], r["config"]) for r in rows]
    vals = [r["mean_codes16"] for r in rows]
    ax.barh(range(len(rows)), vals, height=0.62, color=[COLOR[r["config"]] for r in rows], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(min(vals) / 2, max(vals) * 6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for i, v in enumerate(vals):
        ax.text(v * 1.12, i, f"{v:.2g} codes", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("mean absolute alpha error, 16-bit output codes (log scale)")
    frame(ax, "Alpha accuracy against PyTorch fp32 (shorter is better)",
          "Same 8 frames at 768 × 512; 1 code = 1/65535")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def end_to_end(rows, path: Path) -> None:
    color = {"Rust, CUDA, automatic window": COLOR["Rust CUDA f16"], "Rust, WGPU, automatic window": COLOR["Rust WGPU f16"]}
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=160)
    top = max(r["wall_s"] for r in rows) * 1.45
    for i, r in enumerate(rows):
        ok = r["status"] == "ok"
        c = color.get(r["run"], COLOR["PyTorch bf16 (as shipped)"])
        ax.barh(i, r["wall_s"], height=0.62, color=c if ok else SURFACE, edgecolor=c, hatch=None if ok else "///",
                linewidth=1.5, zorder=3)
        label = f"{r['wall_s']:.0f} s" if ok else f"out of memory after {r['wall_s']:.0f} s"
        ax.text(r["wall_s"] + top * 0.01, i, label, va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [r["run"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, top)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("seconds from launch to finished alpha and foreground videos")
    frame(ax, "A 48-frame 1080p clip, end to end (shorter is better)",
          "Each command line on an H.264 clip: loading, decoding, keying and encoding")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    t = load("throughput")
    grouped(t, "fps", ".2f", "Throughput by frame size (higher is better)",
            f"{MACHINE}.  ✕ = ran out of memory (PyTorch) or refused because it would not fit (Rust)",
            "frames per second", OUT / "throughput.png")
    card = t[0]["card_gb"]
    grouped(t, "peak_gb", ".1f", "Peak GPU memory by frame size (lower is better)",
            f"{MACHINE}.  Whole process, as the driver reports it.  ✕ = did not run",
            "GB", OUT / "memory.png", ref=(card, f"{card:.0f} GB card"))
    accuracy(load("accuracy"), OUT / "accuracy.png")
    end_to_end(load("end_to_end"), OUT / "end_to_end.png")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
