#!/usr/bin/env python3
"""Peak device memory of a command, as the NVIDIA driver reports it for its process (context,
allocator pools and all: the same instrument for PyTorch and every Rust runtime).

Samples `nvidia-smi --query-compute-apps` every 20 ms. When the command prints a line containing
`STAGES_BEGIN`, the peak restarts from there (PyTorch runs its backbones first, then frees them).

usage: peak_vram.py <command> [args...]
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path


def process_tree(pid: int) -> set[int]:
    """`pid` and its descendants (launchers such as `uv run` leave the GPU to a child)."""
    tree, todo = set(), [pid]
    while todo:
        p = todo.pop()
        tree.add(p)
        for task in Path(f"/proc/{p}/task").glob("*"):
            try:
                todo += [int(c) for c in (task / "children").read_text().split()]
            except OSError:
                pass
    return tree


def used_mib(pid: int) -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=2,
    ).stdout
    tree = process_tree(pid)
    total = 0.0
    for row in out.strip().splitlines():
        p, mib = (x.strip() for x in row.split(","))
        if p.isdigit() and int(p) in tree:
            total += float(mib)
    return total


def main() -> None:
    proc = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak = [0.0]
    done = threading.Event()

    def sample() -> None:
        while not done.is_set():
            peak[0] = max(peak[0], used_mib(proc.pid))
            time.sleep(0.02)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    for line in proc.stdout:
        if "STAGES_BEGIN" in line:
            peak[0] = 0.0
        sys.stdout.write(line)
    proc.wait()
    done.set()
    sampler.join(timeout=1)
    print(f"PEAK_VRAM_MIB {peak[0]:.0f}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
