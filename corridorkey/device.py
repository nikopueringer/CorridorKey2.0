from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Accelerator:
    backend: str
    device: torch.device


def _backend_available(backend: str) -> bool:
    if not torch.cuda.is_available():
        return False
    if backend == "nvidia":
        return bool(torch.version.cuda)
    if backend == "amd":
        return bool(torch.version.hip)
    raise ValueError(f"unsupported accelerator backend: {backend}")


def resolve_accelerator(requested: str = "auto") -> Accelerator:
    requested = requested.lower()
    if requested not in {"auto", "nvidia", "amd"}:
        raise ValueError(f"unsupported --gpu value: {requested!r}")

    if requested == "auto":
        if _backend_available("nvidia"):
            backend = "nvidia"
        elif _backend_available("amd"):
            backend = "amd"
        else:
            raise RuntimeError(
                "no supported GPU backend is available. Install a CUDA or "
                "ROCm/HIP PyTorch build and verify that a GPU is visible.")
    else:
        backend = requested
        if not _backend_available(backend):
            build = "ROCm/HIP" if backend == "amd" else "CUDA"
            raise RuntimeError(
                f"--gpu {backend} was requested, but a working {build} "
                "PyTorch backend is not available. Check the installed "
                "PyTorch wheel and GPU visibility.")

    return Accelerator(backend=backend, device=torch.device("cuda"))


def describe_accelerator(accelerator: Accelerator) -> str:
    index = accelerator.device.index
    if index is None:
        index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    return (
        f"GPU backend: {accelerator.backend} "
        f"(device={accelerator.device}, index={index}, name={name}, "
        f"visible_devices={torch.cuda.device_count()}, "
        f"torch={torch.__version__}, cuda={torch.version.cuda or 'none'}, "
        f"hip={torch.version.hip or 'none'})")


def autocast(accelerator: Accelerator):
    return torch.autocast(device_type=accelerator.device.type,
                          dtype=torch.bfloat16)


def clear_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def memory_budget_gb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("automatic VRAM selection requires an available GPU")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    return (free_bytes + reserved_bytes) / 2**30
