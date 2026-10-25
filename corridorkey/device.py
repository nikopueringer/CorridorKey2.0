"""GPU selection and memory operations for CUDA and Apple Metal."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("CorridorKey v2 requires a CUDA GPU or an Apple Metal GPU.")


def model_device(model) -> torch.device:
    # Modules move to CPU between phases, so retain the execution device.
    if hasattr(model, "_runtime_device"):
        return torch.device(model._runtime_device)
    return next(model.foundations.parameters()).device


def area_resize(tensor: torch.Tensor, size) -> torch.Tensor:
    # Metal does not implement adaptive pooling with non-divisible dimensions.
    # Use the same PyTorch operator on CPU for this case, retaining input dtype.
    if tensor.device.type == "mps" and any(n % out for n, out in zip(tensor.shape[-2:], size)):
        return F.interpolate(tensor.cpu(), size=size, mode="area").to(tensor.device)
    return F.interpolate(tensor, size=size, mode="area")


def empty_cache(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def synchronize(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_budget_gb(device) -> float:
    device = torch.device(device)
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return (free + torch.cuda.memory_reserved(device)) / 2**30
    if device.type == "mps":
        # CUDA measurements understate some Metal intermediates. Keep headroom
        # within Metal's recommended working set, including resident weights.
        return max(0.0, (0.5 * torch.mps.recommended_max_memory()
                        - torch.mps.driver_allocated_memory()
                        + torch.mps.current_allocated_memory()) / 2**30)
    raise RuntimeError(f"Unsupported inference device: {device}")
