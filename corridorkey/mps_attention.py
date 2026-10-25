"""Optional native unmasked SDPA, retaining the standard PyTorch fallback."""
from functools import lru_cache
import os
import platform
from pathlib import Path
import warnings

import torch


@lru_cache(maxsize=1)
def _operator():
    # MPSGraph SDPA was introduced in macOS 15. Older systems retain PyTorch.
    if int(platform.mac_ver()[0].split(".")[0] or 0) < 15:
        return None
    build = Path(__file__).parent / 'native/build'
    library = build / 'mps_attention.so'
    version = build / 'torch-version.txt'
    if not library.exists() or not version.exists():
        return None
    if version.read_text().strip() != torch.__version__:
        warnings.warn('Native attention needs rebuilding for this PyTorch version; using PyTorch SDPA.')
        return None
    try:
        torch.ops.load_library(str(library))
    except (OSError, RuntimeError) as error:
        warnings.warn(f'Cannot load native attention; using PyTorch SDPA: {error}')
        return None
    return torch.ops.corridorkey_mps.attention


def available():
    return os.environ.get('CORRIDORKEY_MPS_NATIVE_ATTENTION', '1') != '0' and _operator() is not None


def prepare(q, k, v):
    """Return a query-slice runner, or None for the unchanged PyTorch path.

    MPSGraph's buffer initializer has no offset argument. A contiguous view
    may still have an offset, notably V after QKV unbind, so normalize once.
    """
    if q.device.type != 'mps' or q.ndim != 4 or not available():
        return None
    if any(x.requires_grad for x in (q, k, v)) or torch.is_grad_enabled():
        return None
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None
    operator = _operator()

    def buffer(x):
        if not x.is_contiguous() or x.storage_offset():
            return x.clone(memory_format=torch.contiguous_format)
        return x

    k, v = buffer(k), buffer(v)

    def run(start, end):
        return operator(buffer(q[:, :, start:end]), k, v)
    return run
