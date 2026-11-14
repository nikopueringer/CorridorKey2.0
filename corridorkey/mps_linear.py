"""Preserve Linear bias on Metal with the bundled PyTorch 2.9.1.

The fused MPS matmul can omit bias on M1/macOS 26 (pytorch/pytorch#188438).
Upstream fixes this by adding bias separately. Keep the workaround local to
loaded model instances, including frozen foundations; never patch torch globally.
"""
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


def _linear_forward(module: nn.Linear, input: torch.Tensor) -> torch.Tensor:
    if input.device.type != "mps" or module.bias is None:
        return nn.Linear.forward(module, input)
    output = F.linear(input, module.weight, None)
    # Autocast may produce BF16 from FP32 weights. Do not promote its output
    # back to FP32 when adding bias, or change the checkpoint's parameter dtype.
    return output + module.bias.to(dtype=output.dtype)


def preserve_mps_linear_bias(model: nn.Module) -> int:
    """Install the MPS path in-place, retaining aliases, hooks and state keys.

    Linear subclasses inheriting the standard forward (e.g. C-RADIO patch
    embedding) are included. Custom forwards are deliberately left intact.
    Repeated calls do nothing. Non-MPS execution still uses the original op.
    """
    count = 0
    for module in model.modules():
        if (isinstance(module, nn.Linear) and module.bias is not None
                and getattr(module.forward, "__func__", None) is nn.Linear.forward):
            module.forward = partial(_linear_forward, module)
            count += 1
    return count
