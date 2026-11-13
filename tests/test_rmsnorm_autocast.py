import torch
import pytest

from corridorkey.blocks import LayerNorm2d, RMSNorm


def _best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _under_bf16_autocast(module, x):
    device = _best_device()
    with torch.autocast(device, dtype=torch.bfloat16):
        return module.to(device)(x.to(device).to(torch.bfloat16))


@pytest.mark.skipif(_best_device() == "cpu", reason="bfloat16 autocast requires CUDA or MPS")
def test_rmsnorm_preserves_bf16_under_autocast():
    assert _under_bf16_autocast(RMSNorm(64), torch.randn(4, 16, 64)).dtype == torch.bfloat16


@pytest.mark.skipif(_best_device() == "cpu", reason="bfloat16 autocast requires CUDA or MPS")
def test_layernorm2d_preserves_bf16_under_autocast():
    assert _under_bf16_autocast(LayerNorm2d(8), torch.randn(1, 8, 4, 4)).dtype == torch.bfloat16


if __name__ == "__main__":
    test_rmsnorm_preserves_bf16_under_autocast()
    test_layernorm2d_preserves_bf16_under_autocast()
    print("PASS")
