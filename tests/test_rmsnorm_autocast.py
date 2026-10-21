import torch

from corridorkey.blocks import LayerNorm2d, RMSNorm


def _under_bf16_autocast(module, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return module.cuda()(x.cuda().to(torch.bfloat16))


def test_rmsnorm_preserves_bf16_under_autocast():
    assert _under_bf16_autocast(RMSNorm(64), torch.randn(4, 16, 64)).dtype == torch.bfloat16


def test_layernorm2d_preserves_bf16_under_autocast():
    assert _under_bf16_autocast(LayerNorm2d(8), torch.randn(1, 8, 4, 4)).dtype == torch.bfloat16


if __name__ == "__main__":
    test_rmsnorm_preserves_bf16_under_autocast()
    test_layernorm2d_preserves_bf16_under_autocast()
    print("PASS")
