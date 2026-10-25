"""Native attention must handle QKV views without reading the wrong buffer."""
import pytest
import torch

from corridorkey import mps_attention


def test_cpu_keeps_pytorch_path():
    x = torch.randn(1, 2, 7, 8)
    with torch.inference_mode():
        assert mps_attention.prepare(x, x, x) is None


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('head_dim', [64, 80])
def test_native_attention_handles_noncontiguous_offset_views(dtype, head_dim):
    if not torch.backends.mps.is_available():
        pytest.skip('Apple GPU required')
    if not mps_attention.available():
        pytest.skip('Optional native attention has not been built')
    torch.manual_seed(91)
    base = torch.randn(3, 2, 4, 23, head_dim).to(dtype)
    expected = torch.nn.functional.scaled_dot_product_attention(
        base[1, :, :, 2:19].float(), base[0].float(), base[2].float())
    gpu = base.to('mps')
    q, k, v = gpu[1, :, :, 2:19], gpu[0], gpu[2]
    with torch.inference_mode():
        runner = mps_attention.prepare(q, k, v)
        assert runner is not None
        actual = runner(0, q.shape[2]).cpu().float()
    assert torch.isfinite(actual).all()
    assert actual.shape == expected.shape
    tolerance = 0.005 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_native_attention_can_be_disabled(monkeypatch):
    monkeypatch.setenv('CORRIDORKEY_MPS_NATIVE_ATTENTION', '0')
    assert not mps_attention.available()


@pytest.mark.parametrize('version', ['', '14.7.1'])
def test_unsupported_macos_does_not_load_native_library(monkeypatch, version):
    monkeypatch.setattr(mps_attention.platform, 'mac_ver', lambda: (version, (), ''))
    monkeypatch.setattr(mps_attention, 'Path', lambda *_: pytest.fail('Native files accessed on unsupported OS'))
    assert mps_attention._operator.__wrapped__() is None
