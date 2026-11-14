"""Regression for silent Linear bias loss on M1/macOS 26."""
import copy
import pickle
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from corridorkey.mps_linear import preserve_mps_linear_bias


class InheritedLinear(nn.Linear):
    """Like foundation patch projections, with a subclass-specific attribute."""
    label = "patch"


class CustomLinear(nn.Linear):
    def forward(self, input):
        return super().forward(input) * 2


def test_install_preserves_objects_hooks_parameters_aliases_and_serialization():
    model = nn.Sequential(InheritedLinear(4, 8), CustomLinear(8, 4), nn.Linear(4, 4, bias=False))
    model.add_module("shared", model[0])
    original = model[0]
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    state = copy.deepcopy(model.state_dict())
    x = torch.randn(2, 3, 4)
    expected = original(x)
    called = []
    handle = original.register_forward_hook(lambda *_: called.append(True))
    assert preserve_mps_linear_bias(model) == 1
    assert preserve_mps_linear_bias(model) == 0
    assert model[0] is original is model.shared
    assert type(original) is InheritedLinear and original.label == "patch"
    assert model[1].forward.__func__ is CustomLinear.forward
    assert model[2].forward.__func__ is nn.Linear.forward
    assert parameter_ids == {name: id(value) for name, value in model.named_parameters()}
    assert list(state) == list(model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    torch.testing.assert_close(original(x), expected, atol=0, rtol=0)
    assert called == [True]
    handle.remove()
    for cloned in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert cloned[0] is cloned.shared and cloned[0] is not original
        assert cloned[0].forward.args[0] is cloned[0]
        cloned.load_state_dict(state, strict=True)
        torch.testing.assert_close(cloned[0](x), expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_non_metal_linear_is_bit_exact(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    model = nn.Linear(32, 96).to(device)
    x = torch.randn(2, 7, 32, device=device)
    with torch.inference_mode():
        expected = model(x)
        preserve_mps_linear_bias(model)
        actual = model(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple GPU required")
@pytest.mark.parametrize("shape", [(1, 32), (5, 1, 32), (2, 3, 7, 32)])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_mps_linear_retains_bias_and_output_dtype(shape, autocast, inference):
    torch.manual_seed(81)
    reference = nn.Linear(32, 96).eval()
    with torch.no_grad():
        reference.weight.mul_(0.3)
        reference.bias.copy_(torch.linspace(0.4, 0.9, 96))
    x = torch.randn(shape)
    model = copy.deepcopy(reference).to("mps")
    preserve_mps_linear_bias(model)
    dtype = torch.bfloat16 if autocast else torch.float32
    # Independent CPU oracle with the same input/parameter quantization.
    expected = torch.nn.functional.linear(
        x.to(dtype).float(), reference.weight.to(dtype).float(),
        reference.bias.to(dtype).float()).to(dtype)
    with torch.inference_mode() if inference else nullcontext():
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=autocast):
            actual = model(x.to("mps"))
            bias_only = model(torch.zeros_like(x, device="mps"))
        assert actual.dtype == dtype
        assert torch.isfinite(actual).all()
        tolerance = 0.01 if autocast else 1e-5
        torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(
            bias_only.cpu(), reference.bias.to(dtype).expand_as(expected), atol=0, rtol=0)
        if not inference:
            actual.float().sum().backward()
            torch.testing.assert_close(
                model.bias.grad.cpu(), torch.full_like(reference.bias, x.numel() // 32))


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_runtime_installs_after_loading_all_foundation_layers(monkeypatch, device):
    from corridorkey import runtime

    class Model(nn.Module):
        def __init__(self, **_):
            super().__init__()
            self.foundations = nn.Sequential(InheritedLinear(4, 4))
            self.stage1 = nn.Linear(4, 4)

    checkpoint = {"format": "corridorkey_v2_runtime_checkpoint",
                  "model_config": {}, "state_dict": Model().state_dict()}
    paths = SimpleNamespace(validate=lambda: None, weights="stub", cradio_dir="stub",
                            moge_checkpoint="stub", rvm_ckpt="stub", foundation_stats="stub")
    monkeypatch.setattr(runtime, "prepare_runtime_environment", lambda _: None)
    monkeypatch.setattr(runtime, "load_runtime_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(runtime, "CorridorKeyV3", Model)
    model = runtime.load_model(paths, device)
    assert not model.training
    for layer in (model.foundations[0], model.stage1):
        assert not layer.weight.requires_grad
        patched = getattr(layer.forward, "__func__", None) is not nn.Linear.forward
        assert patched == (device == "mps")
