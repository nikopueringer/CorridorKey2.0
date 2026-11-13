"""Compare bounded window/strip batching with a CPU-generated oracle."""
import copy

import pytest
import torch

from corridorkey.blocks import (
    CSwinAttention2D,
    SwinAttention2D,
    _shifted_region_groups_2d_cached,
    _shifted_region_ids_2d,
)


def _copy_kwargs(kwargs, device):
    return {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in kwargs.items()
    }


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("window_batch", [0, 1, 5, 32])
@pytest.mark.parametrize("mask_kind", ["padding", "explicit", "groups", "ids"])
def test_swin_batching_preserves_masked_results(device, window_batch, mask_kind):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple GPU required")
    torch.manual_seed(7)
    module = SwinAttention2D(32, 4, 4).eval()
    # The production initialization is zero; a zero projection would conceal
    # incorrect masking or chunk placement in this test.
    torch.nn.init.normal_(module.proj.weight, std=0.1)
    # B=2 yields 18 windows, so a batch size of five crosses the boundary
    # between the two images. Zero and 32 exercise the unbatched paths.
    x_cpu = torch.randn(2, 9, 11, 32, device="cpu")
    kwargs_cpu = {}
    if mask_kind == "groups":
        kwargs_cpu["region_groups"] = _shifted_region_groups_2d_cached(9, 11, 12, 12, 4, 2, 2)
    elif mask_kind == "ids":
        ids = _shifted_region_ids_2d(9, 11, 12, 12, 4, 2, 2, torch.device("cpu"))
        kwargs_cpu["region_ids"] = ids.repeat(2, 1)
    elif mask_kind == "explicit":
        mask = torch.ones(9, 16, 16, dtype=torch.bool, device="cpu")
        mask[:, :, -3:] = False
        kwargs_cpu["attn_mask"] = mask

    with torch.inference_mode():
        reference_module = copy.deepcopy(module).cpu().eval()
        reference = reference_module(x_cpu, **kwargs_cpu)

        actual_module = module.to(device).eval()
        x_device = x_cpu.to(device)
        kwargs_device = _copy_kwargs(kwargs_cpu, device)
        actual_module.inference_window_batch = window_batch
        actual = actual_module(x_device, **kwargs_device).cpu()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("window_batch", [0, 1, 32])
def test_swin_single_window_explicit_mask_matches_cpu(device, window_batch):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple GPU required")
    torch.manual_seed(17)
    module = SwinAttention2D(32, 4, 4).eval()
    torch.nn.init.normal_(module.proj.weight, std=0.1)
    x_cpu = torch.randn(1, 4, 4, 32, device="cpu")
    mask_cpu = torch.ones(1, 16, 16, dtype=torch.bool, device="cpu")
    mask_cpu[:, :, -3:] = False
    with torch.inference_mode():
        reference = copy.deepcopy(module).cpu().eval()(x_cpu, attn_mask=mask_cpu)
        actual_module = module.to(device).eval()
        actual_module.inference_window_batch = window_batch
        actual = actual_module(x_cpu.to(device), attn_mask=mask_cpu.to(device)).cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("strip_batch", [0, 1, 3])
def test_cswin_strip_batches_preserve_results(device, strip_batch):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple GPU required")
    torch.manual_seed(8)
    module = CSwinAttention2D(32, 4).eval()
    torch.nn.init.normal_(module.proj.weight, std=0.1)
    x_cpu = torch.randn(2, 7, 11, 32, device="cpu")
    with torch.inference_mode():
        reference = copy.deepcopy(module).cpu().eval()(x_cpu)
        actual_module = module.to(device).eval()
        actual_module.inference_strip_batch = strip_batch
        actual = actual_module(x_cpu.to(device)).cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_runtime_configures_spatial_batches_only_on_metal(device):
    from corridorkey.runtime import InferenceConfig, configure_low_vram_inference

    module = torch.nn.Module()
    module._runtime_device = torch.device(device)
    module.stage1 = torch.nn.Module()
    module.swin = SwinAttention2D(32, 4, 4)
    module.cswin = CSwinAttention2D(32, 4)
    configure_low_vram_inference(module, InferenceConfig(low_vram_swin_window_batch=3))
    if device == "mps":
        assert module.swin.inference_window_batch == 3
        assert module.cswin.inference_strip_batch == 3
    else:
        assert not hasattr(module.swin, "inference_window_batch")
        assert not hasattr(module.cswin, "inference_strip_batch")
