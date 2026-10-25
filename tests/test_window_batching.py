"""Check independent batching against nonzero attention, including boundary masks."""
import pytest
import torch
import copy

from corridorkey.blocks import (CSwinAttention2D, SwinAttention2D,
                               _shifted_region_groups_2d_cached,
                               _shifted_region_ids_2d)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('mask_kind', ['padding', 'explicit', 'groups', 'ids'])
def test_swin_batching_preserves_masked_results(device, mask_kind):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Apple GPU required')
    torch.manual_seed(7)
    module = SwinAttention2D(32, 4, 4).eval()
    # The production initialization is zero; a zero projection would conceal
    # incorrect masking or chunk placement in this test.
    torch.nn.init.normal_(module.proj.weight, std=0.1)
    module = module.to(device)
    x = torch.randn(2, 9, 11, 32, device=device)
    kwargs = {}
    if mask_kind == 'groups':
        kwargs['region_groups'] = _shifted_region_groups_2d_cached(9, 11, 12, 12, 4, 2, 2)
    elif mask_kind == 'ids':
        ids = _shifted_region_ids_2d(9, 11, 12, 12, 4, 2, 2, torch.device(device))
        kwargs['region_ids'] = ids.repeat(2, 1)
    elif mask_kind == 'explicit':
        mask = torch.ones(9, 16, 16, dtype=torch.bool, device=device)
        mask[:, :, -3:] = False
        kwargs['attn_mask'] = mask
    with torch.inference_mode():
        if device == 'mps' and mask_kind == 'explicit':
            # The old 5D masked MPS SDPA call itself raises a stride error.
            # Use its CPU result to check the new 4D batched equivalent.
            reference = copy.deepcopy(module).cpu()(
                x.cpu(), attn_mask=kwargs['attn_mask'].cpu()).to(device)
        else:
            reference = module(x, **kwargs)
        module.inference_window_batch = 5  # crosses the nine-window image boundary
        actual = module(x, **kwargs)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_cswin_strip_batches_preserve_results(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Apple GPU required')
    torch.manual_seed(8)
    module = CSwinAttention2D(32, 4).eval()
    torch.nn.init.normal_(module.proj.weight, std=0.1)
    module = module.to(device)
    x = torch.randn(2, 7, 11, 32, device=device)
    with torch.inference_mode():
        reference = module(x)
        module.inference_strip_batch = 3
        actual = module(x)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('device', ['mps', 'cuda'])
def test_runtime_configures_spatial_batches_only_on_metal(device):
    from corridorkey.runtime import InferenceConfig, configure_low_vram_inference
    module = torch.nn.Module()
    module._runtime_device = torch.device(device)
    module.stage1 = torch.nn.Module()
    module.swin = SwinAttention2D(32, 4, 4)
    module.cswin = CSwinAttention2D(32, 4)
    configure_low_vram_inference(module, InferenceConfig(low_vram_swin_window_batch=3))
    if device == 'mps':
        assert module.swin.inference_window_batch == 3
        assert module.cswin.inference_strip_batch == 3
    else:
        assert not hasattr(module.swin, 'inference_window_batch')
        assert not hasattr(module.cswin, 'inference_strip_batch')
