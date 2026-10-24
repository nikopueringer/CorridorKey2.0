import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from corridorkey import attention


@pytest.fixture(autouse=True)
def tiny_budget(monkeypatch):
    # Force splitting on small tensors so both the batch and query paths run.
    monkeypatch.setattr(attention, "_MAX_SCORE_ELEMENTS", 2048)


def _randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _mask(shape, dtype, seed=1):
    mask = torch.rand(shape, generator=torch.Generator().manual_seed(seed)) > 0.3
    mask[..., 0] = True  # keep every row attendable (avoid all-masked NaNs)
    if dtype is torch.float32:
        return torch.zeros(shape).masked_fill(~mask, float("-inf"))
    return mask


def _check(q, k, v, attn_mask=None, **kwargs):
    # Compare against the math backend: the only one available on GPUs that
    # take the bounded path.
    with sdpa_kernel([SDPBackend.MATH]):
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, **kwargs)
        got = attention._bounded_sdpa(q, k, v, attn_mask, kwargs)
    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("lead", [(1, 2), (5, 2), (3, 4, 2), (7,), (64, 1)])
def test_matches_full_sdpa_without_mask(lead):
    _check(_randn(*lead, 40, 16), _randn(*lead, 24, 16, seed=1), _randn(*lead, 24, 16, seed=2))


def test_query_only_split_for_2d_inputs():
    _check(_randn(200, 16), _randn(24, 16, seed=1), _randn(24, 16, seed=2))


@pytest.mark.parametrize("mask_shape", [
    (5, 1, 40, 24),   # per-batch, broadcast over heads
    (5, 2, 40, 24),   # full
    (1, 1, 40, 24),   # broadcast over batch
    (2, 40, 24),      # 3-D: heads on dim 0, must not be sliced by batch
    (40, 24),         # 2-D
    (5, 2, 1, 24),    # broadcast over queries (key padding)
])
@pytest.mark.parametrize("dtype", [torch.bool, torch.float32])
def test_matches_full_sdpa_with_mask(mask_shape, dtype):
    q, k, v = _randn(5, 2, 40, 16), _randn(5, 2, 24, 16, seed=1), _randn(5, 2, 24, 16, seed=2)
    _check(q, k, v, _mask(mask_shape, dtype))


def test_rank_misaligned_mask_where_heads_equal_batch():
    # dim 0 of a 3-D mask is heads; slicing it as batch would be wrong.
    q, k, v = _randn(4, 4, 40, 16), _randn(4, 4, 24, 16, seed=1), _randn(4, 4, 24, 16, seed=2)
    _check(q, k, v, _mask((4, 40, 24), torch.bool))


def test_lower_rank_kv_broadcast_over_batch():
    _check(_randn(4, 4, 40, 16), _randn(4, 24, 16, seed=1), _randn(4, 24, 16, seed=2))


def test_broadcast_kv_over_batch():
    _check(_randn(6, 2, 40, 16), _randn(1, 2, 24, 16, seed=1), _randn(1, 2, 24, 16, seed=2))


def test_kv_widen_batch_beyond_q():
    _check(_randn(1, 2, 40, 16), _randn(6, 2, 24, 16, seed=1), _randn(6, 2, 24, 16, seed=2))


def test_v_widens_batch_beyond_q_and_k():
    _check(_randn(1, 2, 40, 16), _randn(1, 2, 24, 16, seed=1), _randn(6, 2, 24, 16, seed=2))


def test_v_broadcast_counts_toward_budget(monkeypatch):
    calls = []
    real = F.scaled_dot_product_attention
    monkeypatch.setattr(attention.F, "scaled_dot_product_attention",
                        lambda q, k, v, **kw: calls.append(
                            torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2]).numel()
                            * q.shape[-2] * k.shape[-2]) or real(q, k, v, **kw))
    attention._bounded_sdpa(_randn(1, 2, 40, 16), _randn(1, 2, 24, 16, seed=1),
                            _randn(6, 2, 24, 16, seed=2), None, {})
    assert max(calls) <= attention._MAX_SCORE_ELEMENTS


def test_swin2d_shifted_window_shape():
    # Swin2D: q (B_eff, nW, H, N, D) with mask (1, nW, 1, N, N).
    q = _randn(3, 4, 2, 16, 16)
    _check(q, _randn(3, 4, 2, 16, 16, seed=1), _randn(3, 4, 2, 16, 16, seed=2),
           _mask((1, 4, 1, 16, 16), torch.float32))


def test_multi_entry_batch_chunks_with_mask(monkeypatch):
    monkeypatch.setattr(attention, "_MAX_SCORE_ELEMENTS", 3 * 2 * 40 * 24)
    q, k, v = _randn(8, 2, 40, 16), _randn(8, 2, 24, 16, seed=1), _randn(8, 2, 24, 16, seed=2)
    _check(q, k, v, _mask((8, 1, 40, 24), torch.bool))


def test_scale_kwarg():
    _check(_randn(5, 2, 40, 16), _randn(5, 2, 24, 16, seed=1), _randn(5, 2, 24, 16, seed=2),
           scale=0.37)


def test_bf16_under_autocast_keeps_dtype():
    q, k, v = (t.bfloat16() for t in (_randn(5, 2, 40, 16), _randn(5, 2, 24, 16, seed=1),
                                     _randn(5, 2, 24, 16, seed=2)))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert attention._bounded_sdpa(q, k, v, None, {}).dtype == torch.bfloat16


def test_routing(monkeypatch):
    q = _randn(2, 2, 8, 16)
    assert not attention._needs_bounding(q, 0.0, False, {})  # CPU tensors: never bounded
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(attention, "has_fused_sdpa", lambda device, dtype: False)
    assert attention._needs_bounding(q, 0.0, False, {})
    assert not attention._needs_bounding(q, 0.0, True, {})
    assert not attention._needs_bounding(q, 0.1, False, {})
    monkeypatch.setattr(attention, "has_fused_sdpa", lambda device, dtype: True)
    assert not attention._needs_bounding(q, 0.0, False, {})


def test_cuda_builds_always_pass_through(monkeypatch):
    q = _randn(2, 2, 8, 16)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(attention, "has_fused_sdpa", lambda device, dtype: False)
    assert not attention._needs_bounding(q, 0.0, False, {})


def test_inconclusive_probe_bounds_without_caching(monkeypatch):
    monkeypatch.setattr(attention, "_fused_cache", {})
    monkeypatch.setattr(attention, "_probe_fused_sdpa", lambda device, dtype: None)
    assert not attention.has_fused_sdpa(torch.device("cpu"), torch.float32)
    assert attention._fused_cache == {}


@pytest.mark.parametrize("fail_on", ["alloc", "sdpa"])
def test_probe_oom_is_inconclusive(monkeypatch, fail_on):
    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError("out of memory")
    target = torch if fail_on == "alloc" else attention.F
    name = "zeros" if fail_on == "alloc" else "scaled_dot_product_attention"
    monkeypatch.setattr(target, name, oom)
    assert attention._probe_fused_sdpa(torch.device("cpu"), torch.float32) is None


def test_probe_uses_autocast_dtype(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(attention, "has_fused_sdpa",
                        lambda device, dtype: seen.append(dtype) or True)
    q = _randn(2, 2, 8, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        attention._needs_bounding(q, 0.0, False, {})
    attention._needs_bounding(q, 0.0, False, {})
    assert seen == [torch.bfloat16, torch.float32]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("mask_shape", [None, (4, 1, 80, 24), (1, 1, 80, 24)])
def test_public_sdpa_on_gpu_matches_math_backend(mask_shape):
    # Through the public wrapper, under bf16 autocast as in inference. On GPUs
    # without fused kernels the tiny budget splits by batch entry, and each
    # entry (2 x 80 x 24 scores) again by query.
    dev = torch.device("cuda")
    q, k, v = (t.to(dev) for t in (_randn(4, 2, 80, 16), _randn(4, 2, 24, 16, seed=1),
                                   _randn(4, 2, 24, 16, seed=2)))
    mask = None if mask_shape is None else _mask(mask_shape, torch.bool).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        got = attention.sdpa(q, k, v, attn_mask=mask)
        with sdpa_kernel([SDPBackend.MATH]):
            expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert got.dtype == expected.dtype
    torch.testing.assert_close(got, expected)
