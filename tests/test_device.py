from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from corridorkey import device
from corridorkey.runtime import _phase1_setup, _phase2_setup


def test_cuda_remains_preferred(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    assert device.select_device().type == 'cuda'


def test_apple_gpu_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    assert device.select_device().type == 'mps'


def test_missing_gpu_has_actionable_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA GPU or an Apple Metal GPU'):
        device.select_device()


def test_low_memory_phases_keep_execution_device(monkeypatch):
    from corridorkey import runtime
    monkeypatch.setattr(runtime, 'empty_cache', Mock())
    monkeypatch.setattr(runtime, 'synchronize', Mock())
    class Module:
        def __init__(self):
            self.device = torch.device('cpu')
        def parameters(self):
            return iter([SimpleNamespace(device=self.device)])
        def to(self, target):
            self.device = torch.device(target)
        def cpu(self):
            self.to('cpu')
    model = SimpleNamespace(_runtime_device=torch.device('mps'),
                            foundations=Module(), stage1=Module(), stage2=Module())
    _phase1_setup(model, stage1_on_gpu=False)
    assert model.foundations.device.type == 'mps'
    assert model.stage1.device.type == model.stage2.device.type == 'cpu'
    _phase2_setup(model)
    assert model.foundations.device.type == model.stage1.device.type == 'cpu'
    assert model.stage2.device.type == 'mps'
    _phase1_setup(model)
    assert model.foundations.device.type == model.stage1.device.type == 'mps'
    assert model.stage2.device.type == 'cpu'


def test_cuda_budget_preserves_upstream_semantics(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda dev: (12 * 2**30, 16 * 2**30))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda dev: 2 * 2**30)
    assert device.memory_budget_gb('cuda') == 14


def test_mps_budget_never_negative(monkeypatch):
    monkeypatch.setattr(torch.mps, 'recommended_max_memory', lambda: 40 * 2**30)
    monkeypatch.setattr(torch.mps, 'driver_allocated_memory', lambda: 30 * 2**30)
    monkeypatch.setattr(torch.mps, 'current_allocated_memory', lambda: 2 * 2**30)
    assert device.memory_budget_gb('mps') == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple GPU required')
def test_nondivisible_area_resize_matches_cpu():
    x = torch.arange(2*3*128*128, dtype=torch.float32).reshape(2,3,128,128).remainder(251).to(torch.bfloat16)
    expected = torch.nn.functional.interpolate(x, size=(28,28), mode='area')
    actual = device.area_resize(x.to('mps'), (28,28)).cpu()
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
