import torch

from utils.device import get_device


def test_get_device_uses_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device().type == "cpu"
