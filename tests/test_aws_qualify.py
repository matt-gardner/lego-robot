import pytest

from so101.aws_qualify import qualify, tiny_recovery


def test_tiny_cpu_full_state_recovery_reproduces_next_update(tmp_path):
    tiny_recovery("cpu", tmp_path)


def test_qualification_refuses_cpu_fallback_before_reading_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("so101.aws_qualify.torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="exactly one"):
        qualify(tmp_path / "nonexistent.pt", 61000)
    assert not list(tmp_path.iterdir())


def test_qualification_refuses_multiple_visible_gpu_devices(tmp_path, monkeypatch):
    monkeypatch.setattr("so101.aws_qualify.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("so101.aws_qualify.torch.cuda.device_count", lambda: 2)
    with pytest.raises(ValueError, match="exactly one"):
        qualify(tmp_path / "nonexistent.pt", 61000)
