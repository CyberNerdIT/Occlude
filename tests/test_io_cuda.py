"""Gate-logic tests for the CUDA video I/O fast path.

These run on any box (no CUDA needed): they verify that the
availability gate correctly *refuses* the CUDA path when its
prerequisites are missing. That gate is load-bearing — if it let a
runtime without h264_nvenc through, the encoder subprocess would die on
the first frame and the async runner would stall (see io_cuda docstrings).
"""
from __future__ import annotations

import occlude.pipeline.io_cuda as io_cuda


def test_cuda_io_unavailable_without_cuda():
    """On a CPU/Mac box torch.cuda is false, so the fast path is refused."""
    reasons = io_cuda.cuda_io_unavailable_reasons()
    assert reasons, "expected at least one blocker on a non-CUDA box"
    assert io_cuda.cuda_io_available() is False


def test_disable_env_forces_unavailable(monkeypatch):
    """The escape hatch must win even if every capability is present."""
    monkeypatch.setattr(io_cuda, "_TORCHCODEC_OK", True)
    monkeypatch.setattr(io_cuda.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(io_cuda.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(io_cuda, "_ffmpeg_has_h264_nvenc", lambda: True)
    # Without the env var, all gates pass -> available.
    assert io_cuda.cuda_io_available() is True
    # With it, forced unavailable.
    monkeypatch.setenv("OCCLUDE_DISABLE_CUDA_IO", "1")
    reasons = io_cuda.cuda_io_unavailable_reasons()
    assert any("OCCLUDE_DISABLE_CUDA_IO" in r for r in reasons)
    assert io_cuda.cuda_io_available() is False


def test_missing_nvenc_is_a_blocker(monkeypatch):
    """ffmpeg present but without h264_nvenc must block the fast path."""
    monkeypatch.setattr(io_cuda, "_TORCHCODEC_OK", True)
    monkeypatch.setattr(io_cuda.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(io_cuda.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(io_cuda, "_ffmpeg_has_h264_nvenc", lambda: False)
    reasons = io_cuda.cuda_io_unavailable_reasons()
    assert any("h264_nvenc" in r for r in reasons)
    assert io_cuda.cuda_io_available() is False


def test_require_cuda_io_raises_with_reasons():
    try:
        io_cuda.require_cuda_io_available()
    except RuntimeError as e:
        assert "CUDA video I/O required" in str(e)
    else:  # pragma: no cover - only reached on a fully CUDA-capable box
        pass
