"""NVDEC decode + NVENC encode I/O for the CUDA fast path.

The pre-overhaul pipeline used OpenCV (libavcodec on CPU) for both
decode and encode. On A100 with the perception path itself only at
~33% GPU utilization, codec work on the CPU is the next bottleneck.
This module replaces both endpoints with GPU-accelerated codecs:

  - **Decode** via `torchcodec.decoders.VideoDecoder(device="cuda")`,
    which dispatches to NVDEC. Frames arrive as CUDA tensors; we
    convert RGB CHW → BGR HWC on-device and copy to host once per
    frame (matching the existing pipeline's numpy BGR HWC layout).
  - **Encode** via an `ffmpeg` subprocess with `-c:v h264_nvenc`. We
    pipe raw BGR24 bytes to its stdin. Roughly 5-10× the cv2 mp4v
    writer's encoding throughput on a 1280×720 stream.

    A `cuda_io_available()` gate falls back to the OpenCV path on systems
    without torchcodec or CUDA — Mac dev boxes and CI runners keep working
    unchanged. CUDA-required runs can inspect `cuda_io_unavailable_reasons()`
    and fail before processing starts.
"""
from __future__ import annotations

import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

try:
    from torchcodec.decoders import VideoDecoder as _TorchcodecVideoDecoder
    _TORCHCODEC_OK = True
except Exception:  # noqa: BLE001
    _TorchcodecVideoDecoder = None  # type: ignore[assignment]
    _TORCHCODEC_OK = False


_NVENC_PROBE_CACHE: bool | None = None


def _ffmpeg_has_h264_nvenc() -> bool:
    """Probe whether the installed ffmpeg ships ``h264_nvenc``.

    Cached for the process lifetime — running ``ffmpeg -encoders`` is a
    ~50 ms subprocess and the answer never changes mid-run. We *must*
    check this up-front: if ffmpeg lacks NVENC and we hand it
    ``-c:v h264_nvenc`` anyway, it dies during initialization, the
    encode worker silently raises a BrokenPipeError on the first write,
    and the main thread deadlocks on a full encode queue.
    """
    global _NVENC_PROBE_CACHE
    if _NVENC_PROBE_CACHE is not None:
        return _NVENC_PROBE_CACHE
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        _NVENC_PROBE_CACHE = False
        return False
    _NVENC_PROBE_CACHE = "h264_nvenc" in (result.stdout or "")
    return _NVENC_PROBE_CACHE


def cuda_io_available() -> bool:
    """True when both NVDEC decode and NVENC encode can be wired up.

    Checks: (1) torchcodec is importable, (2) torch sees a CUDA device,
    (3) the ``ffmpeg`` binary on PATH supports the ``h264_nvenc``
    encoder. Without (3) the encoder subprocess dies on first frame
    and the async runner deadlocks — see :func:`_ffmpeg_has_h264_nvenc`.

    The environment variable ``OCCLUDE_DISABLE_CUDA_IO=1`` forces a
    ``False`` result regardless of capabilities — useful as an escape
    hatch when NVDEC or NVENC stalls on a specific runtime (Colab's
    torchcodec + CUDA 12.8 combination is one known case as of late
    2025). With it set, the pipeline runs the sync cv2 path and you
    still get Steps 1+2.
    """
    return not cuda_io_unavailable_reasons()


def cuda_io_unavailable_reasons() -> list[str]:
    """Return human-readable blockers for the CUDA video I/O fast path."""
    import os

    reasons: list[str] = []
    if os.environ.get("OCCLUDE_DISABLE_CUDA_IO") == "1":
        reasons.append("OCCLUDE_DISABLE_CUDA_IO=1 is set")
    if not _TORCHCODEC_OK:
        reasons.append("torchcodec is not importable")
    if not torch.cuda.is_available():
        reasons.append("torch.cuda.is_available() is false")
    if shutil.which("ffmpeg") is None:
        reasons.append("ffmpeg is not on PATH")
    elif not _ffmpeg_has_h264_nvenc():
        reasons.append("ffmpeg does not advertise h264_nvenc")
    return reasons


def require_cuda_io_available() -> None:
    """Raise if the CUDA video I/O fast path cannot be used."""
    reasons = cuda_io_unavailable_reasons()
    if reasons:
        joined = "; ".join(reasons)
        raise RuntimeError(f"CUDA video I/O required but unavailable: {joined}")


class CudaVideoReader:
    """NVDEC-backed reader exposing the subset of cv2.VideoCapture the
    pipeline uses.

    Returns ``(bool, np.ndarray | None)`` from :meth:`read` so the
    existing main loop can swap us in for cv2.VideoCapture with no
    other code changes. The decode itself happens on the GPU via
    torchcodec; the host copy is one `.cpu().numpy()` per frame —
    the same H→D cost cv2 pays internally, just with NVDEC doing the
    actual codec work instead of libavcodec on the CPU.
    """

    def __init__(self, path: Path) -> None:
        if not _TORCHCODEC_OK:
            raise RuntimeError(
                "CudaVideoReader requires torchcodec; install it or "
                "fall back to cv2.VideoCapture"
            )
        self._decoder = _TorchcodecVideoDecoder(str(path), device="cuda")
        meta = self._decoder.metadata
        # torchcodec exposes a small typed metadata namespace; field
        # names match the upstream API.
        self.fps: float = float(getattr(meta, "average_fps", None) or 30.0)
        self.width: int = int(getattr(meta, "width", 0))
        self.height: int = int(getattr(meta, "height", 0))
        self.total_frames: int = int(getattr(meta, "num_frames", 0) or 0)
        self._iter = iter(self._decoder)

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            frame = next(self._iter)
        except StopIteration:
            return False, None
        # torchcodec yields (C, H, W) uint8 RGB on cuda. Convert to the
        # BGR HWC layout the rest of the pipeline expects. flip(0) on
        # the channel dim swaps R↔B (cheap, on-device), then permute
        # gets us HWC.
        bgr = frame.flip(0).permute(1, 2, 0).contiguous()
        return True, bgr.cpu().numpy()

    def release(self) -> None:
        # torchcodec cleans up the decoder when the object is GC'd; we
        # nil the iterator first so the underlying file handle releases.
        self._iter = iter([])
        self._decoder = None  # type: ignore[assignment]


@contextmanager
def cuda_video_writer(
    path: Path, width: int, height: int, fps: float
) -> Iterator:
    """NVENC-backed writer: yields a ``write(frame_bgr_np)`` callable.

    Spawns one long-running ``ffmpeg`` subprocess that reads raw BGR24
    frames from stdin and encodes via ``h264_nvenc``. The subprocess
    is closed on context exit; a non-zero return code raises with the
    last 500 chars of ffmpeg's stderr so failures surface loudly.

    Encoder settings: ``-preset p5 -rc vbr -cq 23 -b:v 0``. p5 is the
    middle of NVENC's quality/perf curve (p1 fastest, p7 best); cq 23
    is a sane visual default that mirrors libx264 crf 23. ``-b:v 0``
    lets ``-cq`` drive the bitrate.
    """
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}",
        "-i", "-",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-rc", "vbr",
        "-cq", "23",
        "-b:v", "0",
        str(path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )

    def _write(frame_bgr: np.ndarray) -> None:
        assert proc.stdin is not None
        proc.stdin.write(frame_bgr.tobytes())

    try:
        yield _write
    finally:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                # ffmpeg already died; let communicate() surface why.
                pass
        try:
            _, stderr = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
        if proc.returncode != 0:
            tail = (stderr or b"").decode(errors="replace")[-500:]
            raise RuntimeError(f"ffmpeg/h264_nvenc failed: {tail}")
