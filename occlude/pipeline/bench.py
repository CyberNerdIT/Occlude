"""Benchmark + golden-frame harness.

Runs the pipeline on the first ``seconds`` seconds of a video and
reports wall-clock fps, average GPU utilization, peak VRAM, and an
MD5 over a fixed set of output frames. The frame-hash MD5 is the
quality regression check: any pipeline change that doesn't intend to
shift output bits must keep it stable.

Used to baseline before each pipeline overhaul step and re-checked
after, so per-step deltas in throughput and quality are attributable
rather than averaged into a single end-of-project number.
"""
from __future__ import annotations

import hashlib
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch

# Frames to hash from the output silent video. Deliberately a fixed
# small set rather than every frame: the MD5 only has to flag bit-level
# drift, not catalog it, and reading 3 frames keeps the bench cheap.
# Indices that don't exist (short clips) are skipped — see _frame_hash.
_HASH_FRAME_INDICES = (0, 50, 150, 300, 600)


@dataclass
class BenchResult:
    """One run's measurements. Stringifies as a single progress.txt line."""

    input_path: Path
    seconds: float
    frames_processed: int
    wall_time_s: float
    fps: float
    gpu_util_avg: float | None       # None on non-CUDA
    gpu_util_samples: int
    peak_vram_mb: float | None       # None on non-CUDA
    frame_hash_md5: str
    device: str

    def format_line(self) -> str:
        util = f"{self.gpu_util_avg:.1f}%" if self.gpu_util_avg is not None else "n/a"
        vram = f"{self.peak_vram_mb:.0f}MB" if self.peak_vram_mb is not None else "n/a"
        return (
            f"bench device={self.device} input={self.input_path.name} "
            f"seconds={self.seconds:.1f} frames={self.frames_processed} "
            f"wall={self.wall_time_s:.2f}s fps={self.fps:.2f} "
            f"gpu_util_avg={util} ({self.gpu_util_samples} samples) "
            f"peak_vram={vram} hash={self.frame_hash_md5}"
        )


class _GpuUtilSampler:
    """Background thread sampling torch.cuda.utilization() every 100 ms.

    torch.cuda.utilization() goes through pynvml; if pynvml isn't
    available the call raises, and we degrade to no samples (the
    averaged value comes back as None). The sampler thread is a
    daemon and stops via the ``stop`` event.
    """

    def __init__(self) -> None:
        self._samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = torch.cuda.is_available()

    def __enter__(self) -> "_GpuUtilSampler":
        if self._enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN001
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while not self._stop.wait(0.1):
            try:
                self._samples.append(torch.cuda.utilization())
            except Exception:
                # pynvml not installed or query failed; abandon sampling.
                return

    @property
    def average(self) -> float | None:
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)

    @property
    def count(self) -> int:
        return len(self._samples)


def _frame_hash(video_path: Path, indices: tuple[int, ...]) -> str:
    """MD5 the raw BGR bytes of the requested frame indices."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path} for hashing")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    md5 = hashlib.md5()
    try:
        for idx in indices:
            if total > 0 and idx >= total:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            md5.update(frame.tobytes())
    finally:
        cap.release()
    return md5.hexdigest()


def run_benchmark(input_path: Path, seconds: float = 30.0) -> BenchResult:
    """Process the first ``seconds`` of ``input_path`` and measure.

    Audio mux is skipped — it's pure ffmpeg subprocess time and unrelated
    to the GPU pipeline being benchmarked. The silent output lives in a
    temp dir for the duration of the hash and is then discarded.
    """
    # Local import to avoid loading torch/transformers/models when
    # someone just runs `occlude --help`.
    from occlude.pipeline.video import VideoProcessor

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    max_frames = max(1, int(round(fps * seconds)))

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    with tempfile.TemporaryDirectory(prefix="occlude_bench_") as tmpdir:
        out_path = Path(tmpdir) / "bench_out.mp4"
        processor = VideoProcessor()

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with _GpuUtilSampler() as sampler:
            t0 = time.perf_counter()
            processor.process(
                input_path, out_path, max_frames=max_frames, skip_mux=True
            )
            wall = time.perf_counter() - t0

        peak_vram_mb: float | None
        if device == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_vram_mb = None

        digest = _frame_hash(out_path, _HASH_FRAME_INDICES)

    return BenchResult(
        input_path=input_path,
        seconds=seconds,
        frames_processed=max_frames,
        wall_time_s=wall,
        fps=max_frames / wall if wall > 0 else 0.0,
        gpu_util_avg=sampler.average,
        gpu_util_samples=sampler.count,
        peak_vram_mb=peak_vram_mb,
        frame_hash_md5=digest,
        device=device,
    )
