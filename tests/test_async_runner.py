"""Orchestration tests for run_async_pipeline.

The async runner is the heart of the CUDA fast path, but its logic —
source ordering, EOF handling, max_frames, and error propagation — is
pure Python threading and runs identically on a Mac with no GPU. These
tests exercise it with in-memory fakes so the orchestration is validated
*before* an A100 session, where a deadlock would otherwise burn an
expensive run.

NOTE: this does NOT validate the GPU throughput / NVDEC / NVENC claims
(3x fps, >=70% util, hash==H1). Those still require a real CUDA run.
What's covered here is the decode/process/encode wiring only.

Every test runs the pipeline under a watchdog timeout so a regression
that reintroduces a hang FAILS fast instead of blocking the suite.
"""
from __future__ import annotations

import threading

import pytest

from occlude.pipeline.async_runner import run_async_pipeline


def _run_with_timeout(fn, seconds: float = 20.0):
    """Run *fn* on a thread; raise if it doesn't finish in *seconds*.

    Catches deadlocks: a hung pipeline would otherwise block forever.
    """
    box: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            fn()
        except BaseException as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=seconds)
    if t.is_alive():
        raise AssertionError("run_async_pipeline did not finish — deadlock")
    if "err" in box:
        raise box["err"]


def _decoder(n: int):
    """Returns a decode_next callable yielding n frames then EOF.

    Each frame is a mutable dict so process_batch can mark it in place,
    matching the real pipeline's in-place blur mutation.
    """
    state = {"i": 0}

    def decode_next():
        i = state["i"]
        if i >= n:
            return False, None
        state["i"] += 1
        return True, {"id": i, "processed": False}

    return decode_next


def test_preserves_source_order_and_processes_every_frame():
    n = 37  # not a multiple of batch_size -> exercises the partial tail
    encoded: list[dict] = []
    batches: list[int] = []

    def process_batch(batch):
        batches.append(len(batch))
        for _idx, frame in batch:
            frame["processed"] = True

    _run_with_timeout(lambda: run_async_pipeline(
        decode_next=_decoder(n),
        encode_frame=encoded.append,
        process_batch=process_batch,
        batch_size=4,
        max_frames=None,
        progress_update=lambda: None,
    ))

    assert [f["id"] for f in encoded] == list(range(n))   # FIFO order
    assert all(f["processed"] for f in encoded)            # processed exactly once
    assert all(b <= 4 for b in batches)                    # never over batch_size
    assert sum(batches) == n                               # every frame in some batch


def test_max_frames_stops_decoding():
    encoded: list[dict] = []

    _run_with_timeout(lambda: run_async_pipeline(
        decode_next=_decoder(1000),
        encode_frame=encoded.append,
        process_batch=lambda batch: None,
        batch_size=8,
        max_frames=5,
        progress_update=lambda: None,
    ))

    assert [f["id"] for f in encoded] == [0, 1, 2, 3, 4]


def test_progress_called_once_per_frame():
    counter = {"n": 0}

    def progress():
        counter["n"] += 1

    _run_with_timeout(lambda: run_async_pipeline(
        decode_next=_decoder(20),
        encode_frame=lambda f: None,
        process_batch=lambda batch: None,
        batch_size=3,
        max_frames=None,
        progress_update=progress,
    ))

    assert counter["n"] == 20


def test_encode_error_propagates_without_deadlock():
    """A dying encoder must surface its exception, not hang.

    This is the regression guard for the cleanup-path deadlock: the
    encode worker dies, the bounded encode queue fills, and a naive
    blocking sentinel put would block forever in the finally block.
    Enough frames (50) are supplied to saturate the queue.
    """
    def encode_frame(frame):
        raise RuntimeError("nvenc died")

    with pytest.raises(RuntimeError, match="nvenc died"):
        _run_with_timeout(lambda: run_async_pipeline(
            decode_next=_decoder(50),
            encode_frame=encode_frame,
            process_batch=lambda batch: None,
            batch_size=2,
            max_frames=None,
            progress_update=lambda: None,
        ))


def test_decode_error_propagates_without_deadlock():
    def decode_next():
        raise RuntimeError("nvdec died")

    with pytest.raises(RuntimeError, match="nvdec died"):
        _run_with_timeout(lambda: run_async_pipeline(
            decode_next=decode_next,
            encode_frame=lambda f: None,
            process_batch=lambda batch: None,
            batch_size=4,
            max_frames=None,
            progress_update=lambda: None,
        ))


def test_process_error_propagates_without_deadlock():
    """An exception in the main-thread perception/blur stage must not hang."""
    def process_batch(batch):
        raise RuntimeError("segformer died")

    with pytest.raises(RuntimeError, match="segformer died"):
        _run_with_timeout(lambda: run_async_pipeline(
            decode_next=_decoder(50),
            encode_frame=lambda f: None,
            process_batch=process_batch,
            batch_size=2,
            max_frames=None,
            progress_update=lambda: None,
        ))
