"""Producer/consumer pipelining for the CUDA fast path.

The synchronous loop pays for decode → perception → blur → encode
strictly in series even though three of those four stages can overlap
on the A100:

  - NVDEC (decode) runs on the GPU's dedicated decoder engine — not
    on SMs — so it overlaps with SegFormer compute for free *if* a
    CPU thread is there to dispatch the next frame.
  - NVENC (encode) is the same story, on the dedicated encoder engine,
    with the host-side overhead being one ``.cpu().numpy().tobytes()``
    + a subprocess write — both happily run on a separate thread.
  - SegFormer + tracker + blur stay on the main thread on the default
    stream (the tracker is stateful in source order, so we don't try
    to parallelize across frames).

This module wires those three stages onto three threads with bounded
queues. Source order is preserved by construction: each queue is FIFO
and only the main thread dequeues into the perception batch, so
frames leave in arrival order. No reorder buffer is needed.

Explicit CUDA streams are deliberately not used. They'd only help if
SegFormer and the blur composite could overlap on SMs for the same
frame, which they can't — blur depends on segmentation output for the
same frame. Cross-frame SM overlap is what ``torch.compile``'s kernel
fusion already buys.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Callable

# Sentinel pushed onto a queue to mark "no more items".
_SENTINEL = object()


def run_async_pipeline(
    *,
    decode_next: Callable[[], tuple[bool, Any]],
    encode_frame: Callable[[Any], None],
    process_batch: Callable[[list[tuple[int, Any]]], None],
    batch_size: int,
    max_frames: int | None,
    progress_update: Callable[[], None],
) -> None:
    """Run the three-stage pipeline.

    Parameters
    ----------
    decode_next:
        Pull the next decoded frame; returns ``(ret, frame)`` matching
        cv2.VideoCapture.read so this works with both CudaVideoReader
        and cv2.VideoCapture.
    encode_frame:
        Write one blurred frame to the encoder. Runs on the encode
        thread, so anything it does (subprocess stdin write,
        ``.tobytes()``) overlaps with the main thread's GPU work.
    process_batch:
        Synchronous callable taking ``[(frame_idx, frame_bgr), ...]``
        of length ``≤ batch_size``. Mutates each ``frame_bgr`` in
        place with the blur composite. The runner enqueues each frame
        to the encode queue in source order after the call returns.
    batch_size:
        Perception batch size B. Decode queue is sized 2×B so NVDEC
        keeps one batch buffered ahead of perception.
    max_frames:
        Stop after this many frames (None = decode until EOF).
    progress_update:
        Called once per blurred frame after blur + before enqueue to
        the encoder. Drives the tqdm progress bar.
    """
    # 2×B keeps one batch buffered ahead of the perception thread without
    # ballooning memory. Single-batch buffering would still leave NVDEC
    # idle during the second half of each perception pass.
    decode_q: queue.Queue = queue.Queue(maxsize=max(2, 2 * batch_size))
    encode_q: queue.Queue = queue.Queue(maxsize=max(2, 2 * batch_size))

    # First exception in each worker; main thread re-raises after join
    # so we never silently swallow a decoder/encoder crash.
    errors: dict[str, BaseException | None] = {"decode": None, "encode": None}

    def _decode_worker() -> None:
        try:
            idx = 0
            while True:
                if max_frames is not None and idx >= max_frames:
                    break
                ret, frame = decode_next()
                if not ret:
                    break
                decode_q.put((idx, frame))
                idx += 1
        except BaseException as e:  # noqa: BLE001
            errors["decode"] = e
        finally:
            decode_q.put(_SENTINEL)

    def _encode_worker() -> None:
        try:
            while True:
                item = encode_q.get()
                if item is _SENTINEL:
                    return
                _idx, frame = item
                encode_frame(frame)
        except BaseException as e:  # noqa: BLE001
            errors["encode"] = e

    dec_th = threading.Thread(target=_decode_worker, name="occlude-decode", daemon=True)
    enc_th = threading.Thread(target=_encode_worker, name="occlude-encode", daemon=True)
    dec_th.start()
    enc_th.start()

    try:
        eof = False
        while not eof:
            # Gather one perception batch from the decode queue. A
            # SENTINEL inside the batch flips the eof flag so the
            # batch we just collected still gets processed.
            batch: list[tuple[int, Any]] = []
            while len(batch) < batch_size:
                item = decode_q.get()
                if item is _SENTINEL:
                    eof = True
                    break
                batch.append(item)

            if not batch:
                break

            # Process the batch synchronously; mutates frames in place.
            process_batch(batch)

            # Submit to encoder in source order. progress_update happens
            # before encode_q.put so the bar moves at the pace of GPU
            # work, not at the pace of the encoder draining (which can
            # be I/O-bound and uneven).
            #
            # The put is timeout-driven specifically so we don't
            # deadlock when the encode worker dies: a dead worker stops
            # draining the queue, the queue fills, and a plain put()
            # would block forever. With a timeout we re-check the error
            # dict every second and surface the underlying exception.
            for idx, frame in batch:
                progress_update()
                while True:
                    try:
                        encode_q.put((idx, frame), timeout=1.0)
                        break
                    except queue.Full:
                        if errors["encode"] is not None:
                            raise errors["encode"]
                        if errors["decode"] is not None:
                            raise errors["decode"]

            # Also check between batches so a stalled decoder is caught
            # even before the encode queue fills.
            if errors["decode"] is not None:
                raise errors["decode"]
            if errors["encode"] is not None:
                raise errors["encode"]
    finally:
        # Bounded put: if we reach here via an encode-worker crash, the
        # worker has stopped draining and the bounded queue may be full.
        # A plain put(_SENTINEL) would then block forever — a deadlock in
        # the very cleanup path that's supposed to surface the error. The
        # sentinel only matters for a *live* worker (clean shutdown); a
        # dead worker doesn't need it, so giving up after a short wait is
        # correct.
        try:
            encode_q.put(_SENTINEL, timeout=5)
        except queue.Full:
            pass
        dec_th.join(timeout=10)
        enc_th.join(timeout=60)

    if errors["decode"] is not None:
        raise errors["decode"]
    if errors["encode"] is not None:
        raise errors["encode"]
