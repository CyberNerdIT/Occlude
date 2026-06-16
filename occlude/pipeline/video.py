"""Stage 6 — Video pipeline.

Glues :class:`occlude.pipeline.perception.Perception` and
:class:`occlude.pipeline.rules.RuleEngine` to per-frame blur application,
temporal smoothing (IoU-matched carry-forward), and ffmpeg audio mux.
The :class:`VideoProcessor` turns an input video file into an output
video file with immodest people blurred and the original audio track
preserved.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os as _os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None

from occlude.pipeline.async_runner import run_async_pipeline
from occlude.pipeline.blur import (
    PIXELATE_BLOCKS_ACROSS,
    PIXELATE_MIN_BLOCK_PX,
    SILHOUETTE_DILATE_FRAC,
    SILHOUETTE_DILATE_MIN_PX,
    SILHOUETTE_FEATHER_FRAC,
    SILHOUETTE_FEATHER_MIN_PX,
    _BLUR_DEVICE,
    _prepare_blur_mask_torch,
    _scale_blur_mask,
    apply_blur_mask,
    blur_region,
    prepare_blur_mask,
)
from occlude.pipeline.config import (
    DEFAULT_BLUR_KERNEL,
    DEFAULT_PERCEPTION_BATCH_CUDA,
    DEFAULT_PERCEPTION_BATCH_OTHER,
)
from occlude.pipeline.io_cuda import (
    CudaVideoReader,
    cuda_io_available,
    cuda_io_unavailable_reasons,
    cuda_video_writer,
    require_cuda_io_available,
)
from occlude.pipeline.perception import Perceiver, Perception, Person
from occlude.pipeline.rules import RuleEngine

__all__ = [
    "PIXELATE_BLOCKS_ACROSS",
    "PIXELATE_MIN_BLOCK_PX",
    "SILHOUETTE_DILATE_FRAC",
    "SILHOUETTE_DILATE_MIN_PX",
    "SILHOUETTE_FEATHER_FRAC",
    "SILHOUETTE_FEATHER_MIN_PX",
    "Tracker",
    "VideoProcessor",
    "_BLUR_DEVICE",
    "_prepare_blur_mask_torch",
    "apply_blur_mask",
    "blur_region",
    "prepare_blur_mask",
]

# macOS-specific malloc pressure relief. vmmap on a long-running
# occlude shows the working set is small (~400 MB dirty) but the
# heap is fragmented across 8+ GB of MALLOC_LARGE / MALLOC_SMALL
# regions, most of it compressed (Activity Monitor footprint hit 13 GB
# at frame 39). Calling `malloc_zone_pressure_relief(0, 0)` per frame
# asks every malloc zone to return reusable pages to the kernel, which
# is the documented macOS API for explicitly giving freed-but-cached
# heap back to the OS. ~5–20 ms per call. No-op on non-Darwin.
_pressure_relief = None
if sys.platform == "darwin":
    _libc = ctypes.CDLL(ctypes.util.find_library("c"))
    _libc.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _libc.malloc_zone_pressure_relief.restype = ctypes.c_size_t
    _pressure_relief = _libc.malloc_zone_pressure_relief

# IoU threshold for matching a current-frame detection to a previous
# track. >0.5 misses moderate motion (head turns, walking); <0.2 risks
# false-matching unrelated nearby people.
IOU_THRESHOLD = 0.3

# Carry-forward window. If a person was flagged for blur in frame N
# and detection drops them in frame N+1, blur their last-known bbox
# for K more frames before giving up. Spec §Step 7 specifies "frames
# N+1 and N+2" → K=2, but on real video (`laughing_people.mp4`)
# YOLO drops far/small subjects every few frames, producing visible
# blur flicker on background people. Bumped to 10 (~0.3s at 30fps).
CARRY_FORWARD_FRAMES = 10

# Bbox smoothing window. Per-track moving-average over this many
# recent raw YOLO bboxes removes the few-pixel jitter that otherwise
# makes the blur "breathe" frame-to-frame.
BBOX_SMOOTHING_FRAMES = 5

# IoU between a track's new raw bbox and its current smoothed bbox.
# Below this, the bbox jumped enough that the smoothing window is
# mostly stale (typically a scene cut where the IoU-matcher reused
# the track id on tenuous ≥0.3 overlap, but the subject is at a
# substantially different position/size). Reset the bbox history so
# the blur snaps to the new position instead of lagging through the
# 5-frame moving average. Gender votes and blur_history are kept —
# only position smoothing resets.
BBOX_RESET_IOU = 0.6

# Per-track blur-decision smoothing. The rule layer's per-frame
# decision can flicker (cross-check fires when the segmenter happens
# to tag hair-below-face well, doesn't on the next frame, etc.).
# Vote on the last K decisions: if majority say blur, blur this
# frame regardless of the current rule output. Same pattern as the
# gender voting (Finding 9). Eliminates the "comes and goes" blur on
# subjects whose rule decision is borderline.
BLUR_VOTE_WINDOW = 10
BLUR_VOTE_MIN_HISTORY = 3   # don't apply majority below this

# A track whose smoothed bbox comes within this many pixels of any
# frame border is treated as exiting the frame. Carry-forward (which
# exists to bridge brief mid-frame detection drops, not to trail a
# subject who has walked off-screen) is killed immediately for such
# tracks, so the blur stops on the frame the subject leaves instead of
# smearing their last silhouette for CARRY_FORWARD_FRAMES more frames.
# YOLO clamps a partially-off-screen person's bbox to the frame, so an
# exiting subject's box sits flush against the border — a few px of
# slack absorbs the bbox-smoothing moving average.
EDGE_TOUCH_PX = 6

# Per-track gender re-classification cadence. Caching gender once per
# track (Finding 6) bounded the heap, but a single misclassification
# at first detection then locks the track for its entire lifetime —
# the canonical failure was a short-haired woman at the rooftop scene
# in `laughing_people.mp4` who came back M @ 0.91 on her first
# detection (face partially behind a glass) and stayed unblurred for
# 820 frames. Re-classifying every N frames and majority-voting over
# the last K high-confidence answers lets a wrong first call get
# corrected. N=30 (~1s at 30fps) keeps overhead bounded; K=5 means
# ~5× the InsightFace calls per long track instead of 1×.
RECLASSIFY_INTERVAL_FRAMES = 30
GENDER_VOTE_WINDOW = 5
# New tracks burst-reclassify every frame until they have this many
# votes, then fall back to RECLASSIFY_INTERVAL_FRAMES. A bad first
# vote (motion blur on a scene cut, partial face turn) used to lock
# a wrong gender call for a full 30-frame window — exactly the
# "subject unblurred for ~1 sec on scene change" symptom. With burst,
# a 3-vote majority forms within ~3 frames (≈0.1 s).
GENDER_VOTE_BURST = 3

# Cadence for macOS malloc_zone_pressure_relief(0, 0), which hints the
# kernel to reclaim dirty pages without stalling the process. GC and
# MPS/CUDA cache eviction are handled inside Perception._segment_batch
# and need not repeat here. Every 25 frames (~1 s at 25 fps) keeps
# long-video RSS bounded with negligible per-frame overhead.
MEMORY_CLEANUP_EVERY = 25

# Optional path to an append-only memory log. tqdm's progress bar
# overwrites tqdm.write() output in narrow terminals, so we route the
# memory checkpoints to a separate file when this env var is set.
MEM_LOG_PATH = _os.environ.get("OCCLUDE_MEM_LOG")
# Optional per-track decision log (debug). Each line is a single
# track-update event: frame, track_id, new-or-reused, gender,
# face_det_score, blur decision, reason. Used to diagnose why a
# specific person in a video isn't getting the expected blur (cache
# stuck on a wrong first-frame classification, etc.).
TRACK_LOG_PATH = _os.environ.get("OCCLUDE_TRACK_LOG")


@dataclass
class _TrackedPerson:
    bbox: tuple[int, int, int, int]                       # most recent raw bbox (used for IoU match)
    bbox_history: deque                                    # last N raw bboxes; smoothed bbox = mean
    blur: bool
    carry_remaining: int
    # Cached gender + face_det_score from InsightFace at first
    # detection. Reused across all subsequent frames the track is
    # matched. Without this cache, running InsightFace's ONNX face
    # detection every frame leaks ~45 MB per person-call into the
    # CPU heap (see docs/07-video-pipeline.md Finding 6) — on a
    # 6-person frame at 30 fps that's the difference between bounded
    # ~3 GB footprint and Jetsam at frame ~90.
    gender: str | None = None
    face_det_score: float = 0.0
    # Rolling vote window of recent (gender, score, age) classifications
    # for this track. Re-classified every RECLASSIFY_INTERVAL_FRAMES;
    # majority of high-confidence votes determines the active gender
    # used by the rule layer. Recovers from a wrong first-frame call.
    # Age rides the same window (median of high-conf votes) so the
    # child exemption inherits the same noise-smoothing and recovery.
    gender_votes: deque = field(default_factory=lambda: deque(maxlen=GENDER_VOTE_WINDOW))
    last_classify_frame: int = -1
    # Rolling per-frame blur decisions for temporal smoothing.
    # Eliminates flickering when the rule decision is borderline.
    blur_history: deque = field(default_factory=lambda: deque(maxlen=BLUR_VOTE_WINDOW))
    last_smoothed: tuple[int, int, int, int] = (0, 0, 0, 0)
    # Silhouette mask from the most recent detection. Kept so
    # carry-forward frames (where there's no fresh seg_mask) still
    # blur along the subject outline instead of falling back to a
    # rectangle. Resized onto last_smoothed at apply time.
    last_seg_mask: np.ndarray | None = None


def _smoothed_bbox(history: deque) -> tuple[int, int, int, int]:
    arr = np.asarray(history, dtype=np.float32)
    mean = arr.mean(axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]), int(mean[3]))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _touches_edge(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int] | None,
) -> bool:
    """True if *bbox* sits within EDGE_TOUCH_PX of any frame border.

    Returns False when *frame_shape* is None so callers that don't have
    the frame dimensions (unit tests exercising pure carry-forward
    timing) keep the original behaviour.
    """
    if frame_shape is None:
        return False
    h, w = frame_shape
    x1, y1, x2, y2 = bbox
    return (
        x1 <= EDGE_TOUCH_PX
        or y1 <= EDGE_TOUCH_PX
        or x2 >= w - EDGE_TOUCH_PX
        or y2 >= h - EDGE_TOUCH_PX
    )


class Tracker:
    """IoU tracker owning gender voting, blur voting, and carry-forward.

    Call :meth:`update` once per perception frame.  Returns a list of
    ``(smoothed_bbox, seg_mask)`` pairs — the blur targets for that frame.
    The same list can be replayed on perception-skipped frames.
    """

    def __init__(
        self,
        perceiver: Perceiver,
        rules: RuleEngine,
        *,
        carry_forward_frames: int = CARRY_FORWARD_FRAMES,
        gender_vote_burst: int = GENDER_VOTE_BURST,
        gender_vote_window: int = GENDER_VOTE_WINDOW,
        blur_vote_window: int = BLUR_VOTE_WINDOW,
        bbox_smoothing_frames: int = BBOX_SMOOTHING_FRAMES,
    ) -> None:
        self._perceiver = perceiver
        self._rules = rules
        self._tracks: dict[int, _TrackedPerson] = {}
        self._next_id = 0
        self._carry_forward_frames = carry_forward_frames
        self._gender_vote_burst = gender_vote_burst
        self._gender_vote_window = gender_vote_window
        self._blur_vote_window = blur_vote_window
        self._bbox_smoothing_frames = bbox_smoothing_frames
        self._log_fh: IO[str] | None = None

    def update(
        self,
        frame_idx: int,
        people: list[Person],
        frame_shape: tuple[int, int] | None = None,
    ) -> list[tuple[tuple[int, int, int, int], np.ndarray | None, float]]:
        """Match *people* to existing tracks; return blur targets for this frame.

        Each target is ``(smoothed_bbox, seg_mask, alpha)``. ``alpha`` is
        1.0 for tracks detected this frame and decays toward 0 across the
        carry-forward window so a subject who briefly drops out fades
        instead of holding a crisp ghost silhouette. *frame_shape*
        ``(h, w)`` enables the edge-exit carry kill; omit it (tests) to
        keep pure carry-forward timing.
        """
        matched: set[int] = set()
        blur_targets: list[
            tuple[tuple[int, int, int, int], np.ndarray | None, float]
        ] = []

        for person in people:
            bbox = tuple(int(v) for v in person.bbox)

            best_id, best_iou = None, 0.0
            for tid, tp in self._tracks.items():
                if tid in matched:
                    continue
                iou = _iou(bbox, tp.bbox)
                if iou > best_iou and iou >= IOU_THRESHOLD:
                    best_iou, best_id = iou, tid

            if best_id is None:
                g, s, a = self._perceiver.classify(person.crop)
                votes = deque(maxlen=self._gender_vote_window)
                votes.append((g, s, a))
                blur_history = deque(maxlen=self._blur_vote_window)
                last_classify = frame_idx
                best_id = self._next_id
                self._next_id += 1
                history = deque(maxlen=self._bbox_smoothing_frames)
                track_origin = "new"
            else:
                prev = self._tracks[best_id]
                votes = prev.gender_votes
                blur_history = prev.blur_history
                last_classify = prev.last_classify_frame
                if _iou(bbox, prev.last_smoothed) < BBOX_RESET_IOU:
                    history = deque(maxlen=self._bbox_smoothing_frames)
                else:
                    history = prev.bbox_history
                needs_reclassify = (
                    len(votes) < self._gender_vote_burst
                    or frame_idx - last_classify >= RECLASSIFY_INTERVAL_FRAMES
                )
                if needs_reclassify:
                    g, s, a = self._perceiver.classify(person.crop)
                    votes.append((g, s, a))
                    last_classify = frame_idx
                    track_origin = "reclassified"
                else:
                    track_origin = "reused"

            high_conf = [(g, s, a) for g, s, a in votes if s >= 0.50]
            if high_conf:
                f_count = sum(1 for g, _, _ in high_conf if g == "F")
                m_count = sum(1 for g, _, _ in high_conf if g == "M")
                # Over-blur bias on the *startup* of a track. Until the
                # vote window is full the track is young and unstable;
                # InsightFace exposes no calibrated gender probability
                # (docs/04 Finding 2) so a confidently-wrong "M" on a
                # woman is indistinguishable from a correct "M" and would
                # otherwise lock her unblurred until the 30-frame
                # reclassify cadence flips the majority (~2 s — the exact
                # "blurs late" symptom). The tool already commits to this
                # asymmetry elsewhere (no-Face → over-blur); a single
                # early high-conf F vote settles it as F immediately.
                # False-F (extra blur on a man) is the accepted cost; it
                # self-corrects once the window matures and majority
                # rules again.
                immature = len(votes) < self._gender_vote_window
                if immature and f_count > 0:
                    person.gender = "F"
                elif f_count > m_count:
                    person.gender = "F"
                elif m_count > f_count:
                    person.gender = "M"
                else:
                    person.gender = max(high_conf, key=lambda x: x[1])[0]
                person.face_det_score = max(s for _, s, _ in high_conf)
                ages = [a for _, _, a in high_conf if a is not None]
                # Median, not mean: robust to InsightFace's occasional
                # wild age outliers (docs/04 ±5–10 yr) so one bad frame
                # can't tip a track in/out of the child band.
                person.age = float(np.median(ages)) if ages else None
            else:
                person.gender = None
                person.face_det_score = 0.0
                person.age = None
            history.append(bbox)
            matched.add(best_id)

            decision = self._rules.decide(person)
            blur_history.append(decision.blur)
            if len(blur_history) >= BLUR_VOTE_MIN_HISTORY:
                sticky_blur = sum(blur_history) > len(blur_history) / 2
            else:
                sticky_blur = decision.blur

            if self._log_fh is not None:
                self._log_fh.write(
                    f"frame={frame_idx} tid={best_id} {track_origin} "
                    f"bbox={bbox} gender={person.gender} "
                    f"score={person.face_det_score:.2f} "
                    f"votes={list(votes)} "
                    f"rule_blur={decision.blur} sticky={sticky_blur} "
                    f"reason='{decision.reason}'\n"
                )

            smoothed = _smoothed_bbox(history)
            self._tracks[best_id] = _TrackedPerson(
                bbox=bbox,
                bbox_history=history,
                blur=sticky_blur,
                carry_remaining=self._carry_forward_frames if sticky_blur else 0,
                gender=person.gender,
                face_det_score=person.face_det_score,
                gender_votes=votes,
                blur_history=blur_history,
                last_classify_frame=last_classify,
                last_smoothed=smoothed,
                last_seg_mask=person.seg_mask,
            )

            if sticky_blur:
                blur_targets.append((smoothed, person.seg_mask, 1.0))

        for tid in list(self._tracks.keys()):
            if tid in matched:
                continue
            tp = self._tracks[tid]
            # A subject whose last box is flush against a frame border has
            # walked off-screen, not briefly dropped out — stop blurring
            # on this frame instead of trailing their silhouette inward.
            exited = _touches_edge(tp.last_smoothed, frame_shape)
            if tp.blur and tp.carry_remaining > 0 and not exited:
                # Fade the carried silhouette out over the window so a
                # legitimate brief dropout decays smoothly rather than
                # holding a crisp ghost at the last position.
                alpha = tp.carry_remaining / self._carry_forward_frames
                blur_targets.append(
                    (tp.last_smoothed, tp.last_seg_mask, alpha)
                )
                tp.carry_remaining -= 1
                if tp.carry_remaining <= 0:
                    del self._tracks[tid]
            else:
                del self._tracks[tid]

        return blur_targets


# Default perception batch size: how many consecutive frames to bundle
# into a single YOLO+SegFormer forward pass. The previous frame_stride
# knob skipped frames to cut perception cost N×; batching gets a similar
# throughput win without the bbox-lag artifact, because every frame is
# still perceived — just amortized across a single GPU dispatch. CUDA
# defaults to 4 (empirically the elbow on A100 for 1280×720 footage with
# ≤6 people/frame at ~24 crops/batch); MPS/CPU defaults to 1 because
# Apple Silicon and CPU don't benefit nearly as much from cross-frame
# batching and a higher value just inflates latency.
def _default_perception_batch() -> int:
    return (
        DEFAULT_PERCEPTION_BATCH_CUDA
        if torch.cuda.is_available()
        else DEFAULT_PERCEPTION_BATCH_OTHER
    )


class VideoProcessor:
    def __init__(
        self,
        blur_kernel: int = DEFAULT_BLUR_KERNEL,
        *,
        perception_batch: int | None = None,
        perceiver: Perceiver | None = None,
        device: str | None = None,
        detector_model: str | None = None,
        require_cuda_io: bool = False,
    ) -> None:
        if blur_kernel <= 0 or blur_kernel % 2 == 0:
            raise ValueError(
                f"blur_kernel must be a positive odd integer, got {blur_kernel}"
            )
        batch = perception_batch if perception_batch is not None else _default_perception_batch()
        if batch < 1:
            raise ValueError(f"perception_batch must be >= 1, got {batch}")
        self.blur_kernel = blur_kernel
        # Run perception over a sliding window of `perception_batch`
        # consecutive frames in one GPU dispatch. The tracker, blur, and
        # writer remain sequential (the tracker is stateful and the
        # writer expects source order); only the perception forward
        # pass batches across frames. See module-level comment on the
        # default values.
        self.perception_batch = batch
        self.device_request = device
        self.require_cuda_io = require_cuda_io
        self.perception: Perceiver = (
            perceiver
            if perceiver is not None
            else Perception(device=device, detector_model=detector_model)
        )
        self.rules = RuleEngine()
        self.tracker = Tracker(self.perception, self.rules)

    def process(
        self,
        input_path: Path,
        output_path: Path,
        *,
        max_frames: int | None = None,
        skip_mux: bool = False,
    ) -> None:
        # CUDA fast path: NVDEC decode + NVENC encode via torchcodec +
        # ffmpeg. Falls back to cv2 (libavcodec on CPU) on Mac/CPU unless
        # the caller explicitly requires CUDA I/O.
        if self.require_cuda_io:
            require_cuda_io_available()
        use_cuda_io = cuda_io_available()
        if (
            not use_cuda_io
            and self.device_request == "cuda"
            and not self.require_cuda_io
        ):
            reasons = "; ".join(cuda_io_unavailable_reasons()) or "unknown reason"
            print(
                "[occlude] WARNING: CUDA compute is required, but CUDA video "
                f"I/O is unavailable ({reasons}); using OpenCV CPU video I/O. "
                "Pass --require-cuda-io to fail instead.",
                file=sys.stderr,
                flush=True,
            )
        cap: CudaVideoReader | cv2.VideoCapture
        if use_cuda_io:
            cap = CudaVideoReader(input_path)
            fps = cap.fps
            width = cap.width
            height = cap.height
            total_frames = cap.total_frames
        else:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"invalid frame dimensions: {width}x{height}")

        temp_dir = Path(tempfile.mkdtemp(prefix="occlude_"))
        # When skipping mux (benchmark mode) write the silent video
        # straight to the requested output path; otherwise stage it in
        # the temp dir so _mux_audio can produce the final output.
        silent_path = output_path if skip_mux else temp_dir / "processed_silent.mp4"

        # Writer setup: nvenc context manager on the CUDA path, cv2
        # VideoWriter on the fallback. We bind both behind a single
        # `write_frame(frame_bgr)` closure + a `close_writer()` callable
        # so the rest of the function doesn't branch.
        nvenc_cm = None
        writer: cv2.VideoWriter | None = None
        if use_cuda_io:
            nvenc_cm = cuda_video_writer(silent_path, width, height, fps)
            nvenc_write = nvenc_cm.__enter__()

            def write_frame(frame_bgr: np.ndarray) -> None:
                nvenc_write(frame_bgr)

            def close_writer() -> None:
                nvenc_cm.__exit__(None, None, None)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(silent_path), fourcc, fps, (width, height)
            )
            if not writer.isOpened():
                cap.release()
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise RuntimeError(f"could not open VideoWriter at {silent_path}")

            def write_frame(frame_bgr: np.ndarray) -> None:
                writer.write(frame_bgr)

            def close_writer() -> None:
                writer.release()

        pbar = tqdm(
            total=total_frames if total_frames > 0 else None,
            desc="Processing frames",
            unit="frame",
        )
        if TRACK_LOG_PATH:
            self.tracker._log_fh = open(TRACK_LOG_PATH, "a")  # noqa: SIM115
        try:
            frame_idx = 0
            # Test-injected Perceivers (FakePerceiver, ScriptedPerceiver)
            # implement the single-image detect_and_segment but may not
            # implement the batched variant. Keep the protocol surface
            # minimal — fall back to a loop when the optional method
            # isn't there.
            has_batched = hasattr(self.perception, "detect_and_segment_batch")
            B = self.perception_batch

            # Shared closure: process one batch in source order.
            # Mutates each frame_bgr in place with the blur composite.
            # Used both by the sync loop and the async runner so the
            # blur math has exactly one definition.
            def _process_batch(batch_items: list[tuple[int, "np.ndarray"]]) -> None:
                pil_frames = [
                    Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    for _, fr in batch_items
                ]
                if has_batched:
                    people_per_frame = self.perception.detect_and_segment_batch(
                        pil_frames
                    )
                else:
                    people_per_frame = [
                        self.perception.detect_and_segment(p) for p in pil_frames
                    ]
                for (fidx, frame_bgr), people in zip(batch_items, people_per_frame):
                    frame_shape = frame_bgr.shape[:2]
                    blur_targets = self.tracker.update(fidx, people, frame_shape)
                    blur_masks = [
                        (
                            bbox,
                            _scale_blur_mask(
                                prepare_blur_mask(bbox, seg_mask, frame_shape),
                                alpha,
                            ),
                        )
                        for bbox, seg_mask, alpha in blur_targets
                    ]
                    for bbox, mask in blur_masks:
                        apply_blur_mask(frame_bgr, bbox, mask, self.blur_kernel)

            if use_cuda_io:
                # Step 4 fast path: decode and encode each run on their
                # own thread with bounded queues, so NVDEC keeps the
                # decode queue full ahead of SegFormer and NVENC drains
                # the encode queue behind it. Source order is preserved
                # by the FIFO queues; no reorder buffer needed.
                def _progress() -> None:
                    pbar.update(1)

                run_async_pipeline(
                    decode_next=cap.read,
                    encode_frame=write_frame,
                    process_batch=_process_batch,
                    batch_size=B,
                    max_frames=max_frames,
                    progress_update=_progress,
                )
                frame_idx = pbar.n  # for the memory log below
            else:
                # Sync loop — the reference path for Mac/CPU and for any
                # CUDA box where the async runner is disabled. Same
                # batching as the async runner so output is bit-equal at
                # B=1; identical math at larger B (the runner only
                # changes when work happens, not what work is done).
                while True:
                    batch_items: list[tuple[int, "np.ndarray"]] = []
                    while len(batch_items) < B:
                        if (
                            max_frames is not None
                            and frame_idx + len(batch_items) >= max_frames
                        ):
                            break
                        ret, frame_bgr = cap.read()
                        if not ret:
                            break
                        batch_items.append((frame_idx + len(batch_items), frame_bgr))
                    if not batch_items:
                        break
                    _process_batch(batch_items)
                    for _idx, frame_bgr in batch_items:
                        write_frame(frame_bgr)
                        pbar.update(1)
                    frame_idx += len(batch_items)

                    if frame_idx % MEMORY_CLEANUP_EVERY == 0:
                        # gc.collect() + torch.mps.empty_cache() are redundant
                        # here — Perception._segment_batch already runs them after
                        # the forward pass. Keep only the macOS heap-pressure call.
                        if _pressure_relief is not None:
                            _pressure_relief(0, 0)
                    if MEM_LOG_PATH and psutil is not None and (frame_idx % 25 == 0):
                        rss_mb = psutil.Process(_os.getpid()).memory_info().rss / (1024 * 1024)
                        with open(MEM_LOG_PATH, "a") as fh:
                            fh.write(
                                f"frame={frame_idx} rss_mb={rss_mb:.0f} "
                                f"tracks={len(self.tracker._tracks)}\n"
                            )

        finally:
            if self.tracker._log_fh is not None:
                self.tracker._log_fh.close()
                self.tracker._log_fh = None
            pbar.close()
            cap.release()
            close_writer()

        try:
            if not skip_mux:
                self._mux_audio(input_path, silent_path, output_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _mux_audio(self, original: Path, silent_video: Path, output: Path) -> None:
        # `1:a:0?` makes the audio stream optional — ffmpeg won't error
        # if the original has no audio, it just produces a video-only
        # output. Sidesteps the no-audio fallback dance.
        cmd = [
            "ffmpeg", "-y",
            "-i", str(silent_video),
            "-i", str(original),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            str(output),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return

        tail = "\n".join(result.stderr.strip().splitlines()[-20:])
        print(
            f"warning: ffmpeg mux failed (exit {result.returncode}); "
            "writing video without audio"
        )
        if tail:
            print(tail)
        # shutil.move handles cross-filesystem moves; Path.rename does
        # not (the temp dir is on /var/folders, output on the user's
        # disk).
        shutil.move(str(silent_video), str(output))
