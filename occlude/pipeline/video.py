"""OCCLUDE v2 — offline, three-pass video pipeline.

v1 was a stream: it detected, segmented, classified, decided, and blurred
each frame independently, then bolted an IoU tracker on top to fight the
resulting flicker. But OCCLUDE has the whole file on disk — there is no
reason to be causal. v2 processes the video in three passes, each free to
use global context:

  Pass 1 — detect + track:   a high-recall detector finds people every
        frame; IoU association links them into Tracklets (one identity per
        shot). Only boxes are kept, so memory stays bounded.
  Pass 2 — judge:            each Tracklet is judged once, from its clearest
        frames, by a VLM. One verdict per person — decided globally, not
        re-litigated 24× a second.
  Pass 3 — render:           the verdict is applied across the Tracklet's
        entire span (so a person flagged late is blurred from their first
        frame — the "blurs late" bug, gone by construction). SAM2 produces a
        clean silhouette only for the people actually being blurred; ffmpeg
        muxes the original audio back.

This dissolves almost all of v1's machinery — gender-vote windows, blur-vote
smoothing, carry-forward, bbox moving averages — which existed only to
approximate global knowledge from a one-frame-at-a-time view.

The heavy models (detector, SAM2, VLM) are built lazily and can be injected,
so the pass logic is exercised by tests with light fakes on CPU.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from occlude.pipeline.blur import blur_region
from occlude.pipeline.config import DEFAULT_BLUR_KERNEL
from occlude.pipeline.decide import aggregate
from occlude.pipeline.detect import (
    DEFAULT_CONF,
    DEFAULT_DETECTOR_WEIGHTS,
    PersonDetector,
)
from occlude.pipeline.judge import DEFAULT_JUDGE_MODEL, VLMJudge
from occlude.pipeline.progress import pbar as make_pbar
from occlude.pipeline.scenes import ShotSegmenter
from occlude.pipeline.track import SAM2Segmenter, TrackletBuilder
from occlude.pipeline.tracklets import BBox, Tracklet

__all__ = ["VideoProcessor", "blur_region"]

# How many of each tracklet's clearest frames the VLM judges. >1 so one bad
# crop (motion blur, a turned head) can't decide a whole person; decide.py
# aggregates them with the over-blur tie-break.
DEFAULT_JUDGE_FRAMES = 3
# Person crops sent to the VLM in one forward pass.
DEFAULT_JUDGE_BATCH = 8
# Drop tracklets shorter than this (single-frame detector blips). A real
# person persists for more than one frame; a 1-frame ghost is noise we don't
# want to judge or blur.
DEFAULT_MIN_TRACKLET_LEN = 2
# Pad the person bbox by this fraction before cropping for the VLM, so it
# sees a little context (is the head covered? what's below the navel?).
JUDGE_CROP_PAD = 0.12
# Downscale a judge crop's long side to this many pixels. The VLM doesn't
# need full resolution to read sex/modesty and it keeps the batch light.
JUDGE_CROP_MAX = 512


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


class VideoProcessor:
    def __init__(
        self,
        blur_kernel: int = DEFAULT_BLUR_KERNEL,
        *,
        device: str | None = None,
        detector_weights: str | None = None,
        detector_conf: float | None = None,
        judge_model: str | None = None,
        judge_frames: int = DEFAULT_JUDGE_FRAMES,
        judge_batch: int = DEFAULT_JUDGE_BATCH,
        min_tracklet_len: int = DEFAULT_MIN_TRACKLET_LEN,
        use_sam2: bool = True,
        # Injection points (tests / advanced use): pre-built models.
        detector: PersonDetector | None = None,
        segmenter: SAM2Segmenter | None = None,
        judge: VLMJudge | None = None,
    ) -> None:
        if blur_kernel <= 0 or blur_kernel % 2 == 0:
            raise ValueError(
                f"blur_kernel must be a positive odd integer, got {blur_kernel}"
            )
        self.blur_kernel = blur_kernel
        self.device = device
        self.judge_frames = judge_frames
        self.judge_batch = judge_batch
        self.min_tracklet_len = min_tracklet_len
        self._use_sam2 = use_sam2

        self._detector_weights = detector_weights or DEFAULT_DETECTOR_WEIGHTS
        self._detector_conf = detector_conf if detector_conf is not None else DEFAULT_CONF
        self._judge_model = judge_model or DEFAULT_JUDGE_MODEL

        self._detector = detector
        self._segmenter = segmenter
        self._judge = judge

    # --- lazy model accessors (so --help / tests don't load weights) -----

    @property
    def detector(self) -> PersonDetector:
        if self._detector is None:
            self._detector = PersonDetector(
                self._detector_weights, device=self.device, conf=self._detector_conf
            )
        return self._detector

    @property
    def judge(self) -> VLMJudge:
        if self._judge is None:
            self._judge = VLMJudge(self._judge_model, device=self.device)
        return self._judge

    @property
    def segmenter(self) -> SAM2Segmenter | None:
        if not self._use_sam2:
            return None
        if self._segmenter is None:
            try:
                self._segmenter = SAM2Segmenter(device=self.device)
            except ImportError:
                # SAM2 isn't on PyPI, so a plain `pip install occlude` lands
                # here: degrade to feathered-box blur instead of failing the
                # render after the expensive detect and judge passes.
                print(
                    "sam2 is not installed - using feathered-box blur instead "
                    "of silhouettes. For silhouette blur: pip install "
                    '"git+https://github.com/facebookresearch/sam2.git"',
                    file=sys.stderr,
                )
                self._use_sam2 = False
                return None
        return self._segmenter

    # --- public entry point ---------------------------------------------

    def process(
        self,
        input_path: Path,
        output_path: Path,
        *,
        max_frames: int | None = None,
        skip_mux: bool = False,
    ) -> None:
        fps, width, height, total = self._probe(input_path)
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid frame dimensions: {width}x{height}")

        tracklets = self._pass1_track(input_path, total, max_frames)
        self._pass2_judge(input_path, tracklets, max_frames)
        _log_verdicts(tracklets)
        self._pass3_render(
            input_path, output_path, tracklets, fps, width, height, total,
            max_frames=max_frames, skip_mux=skip_mux,
        )

    # --- Pass 1: detect + associate -------------------------------------

    def _pass1_track(
        self, input_path: Path, total: int, max_frames: int | None
    ) -> list[Tracklet]:
        builder = TrackletBuilder()
        shots = ShotSegmenter()
        pbar = make_pbar(
            total=_pbar_total(total, max_frames),
            desc="Pass 1/3 detect+track", unit="frame",
        )
        try:
            for idx, frame_bgr in self._decode(input_path, max_frames):
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                is_cut = shots.push(gray)
                dets = self.detector.detect(frame_bgr, idx)
                builder.add_frame(idx, dets, is_cut=is_cut)
                pbar.update(1)
        finally:
            pbar.close()
        return builder.finalize(min_length=self.min_tracklet_len)

    # --- Pass 2: judge each tracklet ------------------------------------

    def _pass2_judge(
        self, input_path: Path, tracklets: list[Tracklet], max_frames: int | None
    ) -> None:
        if not tracklets:
            return
        # Map every frame we need a crop from -> the (tracklet, bbox) wanting it.
        needed: dict[int, list[tuple[Tracklet, BBox]]] = defaultdict(list)
        for t in tracklets:
            for f in t.best_frames(self.judge_frames):
                needed[f].append((t, t.detections[f].bbox))

        # One sequential decode collects exactly those crops (no random seeks).
        crops_by_tid: dict[int, list[Image.Image]] = defaultdict(list)
        pbar = make_pbar(total=len(needed), desc="Pass 2/3 collect", unit="frame")
        try:
            for idx, frame_bgr in self._decode(input_path, max_frames):
                wants = needed.get(idx)
                if not wants:
                    continue
                for t, bbox in wants:
                    crops_by_tid[t.track_id].append(self._crop(frame_bgr, bbox))
                pbar.update(1)
        finally:
            pbar.close()

        # Judge all crops batched across tracklets, then aggregate per track.
        items = [
            (tid, crop)
            for tid, crops in crops_by_tid.items()
            for crop in crops
        ]
        samples_by_tid: dict[int, list] = defaultdict(list)
        pbar = make_pbar(total=len(items), desc="Pass 2/3 judge", unit="crop")
        try:
            for batch in _chunks(items, self.judge_batch):
                verdicts = self.judge.judge_crops([c for _, c in batch])
                for (tid, _), v in zip(batch, verdicts):
                    samples_by_tid[tid].append(v)
                pbar.update(len(batch))
        finally:
            pbar.close()

        for t in tracklets:
            t.verdict = aggregate(samples_by_tid.get(t.track_id, []))

    # --- Pass 3: render --------------------------------------------------

    def _pass3_render(
        self,
        input_path: Path,
        output_path: Path,
        tracklets: list[Tracklet],
        fps: float,
        width: int,
        height: int,
        total: int,
        *,
        max_frames: int | None,
        skip_mux: bool,
    ) -> None:
        # Per-frame blur targets: only tracklets whose verdict says blur.
        blur_boxes: dict[int, list[BBox]] = defaultdict(list)
        for t in tracklets:
            if t.verdict is not None and t.verdict.blur:
                for f, det in t.detections.items():
                    blur_boxes[f].append(det.bbox)

        temp_dir = Path(tempfile.mkdtemp(prefix="occlude_"))
        silent_path = output_path if skip_mux else temp_dir / "processed_silent.mp4"
        writer = cv2.VideoWriter(
            str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"could not open VideoWriter at {silent_path}")

        seg = self.segmenter
        pbar = make_pbar(
            total=_pbar_total(total, max_frames),
            desc="Pass 3/3 render", unit="frame",
        )
        try:
            for idx, frame_bgr in self._decode(input_path, max_frames):
                boxes = blur_boxes.get(idx)
                if boxes:
                    masks = self._segment(seg, frame_bgr, boxes)
                    for bbox, full_mask in zip(boxes, masks):
                        seg_crop = _mask_crop(full_mask, bbox) if full_mask is not None else None
                        blur_region(frame_bgr, bbox, seg_crop, self.blur_kernel)
                writer.write(frame_bgr)
                pbar.update(1)
        finally:
            pbar.close()
            writer.release()

        try:
            if not skip_mux:
                self._mux_audio(input_path, silent_path, output_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _segment(
        seg: SAM2Segmenter | None, frame_bgr: np.ndarray, boxes: list[BBox]
    ) -> list[np.ndarray | None]:
        if seg is None:
            return [None] * len(boxes)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return list(seg.masks_for(rgb, boxes))

    # --- helpers ---------------------------------------------------------

    def _crop(self, frame_bgr: np.ndarray, bbox: BBox) -> Image.Image:
        """Padded, downscaled RGB PIL crop of one person for the VLM."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * JUDGE_CROP_PAD)
        pad_y = int((y2 - y1) * JUDGE_CROP_PAD)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame_bgr
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        long_side = max(img.size)
        if long_side > JUDGE_CROP_MAX:
            scale = JUDGE_CROP_MAX / long_side
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        return img

    @staticmethod
    def _decode(path: Path, max_frames: int | None):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        idx = 0
        try:
            while True:
                if max_frames is not None and idx >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                yield idx, frame
                idx += 1
        finally:
            cap.release()

    @staticmethod
    def _probe(path: Path) -> tuple[float, int, int, int]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        return fps, width, height, total

    def _mux_audio(self, original: Path, silent_video: Path, output: Path) -> None:
        # `1:a:0?` makes the audio stream optional — ffmpeg won't error if the
        # original has no audio, it just produces a video-only output.
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
            "writing video without audio",
            file=sys.stderr,
        )
        if tail:
            print(tail, file=sys.stderr)
        shutil.move(str(silent_video), str(output))


# Cap the per-person verdict lines so a crowd scene can't flood the log.
_VERDICT_LOG_MAX = 40


def _log_verdicts(tracklets: list[Tracklet]) -> None:
    """One auditable line per person: was it blurred, and why.

    Written to stderr so it lands in host-application logs (the OpenShot
    integration shows these lines live). This is the answer to "why did /
    didn't it blur X" without re-running the pipeline.
    """
    blurred = sum(
        1 for t in tracklets if t.verdict is not None and t.verdict.blur
    )
    print(
        f"verdicts: {len(tracklets)} people tracked, {blurred} to blur, "
        f"{len(tracklets) - blurred} left clear",
        file=sys.stderr,
    )
    for t in tracklets[:_VERDICT_LOG_MAX]:
        v = t.verdict
        if v is None:
            continue
        span = f"frames {t.first_frame}-{t.last_frame}"
        who = f"{v.sex or 'unknown-sex'}/{v.age_bracket}"
        print(
            f"  person {t.track_id} ({span}, {who}): "
            f"{'BLUR' if v.blur else 'clear'} - {v.reason}",
            file=sys.stderr,
        )
    if len(tracklets) > _VERDICT_LOG_MAX:
        print(
            f"  ... and {len(tracklets) - _VERDICT_LOG_MAX} more people",
            file=sys.stderr,
        )


def _mask_crop(full_mask: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop a full-frame boolean mask to the bbox region for blur shaping.

    blur.prepare_blur_mask wants the silhouette in bbox-crop coordinates; it
    resizes to the exact bbox size, so a clamped crop is all it needs.
    """
    h, w = full_mask.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return full_mask[y1:y2, x1:x2]


def _pbar_total(total: int, max_frames: int | None) -> int | None:
    if total <= 0:
        return max_frames
    if max_frames is not None:
        return min(total, max_frames)
    return total
