"""Pass 1 tracking + Pass 3 segmentation.

Two responsibilities, split so the logic half is testable without a GPU:

  TrackletBuilder  — pure IoU association: turn per-frame detections into
                     Tracklets (stable identities), split at shot cuts. No
                     models, fully unit-tested.
  SAM2Segmenter    — wraps SAM2's image predictor to turn a box into a clean
                     full-frame silhouette mask. GPU; verified on Colab.

Why IoU association rather than v1's IoU *tracker*: the v1 tracker made a
blur decision every frame and then smoothed it — that per-frame decision is
what flickered. Here IoU only links a person's boxes across frames into one
identity; the blur decision is made once per Tracklet (Pass 2) and applied to
its whole span (Pass 3). Identity association is unavoidable even with a
video-segmentation model, so this is the honest minimum, not a regression.

Silhouette masks are produced lazily in Pass 3 — only for tracklets that are
actually blurred — so the common case (most people are modest) never pays for
segmentation, and 100k+ masks are never held in memory at once.
"""
from __future__ import annotations

import numpy as np

from occlude.pipeline.tracklets import BBox, Detection, Tracklet

# IoU above which a detection is linked to an open track. 0.3 tolerates head
# turns and walking between frames without false-linking two nearby people.
DEFAULT_IOU_THRESHOLD = 0.3

# A track unseen for more than this many frames is closed. Bridging short
# detector dropouts keeps one person from being split into many tracklets
# (which would each be judged separately); too long risks linking a person
# to a different one who later appears at the same spot. ~0.5 s at 24 fps.
DEFAULT_MAX_GAP = 12


def iou(a: BBox, b: BBox) -> float:
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


class TrackletBuilder:
    """Greedy IoU association of per-frame detections into Tracklets."""

    def __init__(
        self,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_gap: int = DEFAULT_MAX_GAP,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_gap = max_gap
        self._tracklets: dict[int, Tracklet] = {}
        self._last_bbox: dict[int, BBox] = {}
        self._last_seen: dict[int, int] = {}
        self._open: set[int] = set()
        self._next_id = 0

    def add_frame(
        self, frame_idx: int, detections: list[Detection], is_cut: bool = False
    ) -> None:
        """Associate one frame's detections to open tracks.

        A shot cut closes every open track first, so a person present before
        and after the cut becomes two tracklets — never one mask propagated
        across a scene change.
        """
        if is_cut:
            self._open.clear()

        # Retire tracks whose gap has grown past the bridge window.
        for tid in list(self._open):
            if frame_idx - self._last_seen[tid] > self.max_gap:
                self._open.discard(tid)

        # Greedy IoU match: strongest overlaps first, each track used once.
        candidates = []
        for det in detections:
            for tid in self._open:
                ov = iou(det.bbox, self._last_bbox[tid])
                if ov >= self.iou_threshold:
                    candidates.append((ov, det, tid))
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_dets: set[int] = set()
        used_tracks: set[int] = set()
        for _ov, det, tid in candidates:
            if id(det) in matched_dets or tid in used_tracks:
                continue
            self._attach(tid, det, frame_idx)
            matched_dets.add(id(det))
            used_tracks.add(tid)

        # Unmatched detections open new tracks.
        for det in detections:
            if id(det) in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracklets[tid] = Tracklet(track_id=tid)
            self._open.add(tid)
            self._attach(tid, det, frame_idx)

    def _attach(self, tid: int, det: Detection, frame_idx: int) -> None:
        self._tracklets[tid].add(det)
        self._last_bbox[tid] = det.bbox
        self._last_seen[tid] = frame_idx
        self._open.add(tid)

    def finalize(self, min_length: int = 1) -> list[Tracklet]:
        """Return all tracklets with at least *min_length* detections."""
        return [
            t for t in self._tracklets.values() if len(t) >= min_length
        ]


class SAM2Segmenter:
    """Turn a person box into a clean full-frame silhouette mask via SAM2."""

    def __init__(
        self,
        hf_model: str = "facebook/sam2-hiera-large",
        *,
        device: str | None = None,
    ) -> None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.predictor = SAM2ImagePredictor.from_pretrained(hf_model, device=device)

    def masks_for(
        self, frame_rgb: np.ndarray, bboxes: list[BBox]
    ) -> list[np.ndarray]:
        """Boolean (H, W) full-frame mask for each box, in input order.

        One ``set_image`` per frame amortizes the image encoder across all of
        that frame's boxes — the expensive part of SAM2 image inference.
        """
        if not bboxes:
            return []
        import torch

        self.predictor.set_image(frame_rgb)
        box_arr = np.asarray(bboxes, dtype=np.float32)
        with torch.inference_mode():
            masks, _scores, _ = self.predictor.predict(
                box=box_arr, multimask_output=False
            )
        masks = np.asarray(masks)
        # SAM2 returns (N, 1, H, W) for N>1 boxes and (1, H, W) for one box;
        # normalize to a list of (H, W) boolean masks.
        if masks.ndim == 4:
            masks = masks[:, 0]
        elif masks.ndim == 3 and len(bboxes) == 1:
            masks = masks[None, 0] if masks.shape[0] != 1 else masks
        return [m.astype(bool) for m in masks]
