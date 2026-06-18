"""Person detector for Pass 1.

Replaces the YOLOv8n nano detector. The v1 failure was recall: fully visible
people at the frame edges were missed entirely, and a missed detection is a
missed blur — the worst error class for this tool. Offline we no longer
trade recall for speed, so the default is a heavy, high-recall model.

Ultralytics is the carrier because it is already a dependency and serves
both RT-DETR and the YOLO family under one API with automatic weight
download. RT-DETR is a DETR-family transformer: it reasons over global image
context, which is what recovers the edge and partial-body people that
anchor-based YOLO drops. Weights are swappable via the constructor (and the
``--detector`` flag); Co-DETR has higher recall still but needs an
MMDetection/mmcv install, so it is a documented future upgrade, not a v1
dependency.
"""
from __future__ import annotations

import numpy as np

from occlude.pipeline.tracklets import Detection

# COCO class id 0 is "person" across every Ultralytics detector.
PERSON_CLASS = 0

# Default weights: rtdetr-x is the highest-recall option that installs with
# the existing ultralytics dependency and downloads on first use.
DEFAULT_DETECTOR_WEIGHTS = "rtdetr-x.pt"

# Confidence floor. Lower than v1's 0.40 on purpose: offline, the SAM2 mask
# and the VLM judge downstream both tolerate extra candidate boxes, and the
# over-blur policy would rather over-detect an edge person than miss them.
DEFAULT_CONF = 0.25


class PersonDetector:
    """Detect people in frames, returning mask-less :class:`Detection`s.

    Segmentation masks are filled in later by the tracking pass; the detector
    only contributes boxes and scores.
    """

    def __init__(
        self,
        weights: str = DEFAULT_DETECTOR_WEIGHTS,
        *,
        device: str | None = None,
        conf: float = DEFAULT_CONF,
    ) -> None:
        from ultralytics import RTDETR, YOLO

        self.conf = conf
        self.device = device
        name = str(weights).lower()
        model_cls = RTDETR if ("rtdetr" in name or "rt-detr" in name) else YOLO
        self.model = model_cls(weights)

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> list[Detection]:
        """Detect people in one BGR frame (OpenCV/Ultralytics convention)."""
        return self.detect_batch([frame_bgr], [frame_idx])[0]

    def detect_batch(
        self, frames_bgr: list[np.ndarray], frame_indices: list[int]
    ) -> list[list[Detection]]:
        """Detect people across a list of BGR frames, one result list each."""
        results = self.model.predict(
            source=frames_bgr,
            classes=[PERSON_CLASS],
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        return [
            _results_to_detections(r, fidx)
            for r, fidx in zip(results, frame_indices)
        ]


def _results_to_detections(result, frame_idx: int) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    dets: list[Detection] = []
    for (x1, y1, x2, y2), s in zip(xyxy, scores):
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        # Sub-pixel-wide boxes at frame edges collapse to zero area after the
        # int() floor; they crash crop/segmentation downstream. Drop them
        # here — a missed sliver of a person is harmless, a crash aborts the
        # whole run.
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        dets.append(
            Detection(
                frame_idx=frame_idx,
                bbox=(ix1, iy1, ix2, iy2),
                score=float(s),
            )
        )
    return dets
