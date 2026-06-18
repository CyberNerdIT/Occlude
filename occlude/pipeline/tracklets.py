"""v2 data model — the spine that flows through the three offline passes.

OCCLUDE v2 is not a stream. It has the whole file on disk, so it works in
three passes instead of frame-by-frame:

  Pass 1 (detect + track) populates Tracklets with per-frame Detections.
  Pass 2 (judge)          attaches one Verdict to each Tracklet.
  Pass 3 (render)         applies that Verdict across every frame the
                          Tracklet spans.

These dataclasses are the contract between those passes. They carry no
model code and no GPU dependency, so the policy that operates on them
(decide.py) stays auditable and unit-testable on a laptop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# (x1, y1, x2, y2) in source-frame pixels, top-left origin.
BBox = tuple[int, int, int, int]

# Sex / age vocabularies. Centralised here so judge.py (which parses VLM
# output) and decide.py (which applies policy) cannot drift apart on the
# string values.
SEX_MALE = "M"
SEX_FEMALE = "F"

AGE_CHILD = "child"
AGE_ADULT = "adult"
AGE_UNKNOWN = "unknown"


@dataclass
class Detection:
    """One person observed in one frame.

    ``mask`` is a boolean array in *full-frame* (H, W) coordinates once the
    tracking pass has run; it is ``None`` for a raw detector box before
    segmentation. Keeping the mask full-frame rather than crop-relative
    means the render pass composites it directly with no offset bookkeeping.
    """

    frame_idx: int
    bbox: BBox
    score: float
    mask: np.ndarray | None = None

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class Verdict:
    """The per-tracklet decision produced by Pass 2.

    ``blur`` is the single bit the render pass consumes. Everything else is
    the evidence behind it, retained so decide.py can apply policy (child
    exemption, over-blur on uncertainty) over raw model output without
    re-querying, and so failures are debuggable from a log line.
    """

    blur: bool
    is_human: bool = True
    sex: str | None = None            # SEX_MALE | SEX_FEMALE | None
    age_bracket: str = AGE_UNKNOWN     # AGE_CHILD | AGE_ADULT | AGE_UNKNOWN
    reason: str = ""
    confidence: float = 0.0


@dataclass
class Tracklet:
    """One person's continuous presence within a single shot.

    A person who exits and re-enters, or who is split by a shot cut, becomes
    multiple Tracklets, each judged independently. That is deliberate: a shot
    cut can replace who is on screen entirely, so carrying a mask or verdict
    across one would smear one person's silhouette onto another.
    """

    track_id: int
    detections: dict[int, Detection] = field(default_factory=dict)
    verdict: Verdict | None = None

    def add(self, det: Detection) -> None:
        self.detections[det.frame_idx] = det

    @property
    def frames(self) -> list[int]:
        return sorted(self.detections)

    @property
    def first_frame(self) -> int:
        return min(self.detections)

    @property
    def last_frame(self) -> int:
        return max(self.detections)

    def __len__(self) -> int:
        return len(self.detections)

    def best_frames(self, k: int = 1) -> list[int]:
        """Frame indices with the most visual evidence, best first.

        The VLM judges a Tracklet from a few representative frames, not all
        of them. "Best" = largest, most-confident detection: a big bbox means
        the person is close and high-resolution, which is exactly where sex /
        age / modesty cues are legible. This is the offline luxury — we pick
        the clearest view of each person instead of being stuck with whatever
        frame a streaming decision happened to land on.
        """
        ranked = sorted(
            self.detections.values(),
            key=lambda d: d.area * max(d.score, 0.0),
            reverse=True,
        )
        return [d.frame_idx for d in ranked[: max(k, 0)]]
