"""Run perception + rules on a frame with verbose debug output.

For each detected person:
- bbox + det conf
- gender + face_det_score (raw InsightFace)
- seg_mask coverage % per label (in bbox AND in head region)
- rule decision + reason

Used to diagnose why specific persons in laughing_people.mp4 weren't
blurred (Stage 6 Finding 8 / 9 territory).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from occlude.pipeline.perception import SEG_LABELS, TARGET_LABELS, Perception  # noqa: E402
from occlude.pipeline.rules import (  # noqa: E402
    _LABEL_IDX,
    HEAD_SIDE_FACTOR,
    HEAD_UP_FACTOR,
    RuleEngine,
    _face_bounds,
)


def coverage_in(seg_mask: np.ndarray) -> dict[str, float]:
    total = max(seg_mask.size, 1)
    out = {}
    for i, name in enumerate(SEG_LABELS):
        cnt = int((seg_mask == i).sum())
        if cnt > 0:
            out[name] = 100.0 * cnt / total
    return out


def head_region(seg_mask: np.ndarray) -> np.ndarray | None:
    bounds = _face_bounds(seg_mask)
    if bounds is None:
        return None
    face_top, face_bot, face_left, face_right = bounds
    face_h = face_bot - face_top + 1
    face_w = face_right - face_left + 1
    H, W = seg_mask.shape
    h_top = max(0, face_top - int(HEAD_UP_FACTOR * face_h))
    h_bot = face_bot
    h_left = max(0, face_left - int(HEAD_SIDE_FACTOR * face_w))
    h_right = min(W, face_right + int(HEAD_SIDE_FACTOR * face_w))
    return seg_mask[h_top:h_bot, h_left:h_right]


def hair_below_face(seg_mask: np.ndarray) -> tuple[int, float]:
    bounds = _face_bounds(seg_mask)
    if bounds is None:
        return 0, 0.0
    _, face_bot, _, _ = bounds
    hair = seg_mask == _LABEL_IDX["Hair"]
    y = np.arange(seg_mask.shape[0])[:, None]
    cnt = int((hair & (y > face_bot)).sum())
    pct = 100.0 * cnt / seg_mask.size
    return cnt, pct


def main(image_path: str) -> int:
    p = Perception()
    rules = RuleEngine()
    img = Image.open(image_path).convert("RGB")
    print(f"image: {image_path} size={img.size}")
    people = p(img)
    print(f"detected: {len(people)}\n")
    for i, person in enumerate(people):
        x1, y1, x2, y2 = person.bbox
        print(f"=== person #{i} bbox=({x1},{y1})-({x2},{y2}) "
              f"size={x2-x1}x{y2-y1} det_conf={person.det_conf:.2f} ===")
        print(f"  gender={person.gender}  face_det_score={person.face_det_score:.2f}")

        cov_bbox = coverage_in(person.seg_mask)
        print("  coverage in bbox:")
        for n, pct in sorted(cov_bbox.items(), key=lambda x: -x[1]):
            mark = "*" if n in TARGET_LABELS else " "
            print(f"    {mark} {n:<15} {pct:6.2f}%")

        head = head_region(person.seg_mask)
        if head is not None:
            cov_head = coverage_in(head)
            print(f"  coverage in head region (size={head.shape[1]}x{head.shape[0]}):")
            for n, pct in sorted(cov_head.items(), key=lambda x: -x[1]):
                mark = "*" if n in TARGET_LABELS else " "
                print(f"    {mark} {n:<15} {pct:6.2f}%")
        else:
            print("  no Face label → no head region")

        below_count, below_pct = hair_below_face(person.seg_mask)
        print(f"  hair below face: {below_count} px ({below_pct:.2f}% of bbox)")

        decision = rules.decide(person)
        verdict = "BLUR" if decision.blur else "pass"
        print(f"  → {verdict}: {decision.reason} "
              f"(gender_used={decision.gender_used} overridden={decision.overridden})")
        print()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: debug_frame.py <image>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
