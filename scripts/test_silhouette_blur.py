"""Smoke-test the silhouette-shaped blur on a single frame.

Loads one immodest test image, runs perception, then writes a side-by-side
of (original | rectangle blur | silhouette blur) so the user can eyeball
the new look.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from occlude.pipeline.perception import Perception  # noqa: E402
from occlude.pipeline.video import VideoProcessor, blur_region  # noqa: E402

IMG = ROOT / "test_images" / "03_immodest_woman_dress.jpg"
OUT = ROOT / "test_output" / "_silhouette_blur_preview.png"


def rectangle_blur(frame: np.ndarray, bbox, kernel: int) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = bbox
    region = out[y1:y2, x1:x2]
    if region.size == 0:
        return out
    blurred = cv2.GaussianBlur(region, (kernel, kernel), 0)
    out[y1:y2, x1:x2] = blurred
    return out


def main() -> None:
    img = Image.open(IMG).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    perception = Perception()
    people = perception.detect_and_segment(img)
    if not people:
        print("no people detected")
        return

    proc = VideoProcessor()

    rect = bgr.copy()
    sil = bgr.copy()
    for p in people:
        rect = rectangle_blur(rect, p.bbox, proc.blur_kernel)
        blur_region(sil, p.bbox, p.seg_mask, proc.blur_kernel)

    panel = np.concatenate([bgr, rect, sil], axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), panel)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
