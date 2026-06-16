"""Stitch test_images/*.jpg into a 7-frame video for Stage 6 e2e tests.

Each image becomes one frame at 1 fps. Images are letterboxed (padded
with black) to a common canvas — large enough to fit any input — so
no image is stretched. Stretching distorts segmenter output enough to
flip rule decisions (the dark abaya in image 06 stops emitting
Hat+Scarf labels after a 1.2× vertical stretch, for example), which
defeats the point of using these images as a known-good test set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    images_dir = project_root / "test_images"
    output_path = project_root / "test_video.mp4"

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        print(f"error: no .jpg images in {images_dir}", file=sys.stderr)
        return 1

    # First pass: read all images and compute a canvas large enough to
    # fit any of them.
    images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"warning: could not read {path}, skipping", file=sys.stderr)
            continue
        images.append((path, img))

    if not images:
        print("error: no readable images", file=sys.stderr)
        return 1

    canvas_h = max(img.shape[0] for _, img in images)
    canvas_w = max(img.shape[1] for _, img in images)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 1.0, (canvas_w, canvas_h))
    if not writer.isOpened():
        print(f"error: could not open VideoWriter at {output_path}", file=sys.stderr)
        return 1

    written = 0
    try:
        for path, img in images:
            h, w = img.shape[:2]
            if (h, w) == (canvas_h, canvas_w):
                frame = img
            else:
                # Letterbox: paste image centered on a black canvas.
                top = (canvas_h - h) // 2
                left = (canvas_w - w) // 2
                frame = np.zeros((canvas_h, canvas_w, 3), dtype=img.dtype)
                frame[top:top + h, left:left + w] = img
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    print(f"wrote {written} frames at 1 fps -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
