"""Run YOLOv8n on each image in test_images/ and report person bounding boxes.

Stage-2 step 1. We want to confirm the detector finds one tight box per
intended subject before we hand crops to the segmenter. The image to watch
is 05 (man in shorts on a track) where the subject is small in frame —
that's the case where Stage 1 said per-bbox coverage would matter most.
"""
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_ID = "yolov8n.pt"
PERSON_CLASS_ID = 0  # COCO


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    image_dir = project_root / "test_images"
    out_dir = project_root / "test_output"
    out_dir.mkdir(exist_ok=True)

    print(f"loading {MODEL_ID}...")
    model = YOLO(MODEL_ID)

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        print(f"no images found in {image_dir}", file=sys.stderr)
        return 1

    for img_path in images:
        print(f"\n{img_path.name}")
        result = model.predict(source=str(img_path), classes=[PERSON_CLASS_ID], verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print("  no person detected")
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            img_w, img_h = result.orig_shape[1], result.orig_shape[0]
            for i, ((x1, y1, x2, y2), c) in enumerate(zip(xyxy, conf)):
                bw, bh = x2 - x1, y2 - y1
                frac = (bw * bh) / (img_w * img_h)
                print(
                    f"  person {i}: bbox=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) "
                    f"size={bw:.0f}x{bh:.0f} conf={c:.2f} bbox/image={frac*100:.1f}%"
                )

        annotated = result.plot()  # BGR with boxes drawn
        cv2.imwrite(str(out_dir / f"{img_path.stem}_det.png"), annotated)

    print(f"\nannotated images saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
