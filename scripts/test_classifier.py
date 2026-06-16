"""Run InsightFace buffalo_l on each image in test_images/ and report gender.

Stage-3 validation. The pipeline picks which modesty ruleset to apply
based on this prediction. The spec policy is: when the classifier is
uncertain (low face-detection confidence or no face), default to the
female ruleset, which is stricter.

This script answers two questions:
  1. Does InsightFace correctly detect and classify the 7 test subjects?
  2. What does the det_score look like — i.e. what's a reasonable
     threshold for "uncertain → default female"?
"""
import sys
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis

EXPECTED = {
    "01_modest_woman_hijab.jpg":      "F",
    "02_modest_man_suit.jpg":         "M",
    "03_immodest_woman_dress.jpg":    "F",
    "04_immodest_shirtless_man.jpg":  "M",
    "05_immodest_man_shorts.jpg":     "M",
    "06_modest_woman_abaya.jpg":      "F",
    "07_immodest_woman_longsleeve.jpg": "F",
}


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    image_dir = project_root / "test_images"
    out_dir = project_root / "test_output"
    out_dir.mkdir(exist_ok=True)

    print("loading buffalo_l...")
    # CPUExecutionProvider only — avoids CoreML/MPS quirks at validation stage.
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        print(f"no images found in {image_dir}", file=sys.stderr)
        return 1

    correct = 0
    total = 0
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        faces = app.get(bgr)
        expected = EXPECTED.get(img_path.name, "?")
        print(f"\n{img_path.name}  (expected {expected})")

        if not faces:
            print("  no face detected")
            total += 1
            continue

        # Sort by detection score, biggest first.
        faces.sort(key=lambda f: -f.det_score)
        for i, f in enumerate(faces):
            sex = "M" if int(f.gender) == 1 else "F"
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            mark = "✓" if i == 0 and sex == expected else (" " if i > 0 else "✗")
            print(f"  {mark} face {i}: sex={sex} age={f.age} "
                  f"det_score={f.det_score:.3f} "
                  f"bbox=({x1},{y1})-({x2},{y2}) size={x2-x1}x{y2-y1}")
            color = (0, 255, 0) if sex == "F" else (255, 128, 0)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                bgr, f"{sex} {f.det_score:.2f}", (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        # Compare highest-scoring face against expected.
        top_sex = "M" if int(faces[0].gender) == 1 else "F"
        total += 1
        if top_sex == expected:
            correct += 1

        cv2.imwrite(str(out_dir / f"{img_path.stem}_gender.png"), bgr)

    print(f"\noverall: {correct}/{total} top-face predictions match expected")
    print(f"annotated images saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
