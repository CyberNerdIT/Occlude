"""One-shot probe: what does SegformerImageProcessor actually do, and
what is `outputs.logits.shape` on a real multi-person frame?

Answers two questions before we commit to a memory fix:
  1. Does the processor already resize the input internally?
  2. Is the per-frame leak driven by raw-logits size (forward pass) or
     by the upsample-to-crop CPU tensor?
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from occlude.pipeline.perception import Perception  # noqa: E402


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    video = project_root / "test_video_real" / "laughing_people.mp4"
    if not video.exists():
        print(f"missing: {video}")
        return 1

    print("loading perception...")
    p = Perception()
    proc = p.seg_processor
    print("\n=== SegformerImageProcessor config ===")
    print(f"do_resize       = {getattr(proc, 'do_resize', '?')}")
    print(f"size            = {getattr(proc, 'size', '?')}")
    print(f"do_rescale      = {getattr(proc, 'do_rescale', '?')}")
    print(f"do_normalize    = {getattr(proc, 'do_normalize', '?')}")

    cap = cv2.VideoCapture(str(video))
    # Skip in to where the doc says multi-person frames live.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        print("could not read frame 100")
        return 1
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)
    print(f"\nframe size = {pil.size}  (W, H)")

    det = p.detector.predict(source=pil, classes=[0], conf=0.40, verbose=False)[0]
    boxes = det.boxes
    if boxes is None or len(boxes) == 0:
        print("no detections at frame 100; bumping forward")
        return 1
    xyxy = boxes.xyxy.cpu().numpy()
    print(f"detections    = {len(xyxy)}")

    print("\n=== per-crop SegFormer pass ===")
    total_logit_bytes = 0
    total_upsample_bytes = 0
    for i, (x1f, y1f, x2f, y2f) in enumerate(xyxy):
        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
        crop = pil.crop((x1, y1, x2, y2))
        inputs = proc(images=crop, return_tensors="pt").to(p.device)
        pv_shape = tuple(inputs["pixel_values"].shape)
        with torch.no_grad():
            outputs = p.seg_model(**inputs)
        logits_shape = tuple(outputs.logits.shape)
        # Bytes of raw logits (float32):
        logit_bytes = int(np.prod(logits_shape)) * 4
        # Bytes of upsample-to-crop tensor (what we currently allocate on CPU):
        upsample_bytes = (
            logits_shape[0] * logits_shape[1] * crop.size[1] * crop.size[0] * 4
        )
        total_logit_bytes += logit_bytes
        total_upsample_bytes += upsample_bytes
        print(
            f"  person {i}: crop={crop.size}  "
            f"pixel_values={pv_shape}  "
            f"logits={logits_shape}  "
            f"raw={logit_bytes/1e6:.1f}MB  "
            f"upsampled@crop={upsample_bytes/1e6:.1f}MB"
        )
        del outputs, inputs

    print(f"\nTOTALS: raw_logits={total_logit_bytes/1e6:.1f}MB  "
          f"upsampled@crop={total_upsample_bytes/1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
