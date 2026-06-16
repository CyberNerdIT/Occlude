"""YOLO -> per-person crop -> SegFormer. Print per-bbox label coverage.

Stage-2 step 2. The Stage-1 numbers were per-image and therefore noisy
when the subject was small in frame (image 05's 0.20% legs). Here we run
YOLO first, crop each detected person, run the segmenter on that crop,
and report coverage relative to the bbox. This is the data the rule
layer will consume.
"""
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO

YOLO_MODEL_ID = "yolov8n.pt"
SEG_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.40

LABELS = [
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes",
    "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe",
    "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm",
    "Bag", "Scarf",
]
TARGET_LABELS = {
    "Hair", "Hat", "Scarf", "Upper-clothes", "Pants", "Skirt", "Dress",
    "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm",
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def segment_crop(crop: Image.Image, processor, model, device) -> np.ndarray:
    inputs = processor(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled = torch.nn.functional.interpolate(
        logits, size=crop.size[::-1], mode="bilinear", align_corners=False,
    )
    return upsampled.argmax(dim=1)[0].numpy()


def coverage(pred: np.ndarray) -> dict[str, float]:
    total = pred.size
    return {
        LABELS[i]: 100.0 * int((pred == i).sum()) / total
        for i in range(len(LABELS))
        if int((pred == i).sum()) > 0
    }


def visualize(image: Image.Image, crops: list, preds: list, boxes: list,
              output_path: Path, title: str) -> None:
    cmap = plt.get_cmap("tab20", len(LABELS))
    n = len(crops)
    fig, axes = plt.subplots(1, 1 + 2 * n, figsize=(6 + 6 * n, 7))
    if n == 0:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]

    axes[0].imshow(image)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2,
        )
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 8, f"#{i}", color="red", fontsize=10, weight="bold")
    axes[0].set_title("input + bboxes")
    axes[0].axis("off")

    present_all = set()
    for i, (crop, pred) in enumerate(zip(crops, preds)):
        ax_c = axes[1 + 2 * i]
        ax_s = axes[1 + 2 * i + 1]
        ax_c.imshow(crop)
        ax_c.set_title(f"crop #{i}")
        ax_c.axis("off")
        ax_s.imshow(pred, cmap=cmap, vmin=0, vmax=len(LABELS) - 1, interpolation="nearest")
        ax_s.set_title(f"seg #{i}")
        ax_s.axis("off")
        present_all.update(np.unique(pred).tolist())

    if present_all:
        handles = [mpatches.Patch(color=cmap(i), label=LABELS[i]) for i in sorted(present_all)]
        axes[-1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    image_dir = project_root / "test_images"
    out_dir = project_root / "test_output"
    out_dir.mkdir(exist_ok=True)

    device = get_device()
    print(f"device: {device}")
    print(f"loading {YOLO_MODEL_ID}...")
    detector = YOLO(YOLO_MODEL_ID)
    print(f"loading {SEG_MODEL_ID}...")
    processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)
    seg_model = AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL_ID).to(device).eval()

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        print(f"no images found in {image_dir}", file=sys.stderr)
        return 1

    for img_path in images:
        print(f"\n{img_path.name}")
        full_image = Image.open(img_path).convert("RGB")
        det = detector.predict(
            source=str(img_path), classes=[PERSON_CLASS_ID],
            conf=CONF_THRESHOLD, verbose=False,
        )[0]

        boxes = det.boxes
        if boxes is None or len(boxes) == 0:
            print("  no person detected above threshold")
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        crops, preds, kept_boxes = [], [], []
        for i, ((x1, y1, x2, y2), c) in enumerate(zip(xyxy, confs)):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            crop = full_image.crop((x1, y1, x2, y2))
            pred = segment_crop(crop, processor, seg_model, device)
            cov = coverage(pred)
            print(f"  person {i}: bbox=({x1},{y1})-({x2},{y2}) "
                  f"size={x2 - x1}x{y2 - y1} conf={c:.2f}")
            for name, p in sorted(cov.items(), key=lambda x: -x[1]):
                mark = "*" if name in TARGET_LABELS else " "
                print(f"     {mark} {name:<15} {p:5.2f}%")
            crops.append(crop)
            preds.append(pred)
            kept_boxes.append((x1, y1, x2, y2))

        visualize(full_image, crops, preds, kept_boxes,
                  out_dir / f"{img_path.stem}_detseg.png", img_path.name)

    print(f"\nvisualizations saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
