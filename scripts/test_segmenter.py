"""Run SegFormer-B2-clothes on each image in test_images/ and report per-label coverage.

Stage-1 sanity check before building the pipeline. We need to confirm the
segmentation model can distinguish the labels the modesty rules depend on
(hair, arms, legs, upper-clothes) on real photos.
"""
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

MODEL_ID = "mattmdjaga/segformer_b2_clothes"

LABELS = [
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes",
    "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe",
    "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm",
    "Bag", "Scarf",
]

# Labels the modesty rules act on. Marked with '*' in the report.
TARGET_LABELS = {
    "Hair", "Upper-clothes", "Pants", "Skirt", "Dress", "Face",
    "Left-leg", "Right-leg", "Left-arm", "Right-arm",
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def segment(image_path: Path, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False,
    )
    pred = upsampled.argmax(dim=1)[0].numpy()
    return image, pred


def coverage(pred: np.ndarray) -> dict[str, float]:
    total = pred.size
    return {
        LABELS[i]: 100.0 * int((pred == i).sum()) / total
        for i in range(len(LABELS))
        if int((pred == i).sum()) > 0
    }


def visualize(image: Image.Image, pred: np.ndarray, output_path: Path, title: str) -> None:
    cmap = plt.get_cmap("tab20", len(LABELS))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(image)
    axes[0].set_title("input")
    axes[0].axis("off")
    axes[1].imshow(pred, cmap=cmap, vmin=0, vmax=len(LABELS) - 1, interpolation="nearest")
    axes[1].set_title("segmentation")
    axes[1].axis("off")
    present = sorted(set(np.unique(pred).tolist()))
    handles = [mpatches.Patch(color=cmap(i), label=LABELS[i]) for i in present]
    axes[1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
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
    print(f"loading {MODEL_ID}...")
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID).to(device).eval()

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        print(f"no images found in {image_dir}", file=sys.stderr)
        return 1

    for img_path in images:
        print(f"\n{img_path.name}")
        image, pred = segment(img_path, processor, model, device)
        pct = coverage(pred)
        for name, p in sorted(pct.items(), key=lambda x: -x[1]):
            mark = "*" if name in TARGET_LABELS else " "
            print(f"  {mark} {name:<15} {p:5.2f}%")
        visualize(image, pred, out_dir / f"{img_path.stem}_seg.png", img_path.name)

    print(f"\nvisualizations saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
