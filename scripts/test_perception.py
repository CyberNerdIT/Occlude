"""Stage 4 — perception sanity check.

Runs the Stage 4 perception callable on every image in test_images/
and emits a per-person summary plus a visualization. Smoke test:
gender predictions should match `docs/04-gender-classifier.md`'s
EXPECTED map for the unoccluded faces (1–4, 6, 7). Image 05 (masked
face) is the known-failure case from Stage 3 — its prediction here
may differ from the doc because Stage 4 runs InsightFace on the
YOLO crop rather than the full frame, changing the input scale
to the genderage head.
"""
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from occlude.pipeline.perception import SEG_LABELS, TARGET_LABELS, Perception, Person  # noqa: E402


def coverage(seg_mask: np.ndarray) -> dict[str, float]:
    total = seg_mask.size
    return {
        SEG_LABELS[i]: 100.0 * int((seg_mask == i).sum()) / total
        for i in range(len(SEG_LABELS))
        if int((seg_mask == i).sum()) > 0
    }


def visualize(image: Image.Image, people: list[Person],
              output_path: Path, title: str) -> None:
    cmap = plt.get_cmap("tab20", len(SEG_LABELS))
    n = len(people)
    fig, axes = plt.subplots(1, 1 + 2 * n, figsize=(6 + 6 * n, 7))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes[0].imshow(image)
    for i, p in enumerate(people):
        x1, y1, x2, y2 = p.bbox
        axes[0].add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="red", linewidth=2,
        ))
        axes[0].text(x1, y1 - 8, f"#{i}", color="red",
                     fontsize=10, weight="bold")
    axes[0].set_title("input + bboxes")
    axes[0].axis("off")

    present: set[int] = set()
    for i, p in enumerate(people):
        ax_c = axes[1 + 2 * i]
        ax_s = axes[1 + 2 * i + 1]
        ax_c.imshow(p.crop)
        gender_str = p.gender or "?"
        ax_c.set_title(
            f"crop #{i} — {gender_str} (face={p.face_det_score:.2f})"
        )
        ax_c.axis("off")
        ax_s.imshow(p.seg_mask, cmap=cmap,
                    vmin=0, vmax=len(SEG_LABELS) - 1, interpolation="nearest")
        ax_s.set_title(f"seg #{i}")
        ax_s.axis("off")
        present.update(np.unique(p.seg_mask).tolist())

    if present:
        handles = [mpatches.Patch(color=cmap(i), label=SEG_LABELS[i])
                   for i in sorted(present)]
        axes[-1].legend(handles=handles, loc="center left",
                        bbox_to_anchor=(1.02, 0.5), fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    image_dir = project_root / "test_images"
    out_dir = project_root / "test_output"
    out_dir.mkdir(exist_ok=True)

    print("loading perception models...")
    perception = Perception()
    print(f"device: {perception.device}")

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        print(f"no images found in {image_dir}", file=sys.stderr)
        return 1

    for img_path in images:
        print(f"\n{img_path.name}")
        image = Image.open(img_path).convert("RGB")
        people = perception(image)
        if not people:
            print("  no person detected above threshold")
            continue
        for i, p in enumerate(people):
            x1, y1, x2, y2 = p.bbox
            print(f"  person {i}: bbox=({x1},{y1})-({x2},{y2}) "
                  f"size={x2 - x1}x{y2 - y1} det_conf={p.det_conf:.2f}")
            print(f"     gender={p.gender or '?'} "
                  f"face_det_score={p.face_det_score:.2f}")
            cov = coverage(p.seg_mask)
            for name, pct in sorted(cov.items(), key=lambda x: -x[1]):
                mark = "*" if name in TARGET_LABELS else " "
                print(f"     {mark} {name:<15} {pct:5.2f}%")

        visualize(image, people,
                  out_dir / f"{img_path.stem}_perception.png", img_path.name)

    print(f"\nvisualizations saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
