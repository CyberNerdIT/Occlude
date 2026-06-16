"""Stage 5 — rule layer smoke test.

Runs Stage 4 perception + Stage 5 rules on every test image and
compares against the EXPECTED blur outcomes inferred from filenames
("immodest_*" → blur, "modest_*" → pass).

Saves a per-image visualization showing the seg mask with the
head / thigh region overlays drawn on top, plus the decision text.
"""
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from occlude.pipeline.perception import SEG_LABELS, Perception, Person  # noqa: E402
from occlude.pipeline.rules import Decision, RuleEngine  # noqa: E402


def expected_from_name(name: str) -> bool | None:
    if name.startswith(("01_", "02_", "06_")) or "modest_" in name and "immodest_" not in name:
        return False
    if "immodest_" in name:
        return True
    return None


def _draw_region(ax, box: tuple[int, int, int, int] | None, color: str, label: str) -> None:
    if box is None:
        return
    top, bot, left, right = box
    ax.add_patch(mpatches.Rectangle(
        (left, top), right - left, bot - top,
        fill=False, edgecolor=color, linewidth=2, linestyle="--",
    ))
    ax.text(left, max(top - 6, 0), label, color=color, fontsize=9, weight="bold")


def visualize(image: Image.Image, people: list[Person], decisions: list[Decision],
              output_path: Path, title: str) -> None:
    cmap = plt.get_cmap("tab20", len(SEG_LABELS))
    n = len(people)
    fig, axes = plt.subplots(1, 1 + 2 * n, figsize=(6 + 6 * n, 7.5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes[0].imshow(image)
    for i, (p, d) in enumerate(zip(people, decisions)):
        x1, y1, x2, y2 = p.bbox
        color = "red" if d.blur else "lime"
        axes[0].add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=3,
        ))
        axes[0].text(x1, max(y1 - 8, 0), f"#{i} {'BLUR' if d.blur else 'pass'}",
                     color=color, fontsize=11, weight="bold")
    axes[0].set_title("input + decisions")
    axes[0].axis("off")

    present: set[int] = set()
    for i, (p, d) in enumerate(zip(people, decisions)):
        ax_c = axes[1 + 2 * i]
        ax_s = axes[1 + 2 * i + 1]

        ax_c.imshow(p.crop)
        ax_c.set_title(f"#{i} crop — {d.gender_used}"
                       f"{' (override)' if d.overridden else ''} "
                       f"face={p.face_det_score:.2f}")
        ax_c.axis("off")

        ax_s.imshow(p.seg_mask, cmap=cmap, vmin=0, vmax=len(SEG_LABELS) - 1,
                    interpolation="nearest")
        _draw_region(ax_s, d.head_region, "cyan", "head")
        _draw_region(ax_s, d.thigh_region, "magenta", "thigh")
        ax_s.set_title(f"#{i} {'BLUR' if d.blur else 'pass'} — {d.reason}",
                       fontsize=9)
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

    print("loading models...")
    perception = Perception()
    rules = RuleEngine()
    print(f"device: {perception.device}")

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        print(f"no images in {image_dir}", file=sys.stderr)
        return 1

    correct = total = 0
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        people = perception(image)
        if not people:
            print(f"\n{img_path.name}: NO PERSON DETECTED")
            continue

        decisions = [rules.decide(p) for p in people]
        expected = expected_from_name(img_path.name)
        primary = decisions[0]
        ok = expected is None or expected == primary.blur
        mark = "✓" if ok else "✗"
        if expected is not None:
            total += 1
            correct += int(ok)

        print(f"\n{mark} {img_path.name}")
        for i, (p, d) in enumerate(zip(people, decisions)):
            ovr = " (override)" if d.overridden else ""
            print(f"  #{i}: {'BLUR' if d.blur else 'pass '} "
                  f"gender={d.gender_used}{ovr} "
                  f"face_det={p.face_det_score:.2f} — {d.reason}")

        visualize(image, people, decisions,
                  out_dir / f"{img_path.stem}_rules.png", img_path.name)

    if total:
        print(f"\n{correct}/{total} match expected")
    print(f"visualizations → {out_dir}")
    return 0 if correct == total else 1


if __name__ == "__main__":
    sys.exit(main())
