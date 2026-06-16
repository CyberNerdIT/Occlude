# Stage 2 — YOLO Person Detection + Per-Bbox Segmentation

> Status: complete. The detector works as expected and per-bbox label
> percentages confirm what Stage 1 predicted: cropping cleans up the
> signal for arms/legs, leaves the shirtless signal intact, and does
> **not** fix the dark-hijab Hair misclassification.

## Why this stage exists

Stage 1 ran SegFormer on whole images and saw signals get diluted when
the subject was small in the frame (image 05's "0.20% legs"). The fix
laid out in the spec — and confirmed in `01-segmenter-validation.md` —
is to run YOLO first, crop each detected person, and run the segmenter
on that crop so coverage percentages are computed relative to the
person, not the whole image.

This stage adds the YOLO step and re-runs the segmenter on the crops.
It is still a measurement stage; no rule logic has been written yet.

## Models

- **Detector:** `yolov8n.pt` (Ultralytics, COCO classes; class 0 = person).
  Loads in seconds, ~6 MB weights, runs comfortably on the M4.
- **Segmenter:** unchanged from Stage 1
  (`mattmdjaga/segformer_b2_clothes`).

We restrict YOLO to class 0 (`classes=[0]`) so it never returns cars,
bottles, or anything else; we only ever want person boxes.

## What we built

Two scripts, both in `scripts/`:

1. `test_detector.py` — runs YOLO on each `test_images/*.jpg`, prints
   one line per detected person (bbox, size, confidence, fraction of
   the image), and writes an annotated `*_det.png` to `test_output/`.
2. `test_detect_and_segment.py` — runs YOLO with a confidence threshold
   of `0.40`, then for each surviving box crops the image and runs
   SegFormer on the crop. Per-label coverage is reported relative to
   the bbox. Writes a 1×(1+2N) panel — full image with red bboxes,
   then crop+segmentation pair for each detected person — to
   `*_detseg.png`.

The 0.40 confidence threshold matters: image 05 produces a spurious
0.29-confidence detection of a 22×64 px region (0.1% of the image) in
addition to the real subject. Anything above ~0.35 filters it out. We
picked 0.40 as a comfortable margin.

## What the per-bbox numbers say

The columns marked "img" are the Stage 1 numbers (% of full image);
the "bbox" columns are this stage's numbers (% of the person crop).
All numbers in percent.

| # | Subject              | Hair img | Hair bbox | Arms img (L+R) | Arms bbox (L+R) | Legs img (L+R) | Legs bbox (L+R) | Upper img | Upper bbox |
|---|----------------------|---------:|----------:|---------------:|----------------:|---------------:|----------------:|----------:|-----------:|
| 01 | hijab woman (light)  |     0.48 |      0.34 |           0.00 |            0.00 |           0.00 |            0.00 |     18.57 |      51.55 |
| 02 | suit man             |     0.39 |      1.07 |           0.57 |            1.45 |           0.00 |            0.00 |      9.39 |      26.85 |
| 03 | sleeveless jumpsuit  |     0.49 |      1.32 |           2.38 |            8.10 |           0.08 |            0.10 |      1.08 |       3.82 |
| 04 | shirtless man        |     0.36 |      1.02 |           2.82 |            7.96 |           3.83 |           10.61 |      0.00 |       0.01 |
| 05 | t-shirt + ripped shorts |  0.37 |      0.87 |           0.65 |            0.47 |           0.20 |            0.68 |      4.50 |      32.66 |
| 06 | dark hijab abaya     | **4.15** |  **3.01** |           1.57 |            2.86 |           0.00 |            0.00 |     10.41 |      33.40 |
| 07 | uncovered hair       |     2.92 |      4.84 |           0.24 |            0.41 |           0.00 |            0.00 |      4.09 |       6.53 |

## Findings

### Finding 1 — Cropping amplifies the true positives

This is exactly what Stage 1 predicted.

- Image 03 (bare arms): 2.38% → **8.10%**. A clear, easy-to-threshold
  signal where before it was borderline.
- Image 04 (shirtless legs): 3.83% → **10.61%**. Same story.
- Image 07 (uncovered hair): 2.92% → **4.84%**. Stronger.

For the modesty rules, this is the right direction: signals we want to
fire on are louder relative to the person's bbox than they were
relative to the frame.

### Finding 2 — The shirtless signal stays clean

Image 04 went from 0.00% Upper-clothes to **0.01%** within the bbox.
Effectively still zero. Stage 1 called this "almost-binary" and
cropping doesn't degrade it. The male torso rule can lean on
Upper-clothes < some small ε.

### Finding 3 — Cropping does NOT fix the dark hijab false positive

Image 06's Hair % went from 4.15% (image-relative) to **3.01%**
(bbox-relative). Smaller, but still substantially higher than image
01's 0.34% (light hijab, also fully covered). The visualization shows
the same thing Stage 1 documented: a localized Hair patch on the side
of a dark headcovering, with the rest of the hijab labelled
correctly.

This rules out one of the simplest possible rule designs ("normalize
Hair coverage to bbox area, then threshold"). A bbox-relative
threshold strict enough to flag image 07's 4.84% would also flag
image 06's 3.01%. We need the spatial / connected-component logic
Stage 1 sketched out.

### Finding 4 — Image 05 IS a real shorts image, and it reveals a male-rule gotcha

I almost shipped this writeup with the wrong claim. Direct inspection
of `test_images/05_immodest_man_shorts.jpg` shows a man in a black
top and **white ripped denim shorts that end above the knee**. His
calves are bare below the hem; his thighs are partly visible through
the rips in the denim. So the image does test the male thigh trigger
— it just produces a much weaker signal than expected, and that
weakness is informative.

What the segmenter actually does on this crop:

- Labels the **whole shorts garment as Pants** (14.99% of bbox). The
  ATR-trained label set treats "Pants" as any leg garment, so a shorts
  garment is reported as Pants. Fine — it's a labelling convention,
  not an error.
- Reports **Legs at only 0.68% of bbox** (Left+Right combined). That's
  the bare calves below the hem, which the photo angle keeps small
  (subject is in profile, partly occluded by a lamppost).
- Reports **no Legs pixels for the bare thigh area showing through the
  rips**. The denim pattern dominates and the small skin patches
  inside the rips appear to inherit the Pants label.

Two consequences for Stage 5:

1. **The male thigh trigger cannot be a simple bbox-relative Legs
   threshold.** A threshold strict enough to fire on 0.68% would also
   fire on edge cases like rolled-up sleeves or short socks above
   shoes. The rule has to reason *spatially* — "Legs pixels in the
   thigh region (between hip and knee)" — the same way the female
   head-region rule does. Symmetric design.
2. **Ripped / cutout / mesh garments are a known soft spot.** The
   segmenter prefers garment-shape labels and won't reliably surface
   the small skin gaps inside them. Not a blocker for v1, but the
   rule layer should not assume "Pants present → thigh covered."

Image 05 is therefore a *useful* test, not a missed one. It just
showed us that the male rule needs the same kind of spatial reasoning
the female rule does — and that the segmenter's "Pants" label cannot
be used as a positive proxy for "thigh covered."

### Finding 5 — YOLO confidence threshold matters even at this stage

Without `conf=0.40`, image 05 produces a second 0.29-confidence box of
22×64 px on what looks like a lamppost. That box is small enough that
it would still get cropped and fed to the segmenter, wasting a forward
pass and producing noise. A reasonable-confidence floor at the
detector belongs in the pipeline from day one.

## What the rule layer needs to do (still — same conclusion as Stage 1)

The Stage 1 rule sketch survives intact:

1. Define a head region inside the YOLO bbox using the Face label
   location.
2. For Hair: only count Hair pixels inside the head region; require
   them to form a contiguous component above some pixel threshold;
   suppress if Hat/Scarf inside the head region dominate.
3. For Hat/Scarf as headcovering: only count if their pixels are
   inside the head region (handles the neckscarf and the hand-held
   hat).
4. For shirtless: Upper-clothes < ε within bbox.
5. For thigh exposure: Left-leg + Right-leg pixels **inside a thigh
   region** (between estimated hip and knee within the bbox). Plain
   bbox-relative thresholds on Legs will not work — image 05 shows
   why.

Nothing in Stage 2's data argues for a different design.

## Files this stage produced

```
OCCLUDE/
├── scripts/
│   ├── test_detector.py             (new)
│   └── test_detect_and_segment.py   (new)
├── test_output/
│   ├── *_det.png                    (YOLO bbox annotations)
│   └── *_detseg.png                 (full image + per-person crop+seg)
└── docs/
    └── 02-detector-and-bbox-segmentation.md   (this file)
```

`requirements.txt` gained `ultralytics>=8.2` and `opencv-python>=4.9`.

## How to reproduce

```bash
uv pip install -r requirements.txt          # picks up new ultralytics + opencv-python
.venv/bin/python scripts/test_detector.py
.venv/bin/python scripts/test_detect_and_segment.py
```

The first script downloads `yolov8n.pt` (~6 MB) into the project on
first run. The segmenter weights are still cached by HuggingFace from
Stage 1.

## What remains open

Same backlog as the end of Stage 1, minus the YOLO item:

- Stage 3: gender classifier (InsightFace `buffalo_l`), with
  low-confidence → female default.
- Stage 4 + 5: spatial rule logic — head region, connected-component
  filtering on Hair, Hat/Scarf vs Hair tiebreaker, shirtless threshold.
- Stage 6–8: blur application, temporal smoothing, video reconstruction.
- Open performance question: per-frame inference time on the M4 is
  unmeasured. Will revisit once the full per-frame pipeline exists.
