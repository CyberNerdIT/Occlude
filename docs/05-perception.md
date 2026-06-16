# Stage 4 — Perception (Detector + Segmenter + Classifier Wrapper)

> Status: complete. The three validated models from Stages 1–3 are
> wrapped behind a single callable that maps a PIL image to a list of
> per-person observations. End-to-end smoke test on the 7 test images
> runs clean. One behavioural shift relative to Stage 3 surfaced and
> is documented below (Finding 1).

## Why this stage exists

Stages 1–3 validated each ML component in isolation: SegFormer body
parsing (Stage 1), YOLO + per-bbox segmentation (Stage 2), InsightFace
gender classification (Stage 3). Each test script lived in
`scripts/test_*.py` and answered a per-model question.

The rule layer (Stage 5) doesn't consume those models in isolation. It
consumes one record per person, with the bbox, the segmentation map
on that bbox, and the gender prediction all aligned. Stage 4's whole
job is composing the three models so the rule layer has nothing to
worry about except policy — no model loading, no cropping, no per-
model device handling.

There is no new ML in this stage. There is one new dataclass and one
new class. That's the entire surface area.

## What the rule layer takes as input

`docs/03-rule-design.md:23–29` enumerated this. Stage 4 makes it
concrete as the `Person` dataclass:

| Field | Source | Stage 5 uses it for |
|---|---|---|
| `bbox: (x1, y1, x2, y2)` | YOLO | Region anchoring against source frame; final blur target |
| `det_conf: float` | YOLO | Confidence gating (low-conf detections may default to female under the over-blur policy) |
| `crop: PIL.Image` | derived | Visualization, debugging |
| `seg_mask: np.ndarray` | SegFormer | All three rule checks (Region / Shape / Tiebreaker) |
| `gender: 'M' \| 'F' \| None` | InsightFace | Which ruleset to apply |
| `face_det_score: float` | InsightFace | Default-to-female-on-low-confidence gating |

The `face_det_score` naming is deliberate. See "Finding 2" below.

What is **not** on `Person`:

- **No InsightFace face bbox.** The rule layer's anatomy anchor is
  the SegFormer **Face label**, not InsightFace's bbox
  (`docs/03-rule-design.md:65–70`). Putting the InsightFace bbox on
  the dataclass would have suggested it's part of the rule-layer
  contract. It isn't.
- **No "gender confidence."** InsightFace's high-level API does not
  surface one. We deliberately don't fabricate one.

## How the callable works

`Perception.__init__` loads YOLO (`yolov8n.pt`), SegFormer
(`mattmdjaga/segformer_b2_clothes`) on MPS / CUDA / CPU per
availability, and InsightFace `buffalo_l` on CPU explicitly (Apple-
Silicon ONNX-runtime quirks, same reasoning as Stage 3).

`Perception.__call__(image)`:

1. YOLO `predict(source=image, classes=[0], conf=0.40, verbose=False)`
   → person bboxes.
2. For each bbox: crop, run SegFormer on the crop, run InsightFace
   on the crop's BGR view.
3. Pack into a `Person` and append.

YOLO bbox confidence threshold is 0.40 — same as `test_detect_and_segment.py`.
Lower for video footage will be evaluated in Stage 6.

InsightFace runs **per-crop**, not per-frame. See Finding 1.

## Findings

### Finding 1 — Per-crop InsightFace shifts behaviour on the masked-face case

The Stage 3 doc (`docs/04-gender-classifier.md`) ran InsightFace on
the full image. Image 05 (the bald, bearded man in a black surgical
mask) was predicted **F at face_det_score 0.834** — the silent-and-
confident misclassification that motivated the cross-check rule in
Stage 5's design.

In Stage 4, InsightFace runs on the YOLO person crop instead. On the
same image, the prediction is now **M at face_det_score 0.66**. The
genderage head sees a face that occupies a much larger fraction of
its input, and apparently breaks the tie the other way. The face
detection score also drops, which is consistent with the detector
having less context.

The other six images all match Stage 3's expected map exactly.

So the documented Stage 3 failure mode for image 05 — "masked face,
silently flipped to female with high confidence" — is, on this
specific image, no longer observed under the Stage 4 calling
convention. We do **not** read that as "the failure mode is gone."
The right reading is:

- The mask-occlusion failure is real. We have one image showing it.
- That image's particular numerical outcome is sensitive to input
  scale.
- We don't have enough data to characterise *which* scale the
  failure prefers, so we can't pick "per-crop" vs "per-frame" on
  accuracy grounds.

We pick per-crop for Stage 4 anyway because it's the right pipeline
*shape*: each YOLO bbox gets its own gender prediction, with no
face-to-bbox assignment problem when frames contain multiple people.
Per-frame would force IoU matching between InsightFace face boxes
and YOLO person boxes, which is its own cluster of bugs.

The Stage 5 cross-check (Hair extending below the Face label) was
designed to compensate for misclassification *regardless* of which
scale produces it. So the cross-check still earns its keep.

### Finding 2 — `face_det_score`, not `gender_det_score`

Stage 3, Finding 2 was the most operationally important point of
that stage: `det_score` is the **face detector**'s confidence, not
the gender model's. Calling it `gender_det_score` on a dataclass —
which I caught myself almost doing — silently re-introduces the
exact conflation the prior doc fought against.

So the field is `face_det_score`. How Stage 5 uses it (as the
default-to-female threshold input, per the over-blur policy) is a
Stage 5 *policy* decision; Stage 4 reports it under its real name.

The user pushed back on this naming during code review and was
right to. Recording it here so the next person who sees the field
doesn't waste time wondering whether we actually have a calibrated
gender confidence somewhere. We don't.

### Finding 3 — Per-crop InsightFace is robust on small bboxes

The smallest bbox in the test set is image 05 (344×686). InsightFace
still finds the face with `det_size=(640, 640)`. No crop in the test
set produced an empty face list (i.e., `gender = None` did not fire
for any test image). We'll see how this holds on real video frames
where person bboxes can be much smaller.

## What this means for the pipeline

- **Stage 5 can be written.** It needs only `Person` and the
  segmenter label set (`SEG_LABELS`), both exported from
  `pipeline/perception.py`.
- **`TARGET_LABELS` lives in `pipeline/perception.py`.** Previously
  duplicated in `scripts/test_detect_and_segment.py` and the new
  `scripts/test_perception.py`. Stage 5 will import the same set.
- **Per-crop face calibration is a Stage 5 / 6 concern.** The
  default-to-female threshold on `face_det_score` will be picked
  against per-crop scores, not Stage 3's per-frame numbers. The
  numeric difference (Stage 3 ranged 0.748–0.886, Stage 4 ranged
  0.66–0.89) is small enough that the threshold can still come from
  the same test set; we just need to remember which calling
  convention generated the numbers.

## What changed about prior docs

| Earlier doc | Update | Why |
|---|---|---|
| `docs/04-gender-classifier.md`: Image 05 predicts F at 0.834 | Per-crop, Image 05 predicts M at 0.66. Per-image specific outcomes shift with input scale. The general failure mode (silent on occlusion) is still expected to occur, just not necessarily on this exact image at this exact scale. | Stage 4 runs InsightFace per YOLO crop instead of per full image |
| `docs/03-rule-design.md:308`: "Stage 4: wrap detector + segmenter as a single callable" | Wrapped detector + segmenter **and** classifier. The rule-layer input list (lines 23–29 of the same doc) names a gender prediction; folding it into Stage 4 matches what Stage 5 actually consumes. | One-callable shape is right; otherwise Stage 5 does perception orchestration mixed with rule logic |
| Stage 3 doc treats `det_score` carefully | Carried forward: field on `Person` is `face_det_score`. No gender confidence is invented. | Don't re-introduce the Stage-3 conflation |

## What remains open

- **Per-crop calibration for InsightFace on small bboxes.** Real
  video may produce 100×200 person bboxes; we haven't measured.
- **Multi-person frames.** Test set is single-person. Two-person
  scenes will exercise the per-crop classification cleanly (one
  Person record per bbox, no assignment) but might surface
  pathological cases — e.g., two YOLO bboxes with significant
  overlap pulling the same face into both crops. Stage 6 video
  testing will hit this.
- **Detection threshold tuning for video.** 0.40 is the still-image
  number. Video may need lower, with the over-blur bias absorbing
  the extra noise.
- **Alternative gender-confidence path.** Stage 3 left this open
  (drop to genderage logits for a softmax probability, or use
  behavioural signals). Stage 4 doesn't address it. Defer to
  Stage 5 or later.

## Files this stage produced

```
OCCLUDE/
├── pipeline/
│   ├── __init__.py
│   └── perception.py             (Perception, Person, SEG_LABELS, TARGET_LABELS)
├── scripts/
│   └── test_perception.py
├── test_output/
│   └── *_perception.png          (input + bboxes + per-person crop, seg, gender)
└── docs/
    └── 05-perception.md          (this file)
```

No new dependencies — `pipeline/perception.py` reuses the libraries
(`ultralytics`, `transformers`, `insightface`, `onnxruntime`,
`torch`) introduced in Stages 1–3.

## How to reproduce

```bash
.venv/bin/python scripts/test_perception.py
```

Visualizations land in `test_output/*_perception.png`. Console output
prints, per image and per detected person: bbox, YOLO confidence,
predicted gender, face detection score, and label coverage with the
modesty-relevant labels marked with `*`.
