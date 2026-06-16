# Stage 3 — Gender Classification with InsightFace

> Status: complete. The classifier works on unoccluded faces (6/7
> correct on the test set) and has one known failure mode that
> matters: face occlusion causes a silently-wrong prediction at high
> detection confidence. The "default to female on low confidence"
> policy from `01-segmenter-validation.md` partially survives — but
> the confidence signal it relied on is not what we hoped.

## Why this stage exists

The pipeline has to choose which modesty ruleset to apply per person —
female (stricter, checks hair/arms/legs/neck-chest) or male (checks
navel/thighs). Stage 1 named this as the riskiest pipeline step
because misclassifying a woman as male skips the hair check, which is
the female ruleset's primary trigger. Stage 1 also set the policy:
**when the classifier is uncertain, default to female**, on the
asymmetric-cost reasoning that over-blur is annoying but under-blur
defeats the project.

This stage validates whether InsightFace `buffalo_l` is good enough,
and — more importantly — whether its confidence output is actually
usable as the "uncertain" signal the policy relies on.

## What InsightFace gives us

`buffalo_l` is a packaged model bundle from the InsightFace project
that runs face detection, alignment, age, gender, recognition, and
landmarks in one pass. For our purposes only two things matter:

- **`face.gender`** — `0` for female, `1` for male. The argmax of the
  internal genderage model. The high-level API does **not** expose a
  per-class probability.
- **`face.det_score`** — the *face detection* confidence. How sure the
  model is that there's a face here, not whether the gender is right.

That distinction turns out to matter a lot. (See Finding 2 below.)

The bundle downloads on first run (~280 MB) into
`~/.insightface/models/buffalo_l/`. We run it on `CPUExecutionProvider`
explicitly to avoid CoreML / ONNX-runtime quirks on Apple Silicon at
the validation stage. CoreML conversion can come later if needed.

## Test setup

`scripts/test_classifier.py`:

1. Loads `buffalo_l` with `det_size=(640, 640)`.
2. For each image in `test_images/`, runs `app.get(bgr)`.
3. Sorts detected faces by `det_score` descending and reports each.
4. Compares the top face's gender against an `EXPECTED` map.
5. Saves an annotated PNG with the face bbox and predicted sex /
   det_score drawn on it.

## What the model said

| # | Image | Expected | Predicted | det_score | Age | Match |
|---|---|---|---|---:|---:|:---:|
| 01 | modest_woman_hijab | F | F | 0.785 | 24 | ✓ |
| 02 | modest_man_suit | M | M | 0.886 | 31 | ✓ |
| 03 | immodest_woman_dress | F | F | 0.842 | 31 | ✓ |
| 04 | immodest_shirtless_man | M | M | 0.857 | 47 | ✓ |
| 05 | immodest_man_shorts | M | **F** | 0.834 | 35 | ✗ |
| 06 | modest_woman_abaya | F | F | 0.748 | 25 | ✓ |
| 07 | immodest_woman_longsleeve | F | F | 0.821 | 26 | ✓ |

Overall: **6/7 top-face predictions match expected.**

## Findings

### Finding 1 — On unoccluded faces, the classifier works

The six correct cases include a tightly-framed light hijab (01), a
dark-hijab abaya (06), a sleeveless dress (03), a suit (02), a
shirtless beach photo (04), and a long-sleeve casual shot (07). Hair
covered or uncovered, indoor or outdoor, lighting variation,
expression variation — the model didn't flinch on any of them.
det_score across the six is 0.748 to 0.886. There's nothing in this
sample to distinguish "easy" from "hard" cases by the score alone.

### Finding 2 — det_score is detection confidence, not gender confidence

This is the most operationally important finding of the stage.

`det_score` answers "is there a face here?" The genderage head runs
*after* a face is found and produces a gender label with no
calibrated probability surfaced through the API. So the high-level
InsightFace output gives us no direct signal for "is the gender
prediction reliable."

The plan from Stage 1 was implicitly: *low confidence → default to
female*. We need to figure out which confidence signal that means.
The honest answer right now is: **we don't have one**. det_score
won't do it (Finding 3 is the proof). We'll have to either:

- (a) drop down to the genderage model directly and read the raw
  logits to compute a softmax probability, or
- (b) use a *behavioural* uncertainty signal — face bbox very small,
  conflicting predictions across nearby frames, segmenter Hair
  coverage disagreeing with predicted gender — and treat those as the
  "uncertain" trigger.

Both are deferrable to Stage 4/5 once the rule layer exists. For now,
we record the gap.

### Finding 3 — Face occlusion produces a silently-wrong prediction

Image 05 is a bald, bearded man in a black surgical mask covering his
nose and mouth. The face is detected cleanly (det_score 0.834,
comfortably in the same range as the correct cases) but the gender is
flipped to female. Even the visible beard along the jawline isn't
enough to override what the masked features tell the model.

This matters because:

1. Masked faces are common in real footage (post-2020 documentary,
   medical settings, religious contexts, winter clothing with scarves
   pulled up, surgical / industrial PPE).
2. The failure is **silent and confident**. det_score gives no
   warning. Nothing in the high-level API flags this prediction as
   different from the correct ones.
3. It's the first concrete instance of misclassification we've seen,
   and it happens in the wrong direction relative to the over-blur
   policy. (Or does it — see Finding 4.)

### Finding 4 — The over-blur bias accidentally absorbs image 05

The misclassification on 05 sends the man into the *female* ruleset.
Female rules look for: uncovered hair, bare arms, bare legs, exposed
neck-chest. Image 05 the subject:

- is bald with a beard → 0 Hair pixels → hair check doesn't fire.
- has bare arms with tattoos → arms check fires.
- has bare calves below shorts → legs check fires (though small in
  bbox area, this is exactly the kind of case Stage 5's spatial rule
  should catch).

So the female ruleset on image 05 still produces a *blur* decision,
just by a different reasoning path than the male ruleset would have
taken. The over-blur bias rescues the outcome despite the wrong
classification. That's the design working as intended.

But this is a property of *this particular* misclassification
direction. (Finding 5.)

### Finding 5 — The dangerous direction is woman → man, and it's untested

A man classified as a woman is mostly safe under our policy: the
female ruleset is strictly stricter for typical content (it checks
more body regions). The reverse — a *woman* classified as a man —
removes the hair check from the rule layer's repertoire entirely. A
woman with uncovered hair and a long-sleeve top + jeans (image 07's
exact configuration) would skip every female trigger and fall through
the male ruleset, which only checks thigh/navel/groin coverage. None
of those fire. **No blur.** Catastrophic miss.

We have no test image of a woman who could plausibly be misclassified
as male. Until we do, we don't know whether this failure mode is
common enough to design around. Action item: add such a test image
in Stage 5 prep.

## What this means for the pipeline

Concrete plan, given the findings:

1. **Use `face.gender` as the prediction.** It's accurate when the
   face is unoccluded.
2. **Don't trust `det_score` as a gender-confidence proxy.** Use it
   only as "is there a face at all" gating.
3. **Default to female** when:
   - No face is detected in the YOLO bbox, or
   - det_score < some minimum (we'll pick a number once we have more
     data — none of our 7 images are below 0.74), or
   - The segmenter sees Hair pixels extending significantly *below*
     the Face label and the classifier said male (woman-as-man guard,
     addresses Finding 5).
4. **Live with the masked-face failure mode for v1.** Document it
   prominently in the README; the over-blur bias absorbs it for
   typical immodest-male cases (Finding 4). Consider a calibrated
   gender-confidence path in a later iteration.

The cross-check in (3) is the important one. It uses segmenter output
to second-guess the gender prediction in one specific direction
(male → female only), and it triggers on a specific pattern (Hair
*below the face*, not Hair anywhere in the head region). The
distinction matters: a naive "any Hair → override" would over-blur
every clothed man with normal hair, which is catastrophic. Stage 5
holds the cross-check logic; the rule-design doc
(`docs/03-rule-design.md`) documents the full pseudocode and the
edge cases we accept (long-haired men over-blurred, very-short-haired
misclassified women possibly missed).

## What changed about prior docs

| Earlier doc | Update | Why |
|---|---|---|
| Stage 1: "default to female on low confidence" | Confidence signal isn't `det_score`. Need to derive one from raw logits or use behavioural signals. | High-level API doesn't expose calibrated gender probability |
| Stage 1: classifier is the riskiest pipeline step | Confirmed. Worse than expected — failures are silent, not flagged | Image 05 mask case |
| Stage 3 (rule design): "low gender confidence → female" | Add: "predicted male + Hair pixels extending significantly below the Face label → override to female." Specifically NOT "any Hair in head region" — that would over-blur every clothed man with a normal haircut. | Guards the woman-as-man failure direction without breaking the man-with-normal-hair case |

## What remains open

- A "woman who could be misclassified as a man" test image, to
  characterize the dangerous direction.
- A behavioural / cross-check uncertainty signal, since the obvious
  one (det_score) doesn't carry the information we wanted.
- Whether to drop down to the raw genderage model for calibrated
  probabilities. Worth doing if cross-check signals turn out to be
  insufficient.
- Per-frame inference time of detector + segmenter + classifier on
  the M4. Still unmeasured.

## Files this stage produced

```
OCCLUDE/
├── scripts/
│   └── test_classifier.py
├── test_output/
│   └── *_gender.png        (face bbox + predicted sex + det_score)
└── docs/
    └── 04-gender-classifier.md   (this file)
```

`requirements.txt` gained `insightface>=0.7` and `onnxruntime>=1.16`.

## How to reproduce

```bash
uv pip install -r requirements.txt
.venv/bin/python scripts/test_classifier.py
```

First run downloads the `buffalo_l` bundle (~280 MB) into
`~/.insightface/models/`. Subsequent runs are offline.
