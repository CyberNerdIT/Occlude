# Stage 5 — Rule Layer Implementation

> Status: complete. Maps a Stage 4 `Person` to a binary blur decision.
> All 7 test images match their expected outcomes (4 immodest → blur,
> 3 modest → pass). Two findings shifted the tuning relative to the
> Stage 3 design (`docs/03-rule-design.md`); both are documented
> below as constraints to remember when the test set grows.

## What was built

`pipeline/rules.py` exports `RuleEngine` and `Decision`. The engine
runs:

1. **Default-to-female** if `face_det_score < MIN_FACE_DET_SCORE`
   (0.50) or `gender is None`.
2. **Face-label anchor.** Compute the bounding box of the SegFormer
   `Face` label inside the person crop. If the Face label is absent
   entirely → **over-blur** (we have no anatomy to anchor against,
   and the asymmetric-cost policy wants the safe error).
3. **Gender cross-check** (predicted `M` only): count Hair pixels with
   `y > face_bottom_y`. If that count exceeds
   `CROSSCHECK_HAIR_AREA_FACTOR × face_area`, override to `F`. Note:
   the threshold is normalised against face *area*, not face height
   — a pixel count vs a linear dimension is dimensionally
   inconsistent and would trigger on segmenter noise.
4. **Apply ruleset**:
   - Male: shirtless fast-path (`Upper-clothes < SHIRTLESS_EPS_PCT`)
     → thigh-region Legs check.
   - Female: Hair (Region + Shape + Tiebreaker) → bare arms
     (bbox-wide %) → bare legs (bbox-wide %).

The first triggered rule wins. `Decision.reason` records which.
`Decision.head_region` and `.thigh_region` are populated for the
test-script overlay.

## Anatomy regions

The Face label is the anatomy anchor for *every* region, per the
Stage 3 design. Head and thigh regions are sized by Face dimensions,
not bbox geometry — so non-upright poses degrade gracefully (the
regions tilt with the face position rather than stay locked to the
top of the bbox).

- **Head region.** `[face_top - 1·face_h, face_bot] ×
  [face_left - 0.5·face_w, face_right + 0.5·face_w]`. Captures hair
  / hat / scarf above the forehead and at the temples.
- **Thigh region.** `[face_bot + 2.5·face_h, face_bot + 4·face_h] ×
  [bbox full width]`. The hip-to-knee band on a standing adult.
  Below-knee skin is *modest by spec* (`OCCLUDE_SPEC.md:69`) and
  is intentionally outside this region.

## Test-set walk-through

| # | Image | Expected | Got | Reason | gender |
|---|---|---|---|---|---|
| 01 | modest_woman_hijab | pass | pass | female: covered | F (0.77) |
| 02 | modest_man_suit | pass | pass | male: covered | M (0.86) |
| 03 | immodest_woman_dress | BLUR | BLUR | uncovered hair (CC 0.107 of head, Hat+Scarf 0) | F (0.85) |
| 04 | immodest_shirtless_man | BLUR | BLUR | shirtless (Upper-clothes 0.01%) | M (0.87) |
| 05 | immodest_man_shorts | BLUR | BLUR | bare thigh (Legs 3.22% in thigh region) | M (0.66) |
| 06 | modest_woman_abaya | pass | pass | female: covered | F (0.77) |
| 07 | immodest_woman_longsleeve | BLUR | BLUR | uncovered hair (CC 0.312 of head, Hat+Scarf 0) | F (0.89) |

7/7. Visualisations in `test_output/*_rules.png` show the head /
thigh region overlays so the reader can see *why* each decision
fired (or didn't).

## Findings

### Finding 1 — Image 05 surprised us (in a good way)

Going into Stage 5, the working assumption was that image 05 (man in
ripped shorts) wouldn't blur under a design-faithful rule. The
reasoning, walked through with the advisor: bare thigh skin through
the rips is misclassified as Pants by the segmenter
(`docs/03-rule-design.md:46–50`); the only Legs label fires on the
calves; calves are below the knee and therefore outside the thigh
region; the cross-check (Hair below face) doesn't apply to a bald
subject; v1 misses the case.

It blurred anyway. The thigh region as defined
(`[chin + 2.5·face_h, chin + 4·face_h]`) on this specific bbox
contains 3.22% Legs pixels, well above the 1% threshold. Looking at
the visualisation, the thigh band overlaps the upper portion of the
calves — face is positioned high in the bbox, face_h is small
(small face → narrow thigh band scaled to face), and the bbox
extends low enough that 4·face_h below chin lands somewhere on the
shin/knee, not the actual thigh.

So we got the right answer for an only-partly-right reason. If the
person were taller in frame (face_h proportionally larger), the
thigh region would correctly land at the thighs and miss the calves
entirely — and image 05 would not blur. We don't have a test image
where this matters, so we can't tune for it. Recording the
fragility here.

### Finding 2 — The Hat+Scarf "must dominate" intent doesn't survive image 06

Stage 3 design said:

> N is small — bias toward over-blur means we want Hat/Scarf to
> clearly dominate, not just slightly outnumber.

That implies `HATSCARF_HAIR_RATIO ≥ 1` (Hat+Scarf must outnumber
Hair). On image 06 (dark abaya — *the* canonical case this
tiebreaker was designed for), the segmenter emits Hat+Scarf:Hair =
18388:27489 in the head region. **Ratio 0.67. Hat+Scarf is
outnumbered by misclassified Hair pixels.** With `N = 1`, image 06
over-blurs. Defeats the rule's purpose.

`N = 0.5` ("substantial Hat/Scarf presence rescues") matches the
data while keeping the tiebreaker meaningful — `N = 0` would say
"any Hat/Scarf at all rescues," which is wrong.

This is a documented relaxation of the design. Trade-off: a
baseball cap + uncovered ponytail (Hat ≈ 0.5·Hair, hair really is
uncovered) gets under-blurred under `N = 0.5`. Doc 03 already
flagged cap+ponytail as a known v1 gap; we're not making it worse,
but we're also not solving it.

Other thresholds along the same head-rule are tight:

- `HAIR_MIN_CC_FRAC = 0.03`. Image 06's largest Hair CC is 0.092 of
  head area — well above 0.03. The shape filter alone does *not*
  drop the misclassified abaya patch; the tiebreaker has to.
- Image 03's CC fraction is 0.107, image 06 is 0.092. Tightening
  the shape filter to 0.10 would also fix image 06 but creates a
  thin margin (0.015) that won't survive a slightly worse
  segmentation. The tiebreaker fix is more robust because it uses a
  ratio that scales with both labels.

### Finding 3 — The cross-check unit fix matters

The Stage 3 design wrote the cross-check as
`hair_below_face > k * face_height_px`, comparing a pixel *count*
to a *linear dimension*. With `k=1` and a 70-pixel face height, the
threshold is 70 pixels — within segmentation noise on a 344×686
crop. A bald man with a few stray Hair pixels mis-labelled at neck
level could trigger an unintended `M → F` override.

`pipeline/rules.py` uses `k * face_area` instead — comparing pixel
count to a pixel count. With `k = 0.5` ("about half a face-area's
worth of Hair below the face"), the threshold is ~2400 pixels on
the same crop, which is what an actual long-haired subject
produces and is well above noise.

Verified on the test set: cross-check does not fire for any image,
including image 05 (bald, predicted M with face_det 0.66). Doesn't
help image 05 — but doesn't fire spuriously either. The
woman-misclassified-as-man case it was designed for is still
absent from the test set.

## Tunable thresholds, with constraints

```
MIN_FACE_DET_SCORE          = 0.50   # below → default to F
SHIRTLESS_EPS_PCT           = 0.5    # Upper-clothes < this → shirtless
HEAD_UP_FACTOR              = 1.0    # head box extension above Face
HEAD_SIDE_FACTOR            = 0.5    # head box extension on each side
HAIR_MIN_CC_FRAC            = 0.03   # min largest-CC frac of head area
HATSCARF_HAIR_RATIO         = 0.5    # see Finding 2
ARMS_MIN_PCT                = 4.0    # bare arms (female, bbox-wide)
LEGS_MIN_PCT                = 2.0    # bare legs (female, bbox-wide)
THIGH_TOP_FACTOR            = 2.5    # thigh region top, face_h units
THIGH_BOT_FACTOR            = 4.0    # thigh region bottom
THIGH_LEGS_PCT              = 1.0    # Legs % in thigh region (male)
CROSSCHECK_HAIR_AREA_FACTOR = 0.5    # see Finding 3
```

What constrains each:

| Threshold | What it separates |
|---|---|
| `MIN_FACE_DET_SCORE` | All test images sit above 0.66; nothing exercises below |
| `SHIRTLESS_EPS_PCT` | Image 04 at 0.01% vs image 02 at 26%, image 05 at 32% |
| `HAIR_MIN_CC_FRAC` | Below image 06's 0.092 — shape filter alone won't catch the abaya |
| `HATSCARF_HAIR_RATIO` | Image 06 at 0.67 vs image 03/07 at 0 (no Hat/Scarf) |
| `ARMS_MIN_PCT` | Image 03 at 8.10% vs image 06 noise at 2.86% |
| `LEGS_MIN_PCT` | Placeholder; no test image exercises the female legs case |
| `THIGH_LEGS_PCT` | Image 05 at 3.22% vs image 02 at 0% |
| `CROSSCHECK_HAIR_AREA_FACTOR` | Above noise; no test image fires it |

Several thresholds are constrained by exactly one image. They will
need re-tuning on the larger test set Stage 6 produces.

## Female arms / legs are flat percentages — v1 simplification

Stage 3 design's three-check skeleton (Region + Shape + Tiebreaker)
was only fully specified for the female *head* rule. Female arms and
legs in `pipeline/rules.py` are flat bbox-wide percentages: > 4.0%
for arms, > 2.0% for legs.

This works on the current test set:

- Image 03 (sleeveless dress): Left+Right-arm 8.10% → fires.
- Image 06 (abaya, segmenter noise): Left-arm 2.86% → does not fire.

The cases that would collapse under bbox-% but separate cleanly
under a region check on the upper-arm zone:

- A modest woman in a *sleeveless* shirt (arms covered by an
  overlay) where the segmenter still labels arm pixels: fires
  spuriously.
- A strapless top vs. a one-shoulder asymmetric top: bbox-% can't
  distinguish; a shoulder-region check could.

When the test set grows to include those cases, this simplification
will need to be replaced with a Stage-3-style region rule. Recorded
here so the next person doesn't quietly assume the head rule's
sophistication carries over.

## What this stage didn't address

- **Neck / chest exposure** is in `OCCLUDE_SPEC.md:54` for
  women. SegFormer doesn't have a neck/chest label, so we can't
  detect it directly. A heuristic ("Face label extending below the
  chin into the upper-clothes region" or similar) might work but
  isn't built. Documented gap.
- **Per-frame inference time** still unmeasured. Stage 6 will care.
- **Multi-person frames.** Test set is single-person; the rule
  engine handles each Person independently so it should scale, but
  no real test.
- **Cap + ponytail** (`docs/03-rule-design.md:285–288`). Still a
  documented v1 gap; lowering `HATSCARF_HAIR_RATIO` to 0.5 made it
  *more* likely to under-blur, not less. We accept the trade.

## What changed about prior docs

| Earlier doc | Update | Why |
|---|---|---|
| `docs/03-rule-design.md`: "N is small — Hat/Scarf must clearly dominate" | Tuned to N=0.5. The dark-abaya segmentation emits more misclassified Hair than Hat/Scarf pixels, so `N ≥ 1` over-blurs the case the rule was designed for. | Image 06 data forced the relaxation |
| `docs/03-rule-design.md`: cross-check `hair_below_face > k * face_height_px` | Implemented as `> k * face_area` with k=0.5 | Pixel count vs linear dimension is dimensionally inconsistent and would fire on noise (advisor catch) |
| `docs/03-rule-design.md`: implicit assumption that female arms/legs would mirror the head rule's three-check structure | Implemented as flat bbox-% for v1 | Test set doesn't exercise the cases that demand region-level reasoning here; documented as simplification |

## What remains open

Same as the Stage 3 design's open list, plus:

- **Threshold robustness on real video.** Most thresholds are fixed
  by exactly one test image. We need 100s of frames before we trust
  the numbers.
- **The `THIGH_LEGS_PCT` semantics in non-standing poses.** Region
  is anchored on Face proportions, but the thigh region's *content*
  (what body parts actually fall in the band) depends on body pose.
- **Female arms/legs region rule** — see "v1 simplification"
  above.

## Files this stage produced

```
OCCLUDE/
├── pipeline/
│   └── rules.py
├── scripts/
│   └── test_rules.py
├── test_output/
│   └── *_rules.png             (input + bbox + per-person overlay + reason)
└── docs/
    └── 06-rule-implementation.md  (this file)
```

`scipy` is used for `scipy.ndimage.label` (connected components). It
was already a transitive dependency of the existing requirements; no
direct addition was needed for the smoke test to pass.

## How to reproduce

```bash
.venv/bin/python scripts/test_rules.py
```

Per-image decisions print to stdout. Visualisations land in
`test_output/*_rules.png`. The `head` (cyan) and `thigh` (magenta)
overlays on the segmentation panel show what region the rule layer
was reasoning about; the title strip lists the firing reason.
