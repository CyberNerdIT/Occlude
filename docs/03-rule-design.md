# Stage 3 (design) — Rule Layer: Three Stacked Checks

> Status: design only. No rule code has been written. This document is
> the blueprint the rule implementation in Stage 5 will follow.
> Numerical thresholds are deliberately not set here — they'll be tuned
> against the test images once the rule code exists. The *structure*
> below should not change.

## Why this document exists

The Stage 1 and Stage 2 docs each ended with a short rule sketch. Both
sketches were correct but compressed — packed into a single paragraph
that read as "do something spatial." After walking through the design
twice in conversation (once for the female head rule, once for the male
thigh rule), it became clear the rule layer is actually **three
separate mechanisms doing three different jobs**, not one. Conflating
them is the easy way to write a rule that fails on either image 06 or
image 05. So this doc pulls them apart and writes them down before any
code gets written.

## What the rule layer takes as input

For each person detected in a frame, the rule layer receives:

- A YOLO bounding box `(x1, y1, x2, y2)`.
- A pixel-wise label map from SegFormer, computed on the cropped bbox.
  Labels are the 18 classes documented in `01-segmenter-validation.md`.
- (Eventually) a gender prediction from InsightFace with a confidence
  score. Below the confidence threshold, default to female.

Its output is one bit: **blur this person, or don't**.

## Why bbox-relative percentages alone don't work

Stage 2 left us with two findings whose root cause is the same. They're
worth restating side by side because that's what motivates everything
below.

**Female / dark hijab (image 06):** the segmenter labels 3.01% of the
woman's bbox as Hair, even though her hair is fully covered. A truly
uncovered head (image 07) scores 4.84%. The gap is 1.83 percentage
points. No global threshold on Hair % can separate the two without
being wrong on one of them.

**Male / shorts (image 05):** the segmenter labels 0.68% of the man's
bbox as Legs (the bare calves below the hem) and lumps the bare thigh
skin showing through the denim rips into the Pants label. 0.68% is too
small to threshold globally — any threshold low enough to fire on this
case will fire on socks above shoes or a sliver of ankle elsewhere.

Both cases share the same failure mode: **the percentage tells us how
much of a label exists, but throws away where it is and what shape it
forms.** The fix is to bring those two pieces of information back in.

## The three checks

The rule layer is three filters stacked. A label only counts toward a
modesty trigger if it passes all three.

### Check 1 — Region (where in the body is this label?)

The first filter is anatomical. A label only counts if its pixels fall
inside the body region the rule cares about.

The anatomy anchor is **the Face label**, not bbox geometry. Stage 2's
data made this concrete: people aren't always upright. Yoga poses,
people sitting, leaning, bending — bbox top is not always the head. So
we don't say "the head region is the top 30% of the bbox." We let
SegFormer tell us where the face is and define every other region
relative to it.

Rough definitions (proportions to be measured in Stage 5):

- **Head region.** A box around the Face label, extended upward by
  roughly one Face-height to capture hair / hat / scarf above the
  forehead.
- **Thigh region.** A box below the Face, sized by anatomical
  proportion — roughly between 2.5× and 4× Face-heights below the
  chin. That's the hip-to-knee band on a standing adult.

What this filter alone catches:

- **Neckscarf scenario.** A woman with bare hair wearing a fashion
  scarf around her neck. Scarf pixels are below the Face → not
  headcovering. Hair at the top of the head triggers the blur. ✅
- **Hand-held hat (image 07).** Hat pixels are at hip level → not
  headcovering. Hair at the top triggers blur. ✅
- **Sock above shoe.** A thin Legs strip above the ankle is below the
  thigh region → not thigh exposure. ✅

### Check 2 — Shape (is this label a real blob, or scattered noise?)

Region alone is not enough. The dark-hijab failure (image 06) is
proof: the misclassified Hair patch is on the side of the woman's
head, which means it is **inside the head region**. Region passes it.
We need a second filter that asks whether the label forms a shape
consistent with the thing it claims to be.

Real uncovered hair forms one large contiguous component covering the
top and sides of the head. A misclassified patch on a hijab is small
and isolated. So:

> The Hair label only counts if its largest connected component
> inside the head region exceeds a minimum size (proportional to the
> head region's area, not absolute pixels — bbox sizes vary).

Equivalently: throw away small isolated Hair patches before
thresholding.

Without this check, the region filter fails image 06 — the bad patch
is in the right place, just the wrong shape.

The thigh rule probably doesn't need shape filtering. Bare leg pixels
in the thigh region are already a small, specific target; there's no
analogous "skin pretending to be skin in the wrong shape" failure
mode. We'll know for sure once we run real images through Stage 5.

### Check 3 — Tiebreaker (when conflicting labels overlap)

Even after region + shape, ambiguity remains where two labels coexist
in the same region.

**Female head region:** Hair and Hat/Scarf can both be present.

> If Hat+Scarf pixel area in the head region exceeds N× the Hair area,
> treat the head as covered and suppress the Hair trigger. Otherwise
> treat the hair as uncovered.

N is small — bias toward over-blur means we want Hat/Scarf to clearly
dominate, not just slightly outnumber. Tuning happens in Stage 5.

**Male thigh region:** Pants and Legs can both be present (image 05's
ripped shorts are exactly this case).

> Do **not** apply a "Pants dominates" suppression. Pants in the thigh
> region does not mean the thigh is covered — shorts, mesh, rips all
> register as Pants. Exposed skin is determinative on its own.

This asymmetry is real, and it's the part that surprised me when we
walked through it. The female rule needs a tiebreaker because hijab
fabric can be misclassified as hair *and* real headcovering coexists
with hair anatomically. The male rule doesn't need one because pants
don't get misclassified as legs and the modesty rule is "skin in the
wrong place is enough" — there's no nearby fabric that should rescue
it.

## Mirror structure, different tuning

The female and male rules share the three-check skeleton but the
content of each check differs.

| | Female head rule | Male thigh rule |
|---|---|---|
| **Region** | Box around Face, extended upward | Hip-to-knee band below Face |
| **Shape** | Connected-component on Hair, drop small patches | Likely not needed; revisit in Stage 5 |
| **Tiebreaker** | Hat+Scarf area > N× Hair area → covered | None; exposed Legs in region is sufficient |
| **Trigger condition** | Hair survives all three → blur | Legs survives region (and shape if added) → blur |

This symmetry is by design. Implementing one and copy-adapting the
other is much cleaner than treating them as separate codepaths.

## The shirtless fast-path

Stage 1 and Stage 2 both confirmed: image 04 (shirtless man) reports
Upper-clothes ≈ 0.01% within bbox. Effectively zero, no garment, no
ambiguity.

So before running region/shape/tiebreaker for the male case:

> If Upper-clothes coverage in the bbox is below ε (small), blur. Done.

This is the only check in the entire rule layer that can use a
bbox-wide percentage safely, because the failure mode that breaks
percentages elsewhere — fabric pretending to be skin or vice versa —
doesn't exist here. Either there's a torso garment or there isn't.

## Cross-checking the gender classifier

Stage 3 (`docs/04-gender-classifier.md`) showed the gender classifier
silently fails on occluded faces — image 05 (man in a black surgical
mask) was predicted female with high detection confidence and no
warning. The dangerous direction of this failure is **woman
misclassified as male**: the male ruleset doesn't check hair, arms,
or neck-chest, so an immodestly-dressed woman in that case walks
through the pipeline unblurred. That defeats the project.

So the rule layer overrides the classifier in exactly one direction:

> If the classifier says "male" but the segmenter sees Hair pixels
> extending significantly **below** the Face label, treat the person
> as female and run the female ruleset.

The signal is "Hair extending below the Face," not "Hair anywhere in
the head region." This distinction is important and easy to get
wrong:

- A man with a normal short or medium haircut has Hair only at the
  crown, above and around the face. The override does **not** fire.
  Male rules apply. No false blur.
- A woman with uncovered long hair (image 07's exact configuration
  — hair flowing past the face, down the neck, to the shoulders) has
  Hair pixels well below the Face label. The override fires. Female
  rules apply. Blur.
- A long-haired man matches the same pattern as a long-haired woman.
  The override fires; the female rules see his uncovered hair and
  blur. This is a false positive — see "Accepted edge cases" below.

Pseudocode (numbers tuned in Stage 5):

```
if predicted_sex == "M":
    face_bottom_y = bottom edge of Face label pixels
    hair_below_face = count Hair pixels with y > face_bottom_y
    if hair_below_face > k * face_height_px:
        predicted_sex = "F"   # override
```

Why this only goes one way: a man misclassified as female enters the
female ruleset, which is strictly stricter, so the worst outcome is
over-blur — already the project's preferred error. No override
needed in that direction.

### Accepted edge cases

The cross-check has known costs we choose to pay. Recording them
here so a future contributor doesn't quietly "fix" them and reopen
worse problems.

- **Long-haired men get over-blurred.** A clean-shaven man with
  shoulder-length hair triggers the override, ends up in the female
  ruleset, female rules say "uncovered hair → blur." False positive.
  Population is small; cost (one long-haired man visually blurred
  in a video) is minor; under-blurring an actual woman with
  uncovered hair is much worse. We accept the trade.
- **Misclassified women with very short hair may still escape.** A
  woman with cropped hair predicted male won't trigger the override
  — no Hair below the Face. She runs through the male rules; if
  none of them fire, no blur. Failure intersection is small (woman-
  as-man misclassification × very short hair × no thigh / torso
  trigger). If Stage 5 testing shows it happens often we can add
  parallel one-directional overrides on bare-arms or exposed
  neck-chest. We don't pre-build that complexity.
- **Face-occluded subjects (image 05).** The classifier is silently
  wrong; the override may or may not save us depending on whether
  the subject also has visible long hair. We accept this for v1.
  Document in the README so users know.
- **Naive Hair % thresholds — explicitly rejected.** A "if Hair
  pixels exist anywhere in the head region, override to female"
  rule would over-blur every clothed man with normal hair, which
  is catastrophic in the other direction. We need the spatial
  refinement (below the face, not anywhere in the head region) for
  the override to be useful at all.

These trade-offs follow directly from the bias-toward-over-blur
policy: no off-the-shelf model gets every case right; we choose
which error to live with.

## Bias toward over-blur (policy carry-over)

From Stage 1, restated so this doc is self-contained:

> When the rule is uncertain, blur.

Costs are asymmetric. Blurring a modestly dressed person is a minor
visual annoyance — audio still plays, content is still consumable.
Failing to blur an immodest person defeats the whole purpose of
running OCCLUDE. So the thresholds in Stage 5 will be set on the
side of *more* blur, not less. Same policy applies to the gender
classifier: low-confidence prediction defaults to female (stricter
ruleset).

## What is still uncertain

Honest list:

- **Numerical thresholds for everything.** Hair-CC size, Hat+Scarf vs
  Hair multiplier, thigh Legs threshold, shirtless ε. These get
  measured against the test images in Stage 5, then re-tuned on more
  images.
- **Anatomical proportions.** "Hip ≈ 2.5× Face-height below chin,
  knee ≈ 4×" is a placeholder until we measure on real upright
  photos. Non-upright poses (yoga, sitting, bending) are an open
  question — three-check rules anchored on Face will degrade
  gracefully but not always correctly.
- **Bandanas, hoods, caps over ponytails.** Stage 1 flagged these as
  hard. We'll evaluate after the three-check rule exists and decide
  whether they justify a fourth mechanism or fall under the
  over-blur bias.
- **Whether the male rule actually needs shape filtering.** Maybe one
  small Legs patch from a stray pixel inside the thigh region triggers
  spurious blur. We'll know once we run it.

## What changed vs the earlier docs

| Earlier doc said | This doc clarifies | Why it matters |
|---|---|---|
| Stage 1: "rules must be spatial" | Spatial alone fails image 06 — the bad patch is inside the head region. Need shape too. | Dark-hijab false-positive class wouldn't get caught |
| Stage 2: "male rule needs the same kind of spatial reasoning" | Same *structure* (region/shape/tiebreaker), different *content*. Male skips shape and tiebreaker; female needs both. | Avoids over-engineering the male rule and under-engineering the female one |
| Both: "head region defined by Face + bbox" | Face is the anatomy anchor for **every** body region, not just the head. Bbox geometry alone breaks on non-upright poses. | Will affect how the regions are computed in code |
| Both: "Hat or Scarf means head is covered" | Hat/Scarf must (a) be in the head region and (b) dominate the Hair area in that region. Two conditions, not one. | Single-condition rule fails the dark hijab |

## What remains open (for the implementation stages)

Same as before, with the rule design now committed:

- **Stage 3 (impl):** gender classifier — InsightFace `buffalo_l`,
  default-to-female on low confidence.
- **Stage 4:** wrap detector + segmenter as a single callable that
  returns the data structure the rule layer consumes.
- **Stage 5:** implement the three checks above and tune thresholds
  against the 7 test images.
- **Stages 6–8:** blur, temporal smoothing, video reconstruction.
