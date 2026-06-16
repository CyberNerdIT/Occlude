# Stage 1 — Segmentation Model Validation

> Status: complete. This is the log of what we did, what we found, what we
> argued about, and what changed our minds.

## Why this stage exists

The OCCLUDE spec lays out an 8-step pipeline (frame extraction → person
detection → gender classification → body part segmentation → rule
application → blur → temporal smoothing → video reconstruction). It's
tempting to start writing pipeline plumbing first. The spec wisely says
not to, and the reason is this:

**The whole project rests on the assumption that an off-the-shelf
"human-parsing" model can tell us, pixel by pixel, where the hair is, where
the arms are, where the legs are, where the upper-clothes are, and so on.**
If that assumption is wrong — if no public model produces labels that map
cleanly onto our modesty rules — then the project becomes either (a) a
model fine-tuning project (months of work, needs labelled data) or (b)
dead. Either way, we'd want to know on day one, not month two.

So Stage 1 is just one question: **does the segmenter give us what we
need?**

## What "segmentation" means here

For readers who haven't worked with computer vision: a segmentation model
takes an image and outputs a "label map" the same size as the image. Each
pixel gets assigned an integer category. So if you feed it a photo of a
person standing on grass, the output might be a grid where pixels covering
the person's hair are labelled `2`, pixels covering their face are `11`,
pixels covering their pants are `6`, and pixels covering grass are `0`
(background). That's it. We're going to build modesty logic on top of that
label grid.

The model we picked — `mattmdjaga/segformer_b2_clothes` — outputs 18
categories:

```
0 Background       6 Pants            12 Left-leg
1 Hat              7 Dress            13 Right-leg
2 Hair             8 Belt             14 Left-arm
3 Sunglasses       9 Left-shoe        15 Right-arm
4 Upper-clothes    10 Right-shoe      16 Bag
5 Skirt            11 Face            17 Scarf
```

Every label we need to apply the modesty rules is in there.

## Spec review — what we changed before writing any code

Before picking models, we looked at the spec critically. Five things stood
out:

1. **"FASHN-Human-Parser" was named as the primary segmentation model.**
   It isn't a real public standalone model — FASHN is a virtual try-on
   company; the parser referenced doesn't seem to ship as a downloadable
   thing. We dropped it and went with SegFormer-B2-clothes instead. SCHP
   (Self-Correction Human Parsing) is the documented fallback if SegFormer
   underperforms.

2. **The spec didn't name a gender classifier.** This is actually the
   pipeline's weakest link — misclassify a woman as male and her exposed
   hair won't trigger the blur. We picked **InsightFace `buffalo_l`** for
   later (face-based, fast on Apple Silicon via ONNX). When the classifier
   isn't confident, the project policy is to default to the **female
   ruleset**, because the female rules are stricter and we'd rather
   over-blur than under-blur. (More on that bias below.)

3. **The spec said the model weights would live in a `models/` directory.**
   We dropped that — every library involved (HuggingFace `transformers`,
   Ultralytics, InsightFace) has its own download cache and manages it
   well. Reinventing that wheel just adds work.

4. **"Reasonable offline timeframe" had no number.** Still doesn't. We'll
   pin it down once we have real measurements on the M4.

5. **"Anything beyond the eye area" is a vague rule input.** The
   segmentation model doesn't have an "eye area" label; in practice this
   becomes "any non-Face exposed skin in the head region." We'll firm this
   up in Stage 5.

## How we picked the segmentation model

The shortlist:

| Candidate | Why considered | Verdict |
|---|---|---|
| FASHN-Human-Parser | named in spec | not a real public model |
| **SegFormer-B2-clothes** | HuggingFace-hosted, ATR-trained, 18 fine-grained labels | **picked** |
| SCHP | well-established, LIP/ATR-trained, 20 labels | fallback if SegFormer underperforms |
| DeepLabV3+ human | reliable but coarser labels | dropped (labels too coarse) |

SegFormer-B2-clothes won because: every label we need is in its label set,
it loads in two lines via the `transformers` library, the published author
has it set up for clothing/fashion segmentation specifically (which is
exactly our domain), and it runs on the M4's MPS backend.

## Test image curation — where the workflow had to learn

We needed a small set of images that would let us check the model's
output. Six categories, derived directly from the modesty rules:

- A modestly dressed woman (hijab, long sleeves) — should produce
  minimal Hair, no Arms, no Legs.
- A modestly dressed man (suit, full-length pants) — should produce
  Upper-clothes and Pants, no exposed Arms (above the wrists), no Legs.
- A woman with uncovered hair and bare arms (the "female trigger" case).
- A shirtless man (the "male torso trigger" case).
- A man wearing shorts (the "male thigh trigger" case).
- A second modest woman variant for negative case variety.

This is where the workflow had to adjust. The first attempt used Unsplash
search result titles as ground truth — i.e. if a search returned a photo
captioned "Woman in brown shirt and white shorts", we trusted the caption.
**That was wrong.** The actual photo at that URL was a young man with
shoulder-length hair holding a backpack by a lake. The Unsplash titles are
auto-generated and approximate; they cannot be trusted blind.

Another image — captioned as a single man jogging — turned out to be a
fitness class with multiple people of mixed dress styles in frame. Useless
as a single-subject test.

The corrected workflow:

1. Search for candidate photos.
2. **Fetch each Unsplash photo page** and have the page describe what it
   actually contains: how many people, what gender, what they're wearing,
   any other people / signs / overlays in frame.
3. Only commit to a candidate if the description matches.
4. Download the image.
5. **Visually verify the image** before declaring it ready.
6. Re-run if anything doesn't match.

Steps 4–5 are non-skippable. Even after step 2, one of the verified
candidates ("man on track field, single subject") turned out to be a
runner in compression *leggings*, not shorts — so it didn't actually test
the thigh-exposure trigger we wanted. Visually inspecting the image is the
only way to be sure.

The final 7-image set ended up as:

| # | File | What it tests |
|---|---|---|
| 01 | modest_woman_hijab | negative case — light hijab, modest dress |
| 02 | modest_man_suit | negative case — full suit, turtleneck |
| 03 | immodest_woman_dress (jumpsuit) | uncovered hair + bare arms |
| 04 | immodest_shirtless_man | bare torso |
| 05 | immodest_man_shorts | covered torso, exposed thighs |
| 06 | modest_woman_abaya | second negative case — dark hijab |
| 07 | immodest_woman_longsleeve | uncovered hair only (arms + legs covered) |

Image 06 is intentionally a *dark* hijab as a contrast to image 01's light
one. That choice ended up uncovering the most important finding of the
stage. (More below.)

Image 07 isolates the hair-only trigger: she's wearing a long-sleeve top
and jeans, so the only modesty violation is her uncovered hair. It tests
whether the rule logic can fire on hair alone.

## Running the segmenter

We wrote `scripts/test_segmenter.py` — about 100 lines. It loads the
SegFormer model, runs it on each image, prints the percentage of pixels
the model assigned to each label, and saves a side-by-side visualization
(original photo + colored segmentation map with a legend) for each image.
Per-label coverage marked with `*` if it's a label the modesty rules care
about.

It runs on the M4 in a few seconds per image using PyTorch's MPS backend.
Model weights download once on first run (~150 MB) and live in the
HuggingFace cache.

## What the model saw

Per-image label coverage from the actual run:

| # | Subject | Hair % | Arms % (L+R) | Legs % (L+R) | Upper % | Notes |
|---|---|---:|---:|---:|---:|---|
| 01 | hijab woman (light) | 0.48 | 0.00 | 0.00 | 18.57 | Hat 0.37%, Scarf 0.35% |
| 02 | suit man | 0.39 | 0.57 | 0.00 | 9.39 | arms = wrists/hands only |
| 03 | sleeveless jumpsuit | 0.49 | 2.38 | 0.08 | 1.08 | Pants 11.27% (jumpsuit legs) |
| 04 | shirtless man | 0.36 | 2.82 | 3.83 | **0.00** | clean "no shirt" signal |
| 05 | t-shirt + shorts | 0.37 | 0.65 | 0.20 | 4.50 | subject is small in frame |
| 06 | dark hijab abaya | **4.15** ⚠️ | 1.57 | 0.00 | 10.41 | Hat only 0.37% |
| 07 | uncovered hair | 2.92 | 0.24 | 0.00 | 4.09 | Hat 1.03% (held in hand) |

Each row tells us something about whether the segmenter's output matches
what a human would say is going on in the photo.

## Findings

### Finding 1 — Image-relative coverage % is misleading

Image 05 (man in shorts) shows bare legs covering only **0.20%** of the
total image. That sounds tiny. But the *person* in that photo only
occupies about 15% of the frame to begin with — the rest is sky, ground,
buildings, lamppost. If you cropped the image down to just the person, the
bare legs would be more like 1–2% of *that* crop, which is a perfectly
detectable signal.

**Implication:** the spec's plan is right — we have to run YOLO first to
get a tight bounding box around each person, then run the segmenter on the
cropped person, and compute coverage percentages relative to the *bounding
box*, not relative to the whole image. Setting a single global threshold
on full-frame percentages would either miss small-but-real triggers, or
flag pixels of background grass as "leg." YOLO is a non-negotiable
preprocessing step, not an optimization.

### Finding 2 — Dark hijab gets partly mislabelled as Hair

This is the most important finding of the stage and the one that changes
the rule design.

Image 06 shows a woman in a black hijab and a long-sleeve dress. Her hair
is fully covered. The segmenter said her image contained **4.15% Hair**
pixels. Compare image 01 (light gray hijab, also fully covered hair) at
0.48% Hair, or image 07 (uncovered hair, fully visible) at 2.92% Hair.

By naive coverage percentage, the model thinks the *covered* head has more
hair on it than the *uncovered* head. That can't be right.

We looked at the visualization to see what was happening. The model didn't
get the whole hijab wrong — it correctly understood most of the
headcovering, including the long tail at the back, as headcovering-style
labels. The error was localized: a discrete patch on one side of the head
got labelled Hair while the rest was labelled correctly. Visually it's
like the model saw the dark fabric and thought "this is hair" for one
specific region.

**Why it matters:** the simplest rule we could write — "if Hair coverage >
some threshold, blur the woman" — would over-blur every woman in a dark
headcovering. That's a serious false-positive class.

**Three candidate fixes** (all live in the rule layer, not the model):

1. **Composite head-covering signal.** Combine Hat and Scarf labels. If
   they're present in the head region and Hair appears as a smaller patch
   surrounded by them, treat the head as covered.
2. **Connected-component analysis.** Real uncovered hair forms one large
   contiguous blob extending from the top of the head down the sides or
   back. A misclassified patch is small and isolated. Filter out small
   isolated Hair regions.
3. **Spatial constraint via the head region.** Use the YOLO bbox plus the
   Face label location to define a rectangular "head region." Apply rules
   only within that region.

The production rule will probably combine 1 and 2.

### Finding 3 — "Upper-clothes = 0%" is a clean shirtless signal

Image 04 (shirtless man at the beach) had **exactly 0.00%** Upper-clothes.
That's a strong, reliable, almost-binary signal. For male modesty rules,
shirtlessness is the primary trigger (because it automatically exposes the
navel). We can lean on this.

This also tells us something nice: SegFormer's "Upper-clothes" label
genuinely tracks "is the torso covered by a garment." It doesn't get
confused by skin tone, lighting, or pose — at least not in this small
test.

### Finding 4 — The Hat label fires on hand-held hats too

Image 07 had 1.03% of the image labelled as Hat. The woman in the photo
*is* holding a hat — but in her hand, not on her head. So if we wrote the
rule "Hat present → head is covered," we'd falsely suppress the blur on a
woman with bare hair who happens to be holding a hat at her hip.

Same fix as the hijab false positive: spatial constraint. Hat only counts
as headcovering if Hat pixels are inside the head region (top of bbox,
near the Face label).

## Rule design discussion

This is where most of the back-and-forth happened. Two scenarios shaped
the rule design.

### The hijab patch problem

Discussed above. The model assigns a localized Hair patch on a fully
covered head. We need to suppress that without making the rule lax in
general.

### The neckscarf edge case

A different scenario was raised: imagine a woman with **fully uncovered
hair** who is wearing a stylish fashion scarf around her **neck** — looped
at the throat with tails draped down toward her torso. So:

- Hair: clearly visible at the top of the head.
- Scarf label: present, but at the neck region, not the head.

This is a torpedo to a naive rule. If we wrote "Hat OR Scarf present →
headcovering present → suppress Hair trigger," we would suppress the blur
on this clearly-uncovered woman because the model dutifully reported a
Scarf somewhere on her body.

This is exactly why the rule has to be **spatial**. Specifically:

> Hat or Scarf only count as headcovering if their pixels are inside the
> head region — defined as the top portion of the person's YOLO bounding
> box, bounded below by the bottom of the Face label.

Under that rule:

- Image 06 (dark hijab): the Hat/Scarf pixels are inside the head region
  alongside a misclassified Hair patch. Connected-component + spatial
  reasoning suppresses the Hair → no blur. ✅
- Neckscarf scenario: the Scarf pixels are *below* the Face label, so they
  don't count as headcovering. The Hair pixels at the top of the head are
  in the head region with no covering nearby → blur. ✅
- Image 07 (hat held in hand): the Hat pixels are at hip level, far below
  the Face. They don't count as headcovering. Hair is uncovered → blur. ✅

The general principle: **never reason about what labels are present on a
person; reason about where they are relative to anatomy.** Anatomy comes
from the YOLO bbox plus the Face label.

### Edge cases that remain genuinely hard

We're being honest with ourselves: even with spatial rules, some cases
will be hard to call.

- **Bandanas / partial coverings**: hair partially visible at the back,
  fabric covering the crown. Both Hair and Hat fire in the head region.
- **Hoods worn loosely**: hair sometimes visible above the forehead.
- **Caps over ponytails**: Hat on top, Hair label at the back of the head.

We're not going to get these perfect from one segmentation model. Two
options for later: (a) train a small classifier on top of the segmenter's
output specifically for "head coverage", or (b) accept that these are
uncertain cases and apply the policy below.

### The bias toward over-blur

Stated as project policy:

> When the situation is ambiguous, blur.

The reasoning is asymmetric cost. OCCLUDE exists so the user can watch
educational/documentary content without seeing immodest people. A **false
positive** (blurring a modestly dressed person) is a minor visual
annoyance — the audio still plays, the content is still consumable. A
**false negative** (failing to blur an immodest person) defeats the entire
purpose of running the tool. Therefore the rule logic should err
deliberately on the side of over-blurring whenever the signals don't
clearly say "modest."

In practice that means thresholds like:

- Hair triggers blur if Hair pixel count in the head region exceeds a
  small threshold AND Hat/Scarf are not clearly dominant (e.g. Hat+Scarf
  area > 3× Hair area).
- Anything in between → blur.

This bias also covers the gender classifier: if InsightFace can't decide
or returns low confidence, default to the female ruleset (stricter).

## What changed about the spec

| Spec said | We changed it to | Why |
|---|---|---|
| FASHN-Human-Parser primary | SegFormer-B2-clothes primary, SCHP fallback | FASHN-HP isn't a real public model |
| `models/` directory for weights | Use library default caches | Each library handles this well already |
| (no gender classifier named) | InsightFace `buffalo_l`, default to female on low confidence | Spec gap; this is the riskiest pipeline stage |
| (no "what does ambiguous mean") | Documented bias-toward-over-blur policy | Explicit policy is better than implicit |
| Modesty rules as "any visible X → blur" | Same intent, but rules must be **spatial** within the head region | Pixel-percentage thresholds alone produce false positives on dark hijabs and false negatives on neckscarves |

## What remains open

This stage validated the model. It did not build any pipeline. Still to do:

- Stage 2: YOLOv8 person detection + crop the segmenter input to the
  bounding box. This will dramatically reduce the noise we saw with image
  05's "0.20% legs."
- Stage 3: gender classifier (InsightFace), with low-confidence fallback.
- Stage 4 + 5: rule logic — implement the spatial reasoning we worked out
  above, including head-region definition, connected-component filtering
  on Hair, and the Hat/Scarf vs Hair tiebreaker.
- Stage 6–8: blur, temporal smoothing, video reconstruction.
- Open question: M4 inference time per frame at 1080p. Until we measure it
  we don't know whether CoreML conversion is needed.

## Files this stage produced

```
OCCLUDE/
├── requirements.txt
├── scripts/
│   └── test_segmenter.py
├── test_images/
│   ├── 01_modest_woman_hijab.jpg
│   ├── 02_modest_man_suit.jpg
│   ├── 03_immodest_woman_dress.jpg
│   ├── 04_immodest_shirtless_man.jpg
│   ├── 05_immodest_man_shorts.jpg
│   ├── 06_modest_woman_abaya.jpg
│   └── 07_immodest_woman_longsleeve.jpg
├── test_output/                  (segmentation visualizations)
└── docs/
    └── 01-segmenter-validation.md   (this file)
```

## How to reproduce

```bash
uv venv
uv pip install -r requirements.txt
.venv/bin/python scripts/test_segmenter.py
```

The script will print per-label coverage to the terminal and write
`test_output/<filename>_seg.png` for each input image. Outputs are
deterministic for a given input — same image in, same labels out — so any
future regression will be obvious from a re-run.
