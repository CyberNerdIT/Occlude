"""Stage 5 — Rule layer.

Maps a Stage 4 :class:`Person` to a binary blur decision per
``docs/03-rule-design.md``. Three stacked checks (Region / Shape /
Tiebreaker) on the female head rule, region check on the male thigh
rule, plus a shirtless fast-path and a one-directional gender
cross-check (``predicted M + Hair extending below the Face label →
override to F``).

Numerical thresholds were tuned against ``test_images/``. They are
labelled with the test cases that constrain them so the next person
who tunes them can see what would break. See
``docs/06-rule-implementation.md`` for the per-image walk-through.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as cc_label

from occlude.pipeline.perception import Person

# --- Tunable thresholds ---------------------------------------------------

# Child exemption: a person whose estimated age is at or below this is
# never blurred (boy or girl), per explicit user policy. InsightFace's
# buffalo_l age head is noisy (±5–10 yr, docs/04), and un-blurring is a
# safety-reducing action, so the cutoff is deliberately LOW: the test-set
# adult women came back 24–31, so 12 leaves a wide margin against the
# model under-aging an adult woman into the exempt band. Raising this
# trades that safety margin for catching older teens. age is only
# available when a face was detected — a child whose face the detector
# misses falls through to the normal (over-blur) rules; that gap is
# inherent to a face-derived age signal and is accepted, not worked
# around.
CHILD_MAX_AGE = 12

# Default to female (stricter ruleset) when the face detection is
# unreliable or absent. Stage 3 confidence range across our 7 images:
# 0.66–0.89, so 0.50 keeps every test image on its predicted gender
# and only triggers default-to-female on genuine misses.
MIN_FACE_DET_SCORE = 0.50

# Shirtless fast-path: image 04 (shirtless) reports Upper-clothes
# 0.01% of bbox; modestly-clothed males 26% (image 02), 32% (image 05).
# 0.5% comfortably separates them.
SHIRTLESS_EPS_PCT = 0.5

# Female head region: extend Face-label bbox upward by HEAD_UP_FACTOR
# face heights and outward by HEAD_SIDE_FACTOR face widths on each
# side. Capture hair / hat / scarf above the forehead and at the
# temples without grabbing arbitrary background pixels.
HEAD_UP_FACTOR = 1.0
HEAD_SIDE_FACTOR = 0.5

# Hair shape filter: largest connected component of Hair within the
# head region must exceed this fraction of the head-region area. The
# dark-hijab (image 06) misclassified-Hair patches are small and
# scattered; uncovered hair (image 07) forms a single large blob.
HAIR_MIN_CC_FRAC = 0.03

# Hair tiebreaker: treat head as covered when Hat+Scarf area in head
# region exceeds HATSCARF_HAIR_RATIO × Hair area. Image 06 (dark
# abaya, the canonical case this rule exists for) reports
# Hat+Scarf:Hair = 18388:27489 in the head region — ratio 0.67. The
# Stage 3 design preferred N ≥ 1 ("clearly dominate"), but at that
# level image 06 over-blurs, defeating the rule's purpose. 0.5 keeps
# the spirit ("substantial Hat/Scarf presence") while matching what
# the segmenter actually emits. Trade-off: a baseball cap + ponytail
# (Hat ≈ 0.5×Hair, real hair uncovered) gets under-blurred. Doc 03
# already flagged that case as a known v1 gap.
HATSCARF_HAIR_RATIO = 0.5

# Bare-arms (female, bbox-wide). Image 03 (sleeveless dress) has
# Left+Right-arm 8.10%; image 06 (abaya) has 2.86% from segmenter
# noise at the shoulder edge. 4.0 separates them. Note: this is a
# v1 simplification — the Stage 3 design's region/shape/tiebreaker
# skeleton is only fully specified for the head rule.
ARMS_MIN_PCT = 4.0

# Bare-legs (female, bbox-wide). No test image exercises this (all
# our women have full-length bottoms); 2.0 is a placeholder that
# clears the segmenter noise floor on every current test image.
LEGS_MIN_PCT = 2.0

# Male thigh region: y range below chin, in face-height units.
# 2.5–4.0 is the hip-to-knee band on a standing adult per the design.
# Below-knee skin is modest by spec; widening the region would catch
# image 05's calves but break that contract.
THIGH_TOP_FACTOR = 2.5
THIGH_BOT_FACTOR = 4.0
# Bare thigh trigger: Legs (Left+Right) coverage in the thigh region.
# No test image exercises this cleanly; 1.0% is a placeholder above
# the noise floor on the male images.
THIGH_LEGS_PCT = 1.0

# Cross-check (predicted M → F override). Compare Hair pixel count
# below the Face-label bottom against face *area*. With k=0.20 the
# override fires when there's roughly a fifth of a face-area's worth
# of hair below the face. Tuned against the bere/beanie woman in
# `laughing_people.mp4` who came back from InsightFace as M @ 0.91
# despite obvious long hair flowing past her shoulders: her ratio is
# 16275/67500 = 0.24, which clears 0.20 but missed the original 0.5.
# Image 05 (bald M) and 04 (shirtless M with short hair) both have
# essentially no hair below the face label, so they don't trip it.
CROSSCHECK_HAIR_AREA_FACTOR = 0.20

# Hair below face (female rule). The head-region Hat+Scarf:Hair
# ratio check correctly handles a fully-tucked hijab, but it misses
# the case where a hat/beanie/bere covers the *top* of the head while
# long hair flows past the face onto the shoulders. SegFormer tags
# the bere as Hat (28% of head region in the canonical case) and the
# cascade as Hair (16%) — Hat:Hair = 1.77 → "covered" by the existing
# rule, even though the spec is unambiguous: hair extending past the
# scarf/hijab line is uncovered hair → blur. Threshold is 1% of the
# crop bbox: the bere case shows 8.87%, while a fully-covered abaya
# (image 06) shows ~0% (hair confined to the head region under the
# scarf). 1% comfortably separates them.
HAIR_BELOW_FACE_PCT = 1.0


@dataclass
class Decision:
    blur: bool
    reason: str
    gender_used: str        # 'M' or 'F' after override / default
    overridden: bool        # True if cross-check or default-to-female fired
    head_region: tuple[int, int, int, int] | None = None  # (top, bot, left, right) on crop
    thigh_region: tuple[int, int, int, int] | None = None


# --- Helpers --------------------------------------------------------------

def _face_bounds(label_masks: dict[str, np.ndarray]) -> tuple[int, int, int, int] | None:
    """(top, bot, left, right) of Face label, or None if absent."""
    face_mask = label_masks["Face"]
    if not face_mask.any():
        return None
    ys, xs = np.where(face_mask)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _coverage_pct(label_masks: dict[str, np.ndarray], *label_names: str) -> float:
    total = next(iter(label_masks.values())).size
    hit = sum(int(label_masks[n].sum()) for n in label_names)
    return 100.0 * hit / total


def _hair_below_face_count(label_masks: dict[str, np.ndarray], face_bot: int) -> int:
    hair = label_masks["Hair"]
    y_idx = np.arange(hair.shape[0])[:, None]
    return int((hair & (y_idx > face_bot)).sum())


# --- Engine ---------------------------------------------------------------

class RuleEngine:
    def decide(self, person: Person) -> Decision:
        # Child exemption fires before every other rule (including the
        # no-Face over-blur bias): a detected child is never blurred.
        if person.age is not None and person.age <= CHILD_MAX_AGE:
            return Decision(
                blur=False,
                reason=f"child (age≈{person.age:.0f} ≤ {CHILD_MAX_AGE}) — exempt",
                gender_used=person.gender or "?",
                overridden=False,
            )

        gender = person.gender
        overridden = False

        # Default-to-female on low / missing face confidence.
        if gender is None or person.face_det_score < MIN_FACE_DET_SCORE:
            gender = "F"
            overridden = True

        bounds = _face_bounds(person.label_masks)

        # Cross-check: predicted M, but Hair extends below the Face → F.
        if gender == "M" and bounds is not None:
            face_top, face_bot, face_left, face_right = bounds
            face_area = (face_bot - face_top + 1) * (face_right - face_left + 1)
            below = _hair_below_face_count(person.label_masks, face_bot)
            if below > CROSSCHECK_HAIR_AREA_FACTOR * face_area:
                gender = "F"
                overridden = True

        # No face anchor → over-blur.
        if bounds is None:
            return Decision(
                blur=True,
                reason="no Face label (over-blur bias)",
                gender_used=gender,
                overridden=overridden,
            )

        if gender == "M":
            return self._male(person, bounds, overridden)
        return self._female(person, bounds, overridden)

    def _male(self, person: Person, bounds, overridden: bool) -> Decision:
        # Shirtless fast-path.
        upper_pct = _coverage_pct(person.label_masks, "Upper-clothes")
        if upper_pct < SHIRTLESS_EPS_PCT:
            return Decision(
                blur=True,
                reason=f"shirtless (Upper-clothes {upper_pct:.2f}%)",
                gender_used="M", overridden=overridden,
            )

        # Thigh region.
        face_top, face_bot, _, _ = bounds
        face_h = face_bot - face_top + 1
        H, W = person.seg_mask.shape
        t_top = max(0, face_bot + int(THIGH_TOP_FACTOR * face_h))
        t_bot = min(H, face_bot + int(THIGH_BOT_FACTOR * face_h))
        thigh_box = (t_top, t_bot, 0, W)

        if t_bot > t_top:
            left_leg = person.label_masks["Left-leg"][t_top:t_bot, :]
            right_leg = person.label_masks["Right-leg"][t_top:t_bot, :]
            legs_pct = 100.0 * int((left_leg | right_leg).sum()) / left_leg.size
            if legs_pct > THIGH_LEGS_PCT:
                return Decision(
                    blur=True,
                    reason=f"bare thigh (Legs {legs_pct:.2f}% in thigh region)",
                    gender_used="M", overridden=overridden,
                    thigh_region=thigh_box,
                )

        return Decision(
            blur=False,
            reason="male: covered",
            gender_used="M", overridden=overridden,
            thigh_region=thigh_box,
        )

    def _female(self, person: Person, bounds, overridden: bool) -> Decision:
        face_top, face_bot, face_left, face_right = bounds
        face_h = face_bot - face_top + 1
        face_w = face_right - face_left + 1
        H, W = person.seg_mask.shape

        h_top = max(0, face_top - int(HEAD_UP_FACTOR * face_h))
        h_bot = face_bot
        h_left = max(0, face_left - int(HEAD_SIDE_FACTOR * face_w))
        h_right = min(W, face_right + int(HEAD_SIDE_FACTOR * face_w))
        head_box = (h_top, h_bot, h_left, h_right)
        head_area = max((h_bot - h_top) * (h_right - h_left), 1)

        # Hair: Region + Shape + Tiebreaker.
        hair_mask = person.label_masks["Hair"][h_top:h_bot, h_left:h_right]
        if hair_mask.any():
            labelled, n = cc_label(hair_mask)
            if n > 0:
                largest_cc = int(np.bincount(labelled.ravel())[1:].max())
                largest_frac = largest_cc / head_area
                if largest_frac >= HAIR_MIN_CC_FRAC:
                    hair_area = int(hair_mask.sum())
                    hatscarf_area = int(
                        (person.label_masks["Hat"][h_top:h_bot, h_left:h_right]
                         | person.label_masks["Scarf"][h_top:h_bot, h_left:h_right]).sum()
                    )
                    if hatscarf_area <= HATSCARF_HAIR_RATIO * hair_area:
                        return Decision(
                            blur=True,
                            reason=(f"uncovered hair (CC {largest_frac:.3f} of head, "
                                    f"Hat+Scarf:Hair {hatscarf_area}:{hair_area})"),
                            gender_used="F", overridden=overridden,
                            head_region=head_box,
                        )

        # Hair-below-face: separate from the head-region check above
        # so it fires even when a hat/beanie/bere "covers" the head
        # but visible hair flows past the face line.
        below_count = _hair_below_face_count(person.label_masks, face_bot)
        below_pct = 100.0 * below_count / max(person.seg_mask.size, 1)
        if below_pct >= HAIR_BELOW_FACE_PCT:
            return Decision(
                blur=True,
                reason=f"hair below face ({below_pct:.2f}% of bbox)",
                gender_used="F", overridden=overridden,
                head_region=head_box,
            )

        # Arms.
        arms_pct = _coverage_pct(person.label_masks, "Left-arm", "Right-arm")
        if arms_pct > ARMS_MIN_PCT:
            return Decision(
                blur=True,
                reason=f"bare arms ({arms_pct:.2f}%)",
                gender_used="F", overridden=overridden,
                head_region=head_box,
            )

        # Legs.
        legs_pct = _coverage_pct(person.label_masks, "Left-leg", "Right-leg")
        if legs_pct > LEGS_MIN_PCT:
            return Decision(
                blur=True,
                reason=f"bare legs ({legs_pct:.2f}%)",
                gender_used="F", overridden=overridden,
                head_region=head_box,
            )

        return Decision(
            blur=False,
            reason="female: covered",
            gender_used="F", overridden=overridden,
            head_region=head_box,
        )
