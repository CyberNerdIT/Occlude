"""Pass 2 policy: turn raw VLM samples into one Verdict per tracklet.

The VLM (judge.py) returns a structured judgment for each sampled frame of a
tracklet. This module is the *policy* layer over those raw answers, kept
deliberately model-free so the rules are auditable and testable without a
GPU. Three jobs:

  1. Aggregate several samples of one person into a single decision, since
     different frames of the same tracklet can disagree and the render pass
     commits to one verdict for the whole track.
  2. Apply the child exemption — under-13s are never blurred.
  3. Carry the project's bias toward over-blur into the uncertain cases:
     missing a blur defeats the tool; an extra blur is a minor annoyance.

On the geometric child backstop (size-based) that the research suggested:
rejected for v1. A small bounding box means "far from camera" at least as
often as "child", so a size rule would wrongly exempt distant adults — the
one error class the over-blur policy exists to avoid. We trust the VLM's
age bracket instead, which is materially better than the old InsightFace age
head, and keep the exemption gated on a confident child call (below).
"""
from __future__ import annotations

from collections import Counter

from occlude.pipeline.tracklets import (
    AGE_ADULT,
    AGE_CHILD,
    AGE_UNKNOWN,
    SEX_FEMALE,
    SEX_MALE,
    Verdict,
)

# A "child" exemption only fires when this fraction of usable samples agree
# the person is a child. Gating it (rather than honoring a single child vote)
# stops a lone mislabeled frame from letting an immodestly-dressed adult
# escape blur — escaping is the worse error under the over-blur policy.
CHILD_EXEMPT_MIN_FRACTION = 0.5

# When the model confidently and in the majority says a figure is not a real
# human, we honor it (cartoons, CGI avatars, mannequins → no blur). Ties and
# uncertainty fall through to "treat as human", i.e. keep the blur in play.
NONHUMAN_MIN_FRACTION = 0.5


def aggregate(samples: list[Verdict]) -> Verdict:
    """Combine per-frame Verdicts for one tracklet into a single decision.

    The aggregation order encodes the policy priority:
      non-human  >  child exemption  >  modesty trigger (over-blur on tie).
    """
    usable = [s for s in samples if s is not None]
    if not usable:
        # No judgment at all (every sample failed/parsed empty). Over-blur
        # bias: blur rather than silently pass an unjudged person.
        return Verdict(
            blur=True,
            is_human=True,
            reason="no judgment available (over-blur default)",
            confidence=0.0,
        )

    n = len(usable)

    # --- non-human gate -------------------------------------------------
    nonhuman = sum(1 for s in usable if not s.is_human)
    if nonhuman / n >= NONHUMAN_MIN_FRACTION:
        return Verdict(
            blur=False,
            is_human=False,
            reason=f"non-human figure ({nonhuman}/{n} samples)",
            confidence=nonhuman / n,
        )

    # From here on the tracklet is treated as a real person.
    human = [s for s in usable if s.is_human] or usable

    # --- sex (F on tie: stricter ruleset, matches over-blur bias) -------
    sex = _majority_sex(human)

    # --- age ------------------------------------------------------------
    child_votes = sum(1 for s in human if s.age_bracket == AGE_CHILD)
    is_child = child_votes / len(human) >= CHILD_EXEMPT_MIN_FRACTION
    age = AGE_CHILD if is_child else _majority_age(human)

    if is_child:
        return Verdict(
            blur=False,
            is_human=True,
            sex=sex,
            age_bracket=AGE_CHILD,
            reason=f"child exemption ({child_votes}/{len(human)} samples)",
            confidence=child_votes / len(human),
        )

    # --- modesty trigger (over-blur on tie) -----------------------------
    blur_votes = sum(1 for s in human if s.blur)
    blur = blur_votes >= (len(human) - blur_votes)  # ties -> blur
    confidence = (blur_votes if blur else len(human) - blur_votes) / len(human)
    return Verdict(
        blur=blur,
        is_human=True,
        sex=sex,
        age_bracket=age,
        reason=f"modesty {'blur' if blur else 'clear'} ({blur_votes}/{len(human)} samples)",
        confidence=confidence,
    )


def _majority_sex(samples: list[Verdict]) -> str | None:
    counts = Counter(s.sex for s in samples if s.sex in (SEX_MALE, SEX_FEMALE))
    if not counts:
        return None
    f = counts.get(SEX_FEMALE, 0)
    m = counts.get(SEX_MALE, 0)
    if f >= m:                # tie -> female (stricter ruleset)
        return SEX_FEMALE
    return SEX_MALE


def _majority_age(samples: list[Verdict]) -> str:
    counts = Counter(
        s.age_bracket for s in samples if s.age_bracket in (AGE_CHILD, AGE_ADULT)
    )
    if not counts:
        return AGE_UNKNOWN
    return counts.most_common(1)[0][0]
