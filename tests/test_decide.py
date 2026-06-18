"""Verdict aggregation policy — the WHY of each over-blur decision.

These tests pin the policy priority (non-human > child > over-blur on tie)
so a future change to the aggregation can't silently flip the project's
core asymmetry: never miss a blur, tolerate an extra one.
"""
from occlude.pipeline.decide import aggregate
from occlude.pipeline.tracklets import (
    AGE_ADULT,
    AGE_CHILD,
    SEX_FEMALE,
    SEX_MALE,
    Verdict,
)


def _s(blur, *, is_human=True, sex=None, age=AGE_ADULT):
    return Verdict(blur=blur, is_human=is_human, sex=sex, age_bracket=age)


def test_no_samples_defaults_to_blur():
    # An unjudged person must not silently pass — that defeats the tool.
    v = aggregate([])
    assert v.blur is True


def test_nonhuman_majority_never_blurs():
    # The CGI/dummy-character false positive: a confident non-human call
    # must suppress the blur even though "no clothes visible" would
    # otherwise trip the rules.
    v = aggregate([_s(True, is_human=False), _s(True, is_human=False), _s(True)])
    assert v.blur is False
    assert v.is_human is False


def test_child_exemption_overrides_immodesty():
    # Kids are never blurred even when flagged immodest.
    v = aggregate([_s(True, age=AGE_CHILD), _s(True, age=AGE_CHILD)])
    assert v.blur is False
    assert v.age_bracket == AGE_CHILD


def test_lone_child_vote_does_not_exempt():
    # A single mislabeled "child" frame must not let an immodest adult
    # escape — the exemption is gated on a majority.
    v = aggregate([_s(True, age=AGE_CHILD), _s(True, age=AGE_ADULT), _s(True, age=AGE_ADULT)])
    assert v.blur is True


def test_modesty_tie_blurs():
    # Over-blur bias: a 1–1 split blurs.
    v = aggregate([_s(True), _s(False)])
    assert v.blur is True


def test_clear_majority_modest_does_not_blur():
    v = aggregate([_s(False), _s(False), _s(True)])
    assert v.blur is False


def test_sex_tie_resolves_female():
    # Tie on sex picks the stricter (female) ruleset.
    v = aggregate([_s(True, sex=SEX_MALE), _s(True, sex=SEX_FEMALE)])
    assert v.sex == SEX_FEMALE
