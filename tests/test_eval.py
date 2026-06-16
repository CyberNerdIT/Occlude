"""Logic tests for the eval harness — no model weights, no real images.

A scripted perceiver + scripted rule engine drive evaluate() so the
confusion-matrix math, reason categorisation, and coverage map are
verified deterministically. The real-model run lives in
scripts/eval_accuracy.py and is invoked deliberately, not in CI.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from occlude.eval.harness import (
    Case,
    _coverage_from_cases,
    categorize_reason,
    evaluate,
    format_report,
    load_cases,
)
from occlude.pipeline.perception import Person
from occlude.pipeline.rules import Decision


def _person(bbox=(0, 0, 10, 10)) -> Person:
    return Person(
        bbox=bbox,
        det_conf=0.9,
        crop=Image.new("RGB", (10, 10)),
        seg_mask=np.zeros((10, 10), dtype=np.uint8),
        gender=None,
        face_det_score=0.0,
        label_masks={},
    )


class _ScriptedPerceiver:
    """Returns scripted detections per case, in order."""

    def __init__(self, detections: list[list[Person]]) -> None:
        self._it = iter(detections)

    def detect_and_segment(self, image):
        return next(self._it)

    def classify(self, crop):
        return "F", 0.9, None


class _ScriptedRules:
    """Returns scripted Decisions, consumed only when a subject exists."""

    def __init__(self, decisions: list[Decision]) -> None:
        self._it = iter(decisions)

    def decide(self, person):
        return next(self._it)


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("child (age≈8 ≤ 12) — exempt", "child_exempt"),
        ("no Face label (over-blur bias)", "no_face_overblur"),
        ("shirtless (Upper-clothes 0.01%)", "shirtless"),
        ("bare thigh (Legs 3.0% in thigh region)", "male_bare_thigh"),
        ("male: covered", "male_covered"),
        ("uncovered hair (CC 0.10 of head, ...)", "female_uncovered_hair"),
        ("hair below face (2.30% of bbox)", "female_hair_below_face"),
        ("bare arms (8.10%)", "female_bare_arms"),
        ("bare legs (5.00%)", "female_bare_legs"),
        ("female: covered", "female_covered"),
    ],
)
def test_categorize_reason_maps_every_branch(reason, expected):
    assert categorize_reason(reason) == expected


def _img(tmp_path: Path, name: str) -> None:
    Image.new("RGB", (10, 10)).save(tmp_path / name)


def test_evaluate_confusion_matrix(tmp_path):
    # 4 cases, one of each confusion cell.
    cases = [
        Case("tp.jpg", expect_blur=True, expect_branch=["shirtless"]),
        Case("tn.jpg", expect_blur=False, expect_branch=["female_covered"]),
        Case("fn.jpg", expect_blur=True, expect_branch=["female_bare_arms"]),
        Case("fp.jpg", expect_blur=False, expect_branch=["female_covered"]),
    ]
    for c in cases:
        _img(tmp_path, c.image)

    detections = [
        [_person()],   # tp: detected
        [_person()],   # tn: detected
        [],            # fn: NO detection -> missed blur
        [_person()],   # fp: detected
    ]
    decisions = [
        Decision(blur=True, reason="shirtless (x)", gender_used="M", overridden=False),
        Decision(blur=False, reason="female: covered", gender_used="F", overridden=False),
        # fn has no detection, so no decision consumed for it
        Decision(blur=True, reason="bare arms (9%)", gender_used="F", overridden=False),
    ]

    report = evaluate(
        cases,
        _ScriptedPerceiver(detections),
        tmp_path,
        rules=_ScriptedRules(decisions),
    )

    assert (report.tp, report.tn, report.fp, report.fn) == (1, 1, 1, 1)
    assert report.precision == pytest.approx(0.5)
    assert report.recall == pytest.approx(0.5)
    # The fn case must be recorded as a detection miss.
    fn = next(r for r in report.results if r.image == "fn.jpg")
    assert fn.detected is False
    assert fn.actual_branch == "no_detection"


def test_coverage_flags_branch_with_no_positive_example():
    cases = [
        Case("a.jpg", expect_blur=True, expect_branch=["shirtless"]),
        Case("b.jpg", expect_blur=True, expect_branch=["male_bare_thigh"]),
        Case("c.jpg", expect_blur=False, expect_branch=["female_covered"]),
    ]
    coverage = _coverage_from_cases(cases)
    assert coverage["shirtless"] == 1
    assert coverage["male_bare_thigh"] == 1
    # No labelled positive for bare legs -> placeholder, must read as 0.
    assert coverage["female_bare_legs"] == 0


def test_bundled_labels_have_the_known_coverage_gap():
    """The shipped label set must expose female_bare_legs as unvalidated.

    This pins the honesty claim: rules.py admits the female bare-legs
    threshold is a placeholder, and the eval set has no example for it.
    If someone later adds one, this test should be updated, not deleted.
    """
    labels = Path(__file__).resolve().parent.parent / "occlude" / "eval" / "labels.json"
    cases = load_cases(labels)
    assert len(cases) == 7
    coverage = _coverage_from_cases(cases)
    assert coverage["female_bare_legs"] == 0
    assert coverage["male_bare_thigh"] >= 1
    assert coverage["shirtless"] >= 1


def test_format_report_warns_about_unvalidated_branches(tmp_path):
    cases = [Case("a.jpg", expect_blur=True, expect_branch=["shirtless"])]
    _img(tmp_path, "a.jpg")
    report = evaluate(
        cases,
        _ScriptedPerceiver([[_person()]]),
        tmp_path,
        rules=_ScriptedRules(
            [Decision(blur=True, reason="shirtless (x)", gender_used="M", overridden=False)]
        ),
    )
    text = format_report(report)
    assert "UNVALIDATED" in text
    assert "female_bare_legs" in text
    assert "TRAINING-SET" in text  # the honesty caveat is always printed
