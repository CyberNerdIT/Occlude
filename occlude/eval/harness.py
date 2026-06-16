"""Run the rule layer over labelled images and report accuracy + coverage.

The heavy model stack is injected (``perceiver``), so the pure logic
here — reason categorisation, the confusion matrix, and the coverage
map — is unit-testable with a fake perceiver and never has to load
weights. ``scripts/eval_accuracy.py`` wires in the real
:class:`occlude.pipeline.perception.Perception`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from occlude.pipeline.perception import Person
from occlude.pipeline.rules import Decision, RuleEngine

# Every rule branch the engine can land on, mapped to a human label.
# The coverage map is computed against this set: a branch with no
# labelled positive example is reported as UNVALIDATED.
BRANCHES: dict[str, str] = {
    "shirtless": "male: shirtless fast-path (navel exposed)",
    "male_bare_thigh": "male: bare thigh in the hip-knee band",
    "male_covered": "male: modestly covered (no blur)",
    "female_uncovered_hair": "female: uncovered hair in the head region",
    "female_hair_below_face": "female: hair flowing past the face line",
    "female_bare_arms": "female: bare arms",
    "female_bare_legs": "female: bare legs",
    "female_covered": "female: modestly covered (no blur)",
    "no_face_overblur": "no face anchor -> over-blur bias",
    "child_exempt": "child exemption (never blur)",
}

# Branches that represent a positive (blur=True) decision. Used to score
# coverage: only positive branches need a labelled positive example.
_POSITIVE_BRANCHES = {
    "shirtless",
    "male_bare_thigh",
    "female_uncovered_hair",
    "female_hair_below_face",
    "female_bare_arms",
    "female_bare_legs",
    "no_face_overblur",
}


def categorize_reason(reason: str) -> str:
    """Map a :class:`Decision.reason` string to a stable branch key.

    The rule engine emits free-text reasons with stable leading phrases;
    this collapses them to the :data:`BRANCHES` keys so the harness can
    check *which* branch fired, not just the blur bool.
    """
    r = reason.lower()
    if r.startswith("child"):
        return "child_exempt"
    if r.startswith("no face"):
        return "no_face_overblur"
    if r.startswith("shirtless"):
        return "shirtless"
    if r.startswith("bare thigh"):
        return "male_bare_thigh"
    if r.startswith("male: covered"):
        return "male_covered"
    if r.startswith("uncovered hair"):
        return "female_uncovered_hair"
    if r.startswith("hair below face"):
        return "female_hair_below_face"
    if r.startswith("bare arms"):
        return "female_bare_arms"
    if r.startswith("bare legs"):
        return "female_bare_legs"
    if r.startswith("female: covered"):
        return "female_covered"
    return "unknown"


@dataclass
class Case:
    image: str
    expect_blur: bool
    expect_branch: list[str]
    note: str = ""


@dataclass
class CaseResult:
    image: str
    expect_blur: bool
    expect_branch: list[str]
    detected: bool                 # did the detector find any person?
    actual_blur: bool
    actual_branch: str
    blur_correct: bool
    branch_correct: bool
    note: str = ""


@dataclass
class Report:
    results: list[CaseResult] = field(default_factory=list)
    # Confusion matrix on the binary blur decision.
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    # branch key -> number of labelled positive examples exercising it.
    coverage: dict[str, int] = field(default_factory=dict)

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def n_blur_correct(self) -> int:
        return sum(1 for r in self.results if r.blur_correct)

    @property
    def unvalidated_branches(self) -> list[str]:
        """Positive branches with zero labelled positive examples."""
        return [
            b for b in _POSITIVE_BRANCHES if self.coverage.get(b, 0) == 0
        ]


def load_cases(labels_path: Path) -> list[Case]:
    data = json.loads(Path(labels_path).read_text())
    cases: list[Case] = []
    for c in data["cases"]:
        branch = c.get("expect_branch", [])
        if isinstance(branch, str):
            branch = [branch]
        cases.append(
            Case(
                image=c["image"],
                expect_blur=bool(c["expect_blur"]),
                expect_branch=list(branch),
                note=c.get("note", ""),
            )
        )
    return cases


def _area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _subject(people: list[Person]) -> Person | None:
    """The test images are single-subject portraits; the subject is the
    largest-area detection."""
    if not people:
        return None
    return max(people, key=lambda p: _area(p.bbox))


def _decide_subject(perceiver, rules: RuleEngine, image: Image.Image) -> tuple[bool, Decision | None]:
    """Run detect -> classify -> decide on the primary subject.

    Returns ``(detected, decision)``. ``decision`` is None when no
    person was detected (a detection miss).
    """
    people = perceiver.detect_and_segment(image)
    subject = _subject(people)
    if subject is None:
        return False, None
    subject.gender, subject.face_det_score, subject.age = perceiver.classify(
        subject.crop
    )
    return True, rules.decide(subject)


def evaluate(
    cases: list[Case],
    perceiver,
    image_dir: Path,
    rules: RuleEngine | None = None,
) -> Report:
    """Score *cases* end to end and build the confusion + coverage report."""
    rules = rules or RuleEngine()
    report = Report()

    for case in cases:
        img_path = Path(image_dir) / case.image
        image = Image.open(img_path).convert("RGB")
        detected, decision = _decide_subject(perceiver, rules, image)

        if decision is None:
            # No detection -> no blur applied. Scored as actual_blur=False.
            actual_blur = False
            actual_branch = "no_detection"
        else:
            actual_blur = decision.blur
            actual_branch = categorize_reason(decision.reason)

        blur_correct = actual_blur == case.expect_blur
        branch_correct = (
            actual_branch in case.expect_branch
            if case.expect_branch
            else blur_correct
        )

        report.results.append(
            CaseResult(
                image=case.image,
                expect_blur=case.expect_blur,
                expect_branch=case.expect_branch,
                detected=detected,
                actual_blur=actual_blur,
                actual_branch=actual_branch,
                blur_correct=blur_correct,
                branch_correct=branch_correct,
                note=case.note,
            )
        )

        if case.expect_blur and actual_blur:
            report.tp += 1
        elif not case.expect_blur and not actual_blur:
            report.tn += 1
        elif not case.expect_blur and actual_blur:
            report.fp += 1
        else:
            report.fn += 1

    report.coverage = _coverage_from_cases(cases)
    return report


def _coverage_from_cases(cases: list[Case]) -> dict[str, int]:
    """Count labelled positive examples per positive branch."""
    coverage = {b: 0 for b in _POSITIVE_BRANCHES}
    for case in cases:
        if not case.expect_blur:
            continue
        for branch in case.expect_branch:
            if branch in coverage:
                coverage[branch] += 1
    return coverage


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def format_report(report: Report) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("OCCLUDE rule-layer evaluation")
    lines.append("=" * 72)
    lines.append(
        "NOTE: the bundled label set is the SAME images the thresholds were\n"
        "tuned on. Numbers below are TRAINING-SET / regression performance,\n"
        "NOT held-out accuracy. Held-out labelled data is still the open gap."
    )
    lines.append("")

    # Per-case table.
    lines.append(
        f"{'image':<34}{'expect':<8}{'actual':<8}{'branch':<24}{'ok'}"
    )
    lines.append("-" * 72)
    for r in report.results:
        exp = "BLUR" if r.expect_blur else "keep"
        act = "BLUR" if r.actual_blur else "keep"
        ok = "ok" if (r.blur_correct and r.branch_correct) else "FAIL"
        miss = "" if r.detected else " (no-detect)"
        lines.append(
            f"{r.image:<34}{exp:<8}{act:<8}{r.actual_branch + miss:<24}{ok}"
        )
    lines.append("")

    # Confusion + metrics.
    lines.append(
        f"blur decision: {report.n_blur_correct}/{report.n} correct  "
        f"(TP={report.tp} TN={report.tn} FP={report.fp} FN={report.fn})"
    )
    lines.append(
        f"precision={_fmt_pct(report.precision)}  "
        f"recall={_fmt_pct(report.recall)}"
    )
    lines.append("")

    # Coverage map — the honest core.
    lines.append("Branch coverage (labelled positive examples per branch):")
    for branch in sorted(_POSITIVE_BRANCHES):
        count = report.coverage.get(branch, 0)
        flag = "  <-- UNVALIDATED (placeholder threshold)" if count == 0 else ""
        lines.append(f"  {branch:<26}{count} example(s){flag}")
    unval = report.unvalidated_branches
    if unval:
        lines.append("")
        lines.append(
            "WARNING: the following positive branches fire only on\n"
            "placeholder thresholds with NO labelled example constraining\n"
            f"them: {', '.join(sorted(unval))}.\n"
            "Their thresholds in rules.py are tuned to the segmenter noise\n"
            "floor, not to real positives. Add labelled examples before\n"
            "trusting these decisions."
        )
    lines.append("=" * 72)
    return "\n".join(lines)
