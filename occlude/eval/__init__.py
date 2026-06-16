"""Accuracy / coverage evaluation for the OCCLUDE rule layer.

This package answers two distinct questions that the unit tests in
``tests/`` do not:

1. **End-to-end regression**: run the *real* perception stack (YOLO +
   InsightFace + SegFormer) plus the rule layer over a labelled image
   set and check the binary blur decision against ground truth. The
   rule unit tests feed synthetic ``Person`` objects; this exercises
   the models that actually produce them.

2. **Coverage**: which rule branches are backed by at least one
   labelled positive example, and which fire only on placeholder
   thresholds with *no* example constraining them (``rules.py`` openly
   admits the female bare-legs and male bare-thigh thresholds are tuned
   to the segmenter noise floor, not to real positives).

**Important honesty caveat**: the bundled ``labels.json`` points at the
same seven ``test_images/`` that every threshold in ``rules.py`` was fit
against. A pass-rate over them is *training-set* performance — a
regression / sanity guard — **not** held-out accuracy. Real accuracy
measurement needs labelled data the thresholds never saw. The harness
prints this caveat in its report so the number is never read as more
than it is.
"""

from occlude.eval.harness import (
    BRANCHES,
    CaseResult,
    Report,
    categorize_reason,
    evaluate,
    format_report,
    load_cases,
)

__all__ = [
    "BRANCHES",
    "CaseResult",
    "Report",
    "categorize_reason",
    "evaluate",
    "format_report",
    "load_cases",
]
