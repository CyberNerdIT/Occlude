#!/usr/bin/env python3
"""Run the OCCLUDE rule layer over the labelled image set and print a report.

Loads the real perception stack (YOLO + InsightFace + SegFormer), so it
takes a few seconds to start and downloads model weights on first run.
This is a deliberately-invoked accuracy / coverage check, not a CI unit
test — the fast, weights-free logic tests live in tests/test_eval.py.

Usage:
    python scripts/eval_accuracy.py
    python scripts/eval_accuracy.py --device mps
    python scripts/eval_accuracy.py --images test_images --labels occlude/eval/labels.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from occlude.eval import evaluate, format_report, load_cases  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, default=_REPO / "test_images",
        help="directory holding the labelled images",
    )
    parser.add_argument(
        "--labels", type=Path, default=_REPO / "occlude" / "eval" / "labels.json",
        help="ground-truth labels JSON",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
        help="inference device for the perception stack",
    )
    parser.add_argument(
        "--detector-model", type=str, default=None,
        help="YOLO weights override (default yolov8n.pt)",
    )
    args = parser.parse_args(argv)

    cases = load_cases(args.labels)
    if not cases:
        print("no cases in labels file", file=sys.stderr)
        return 1

    # Imported here so --help stays free of the multi-second model import.
    from occlude.pipeline.perception import Perception

    print(f"loading perception stack on device={args.device} ...", file=sys.stderr)
    perceiver = Perception(device=args.device, detector_model=args.detector_model)

    report = evaluate(cases, perceiver, args.images)
    print(format_report(report))

    # Exit non-zero if any blur decision regressed, so this can gate a
    # release. Coverage gaps are a warning, not a failure (they reflect
    # missing data, not a code regression).
    return 0 if report.n_blur_correct == report.n else 2


if __name__ == "__main__":
    raise SystemExit(main())
