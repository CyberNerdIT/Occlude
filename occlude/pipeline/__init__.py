"""Pipeline package — the v2 offline, three-pass stack.

The heavy model stack lives in submodules. Keep the package import itself
lightweight so `python -m occlude --help` can load constants without
importing torch, OpenCV, Ultralytics, transformers, or SAM2.
"""
from importlib import import_module

__all__ = [
    # data model
    "Detection", "Tracklet", "Verdict", "BBox",
    # passes
    "PersonDetector",          # Pass 1 detect
    "TrackletBuilder", "SAM2Segmenter",  # Pass 1 track / Pass 3 segment
    "ShotSegmenter",           # shot-cut detection
    "VLMJudge",                # Pass 2 judge
    "aggregate",               # Pass 2 verdict policy
    "VideoProcessor", "blur_region",  # orchestrator + render
]

_MODULE_OF = {
    "Detection": "tracklets", "Tracklet": "tracklets", "Verdict": "tracklets",
    "BBox": "tracklets",
    "PersonDetector": "detect",
    "TrackletBuilder": "track", "SAM2Segmenter": "track",
    "ShotSegmenter": "scenes",
    "VLMJudge": "judge",
    "aggregate": "decide",
    "VideoProcessor": "video", "blur_region": "video",
}


def __getattr__(name: str):  # noqa: ANN001
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"occlude.pipeline.{module}"), name)
