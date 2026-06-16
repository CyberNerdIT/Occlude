"""Pipeline package — perception, rules, blur, video.

The heavy model stack lives in submodules. Keep package import itself
lightweight so `python -m occlude --help` can load constants without
importing torch, OpenCV, Ultralytics, or Matplotlib-adjacent deps.
"""
from importlib import import_module

__all__ = [
    "SEG_LABELS", "TARGET_LABELS", "Perception", "Perceiver", "Person",
    "Decision", "RuleEngine",
    "VideoProcessor", "Tracker", "blur_region",
]


def __getattr__(name: str):  # noqa: ANN001
    if name in {"SEG_LABELS", "TARGET_LABELS", "Perception", "Perceiver", "Person"}:
        perception = import_module("occlude.pipeline.perception")
        return getattr(perception, name)
    if name in {"Decision", "RuleEngine"}:
        rules = import_module("occlude.pipeline.rules")
        return getattr(rules, name)
    if name in {"VideoProcessor", "Tracker", "blur_region"}:
        video = import_module("occlude.pipeline.video")
        return getattr(video, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
