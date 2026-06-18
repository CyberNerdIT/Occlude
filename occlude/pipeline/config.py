"""Lightweight pipeline defaults shared by CLI and runtime code.

Kept import-cheap so `occlude --help` can read constants without pulling in
torch / OpenCV. Model-specific defaults (detector weights, judge model id)
live next to their wrappers in detect.py / judge.py.
"""

# Default Gaussian blur kernel (overridable via --blur-strength). 199 on a
# 720p frame is a thoroughly opaque smudge.
DEFAULT_BLUR_KERNEL = 199
