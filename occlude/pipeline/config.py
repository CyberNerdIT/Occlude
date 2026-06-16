"""Lightweight pipeline defaults shared by CLI and runtime code."""

DEFAULT_BLUR_KERNEL = 199
DEFAULT_PERCEPTION_BATCH_CUDA = 4
DEFAULT_PERCEPTION_BATCH_OTHER = 1

# YOLO person-detector weights. yolov8n (nano) is the default on every
# device because the locked benchmark hash (progress.txt H1) was
# established with it; changing the default would silently invalidate
# that regression lock. A missed detection means a missed blur — the
# worst error class for this tool — so on a CUDA box with spare VRAM
# (the A100 bench peaked at 2.66 GB of 40 GB) passing a larger model
# via --detector-model (e.g. yolov8m.pt or yolov8l.pt) trades a little
# throughput for materially better recall on small / distant / odd-angle
# subjects. Ultralytics downloads the weights on first use.
DEFAULT_DETECTOR_MODEL = "yolov8n.pt"
