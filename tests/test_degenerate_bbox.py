"""A thin YOLO detection truncates to a zero-area crop.

`int(x1f), int(x2f)` in detect_and_segment floors both coordinates, so a
detection only a fraction of a pixel wide (x1f=100.2, x2f=100.7) collapses
to x1 == x2. `image.crop` then yields a 0-width PIL image, which crashes
the segmenter (`F.interpolate` / `cv2.resize` reject a zero spatial dim).
On a feature-length film YOLO produces such boxes at frame edges, so a
single one aborts the whole run. detect_and_segment must drop these
before they reach segmentation.

No model weights are loaded: a fake detector feeds detect_and_segment
scripted boxes and _segment_batch is stubbed to record the crop sizes
it is handed.
"""
import numpy as np
from PIL import Image

from occlude.pipeline.perception import Perception


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = _FakeArr(xyxy)
        self.conf = _FakeArr(conf)

    def __len__(self) -> int:
        return len(self.xyxy.arr)


class _FakeArr:
    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr

    def cpu(self) -> "_FakeArr":
        return self

    def numpy(self) -> np.ndarray:
        return self.arr


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self.boxes = boxes


class _FakeDetector:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self._boxes = boxes

    def predict(self, **_kwargs):  # noqa: ANN003
        return [_FakeResult(self._boxes)]


def _bare_perception(detector: _FakeDetector, seen_sizes: list) -> Perception:
    p = object.__new__(Perception)
    p.person_conf = 0.4
    p._yolo_device = "cpu"
    p._yolo_half = False
    p.detector = detector

    def _fake_segment_batch(crops):
        seen_sizes.extend(c.size for c in crops)
        return [np.zeros((h, w), dtype=np.int32) for (w, h) in (c.size for c in crops)]

    p._segment_batch = _fake_segment_batch
    return p


def test_degenerate_bbox_is_filtered_before_segmentation():
    image = Image.new("RGB", (640, 480))
    # One valid person, one sub-pixel-wide detection (x2 - x1 < 1 → x1 == x2
    # after int truncation), and one zero-height detection.
    xyxy = np.array(
        [
            [50.0, 40.0, 200.0, 400.0],   # valid
            [300.2, 100.0, 300.7, 350.0],  # zero width after int()
            [400.0, 220.4, 480.0, 220.9],  # zero height after int()
        ],
        dtype=np.float32,
    )
    conf = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    seen_sizes: list = []
    perception = _bare_perception(_FakeDetector(_FakeBoxes(xyxy, conf)), seen_sizes)

    people = perception.detect_and_segment(image)

    assert all(w > 0 and h > 0 for (w, h) in seen_sizes), (
        f"a zero-area crop reached segmentation: {seen_sizes}"
    )
    assert len(people) == 1
    assert people[0].bbox == (50, 40, 200, 400)
