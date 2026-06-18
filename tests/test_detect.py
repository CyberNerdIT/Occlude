"""Detector result parsing — degenerate edge boxes must be dropped.

On a feature-length film the detector produces sub-pixel-wide boxes at frame
edges. After the int() floor these collapse to zero area and would crash
crop/segmentation downstream, aborting the whole run. _results_to_detections
must drop them. No model weights: a fake result feeds scripted boxes.
"""
import numpy as np

from occlude.pipeline.detect import _results_to_detections


class _Arr:
    def __init__(self, arr):
        self.arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _Arr(xyxy)
        self.conf = _Arr(conf)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


def test_degenerate_boxes_filtered():
    xyxy = np.array(
        [
            [50.0, 40.0, 200.0, 400.0],    # valid
            [300.2, 100.0, 300.7, 350.0],  # zero width after int()
            [400.0, 220.4, 480.0, 220.9],  # zero height after int()
        ],
        dtype=np.float32,
    )
    conf = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    dets = _results_to_detections(_Result(_Boxes(xyxy, conf)), frame_idx=7)
    assert len(dets) == 1
    assert dets[0].bbox == (50, 40, 200, 400)
    assert dets[0].frame_idx == 7


def test_no_boxes_is_empty():
    assert _results_to_detections(_Result(None), frame_idx=0) == []
