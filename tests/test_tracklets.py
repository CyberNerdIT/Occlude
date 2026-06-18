"""Tracklet data-model behavior the later passes depend on."""
import numpy as np

from occlude.pipeline.tracklets import Detection, Tracklet


def _det(frame, bbox, score=0.9):
    return Detection(frame_idx=frame, bbox=bbox, score=score)


def test_add_and_frame_span():
    t = Tracklet(track_id=1)
    t.add(_det(5, (0, 0, 10, 10)))
    t.add(_det(2, (0, 0, 10, 10)))
    t.add(_det(9, (0, 0, 10, 10)))
    assert t.frames == [2, 5, 9]
    assert t.first_frame == 2
    assert t.last_frame == 9
    assert len(t) == 3


def test_add_same_frame_overwrites():
    # Re-detecting in a frame replaces, never duplicates — the render pass
    # indexes detections by frame and must get exactly one per frame.
    t = Tracklet(track_id=1)
    t.add(_det(3, (0, 0, 10, 10)))
    t.add(_det(3, (1, 1, 20, 20)))
    assert len(t) == 1
    assert t.detections[3].bbox == (1, 1, 20, 20)


def test_best_frames_ranks_by_evidence():
    # "Best" must prefer the largest, most-confident view so the VLM judges
    # the clearest evidence — the whole point of deciding offline.
    t = Tracklet(track_id=1)
    t.add(_det(0, (0, 0, 10, 10), score=0.99))     # tiny, area 100
    t.add(_det(1, (0, 0, 100, 100), score=0.5))    # large, area 10000
    t.add(_det(2, (0, 0, 50, 50), score=0.9))      # medium, area 2500
    assert t.best_frames(1) == [1]
    assert t.best_frames(2) == [1, 2]


def test_detection_mask_defaults_full_frame_none():
    d = Detection(frame_idx=0, bbox=(0, 0, 4, 4), score=0.8)
    assert d.mask is None
    d.mask = np.ones((8, 8), dtype=bool)
    assert d.mask.shape == (8, 8)
