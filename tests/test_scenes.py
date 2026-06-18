"""Shot-cut detection: catch real cuts, ignore within-shot motion."""
import numpy as np

from occlude.pipeline.scenes import (
    ShotSegmenter,
    gray_histogram,
    histogram_distance,
)


def test_histogram_normalized():
    g = np.full((10, 10), 100, dtype=np.uint8)
    h = gray_histogram(g)
    assert abs(h.sum() - 1.0) < 1e-6


def test_empty_frame_histogram_is_zero():
    h = gray_histogram(np.zeros((0, 0), dtype=np.uint8))
    assert h.sum() == 0.0


def test_identical_frames_zero_distance():
    g = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    assert histogram_distance(gray_histogram(g), gray_histogram(g)) == 0.0


def test_disjoint_frames_max_distance():
    black = gray_histogram(np.zeros((32, 32), dtype=np.uint8))
    white = gray_histogram(np.full((32, 32), 255, dtype=np.uint8))
    assert histogram_distance(black, white) == 1.0


def test_segmenter_first_frame_is_shot_start():
    seg = ShotSegmenter()
    assert seg.push(np.full((16, 16), 50, dtype=np.uint8)) is True


def test_segmenter_flags_cut_not_motion():
    seg = ShotSegmenter()
    base = np.full((64, 64), 80, dtype=np.uint8)
    seg.push(base)
    # Small motion-like perturbation keeps the intensity distribution close:
    # not a cut.
    jittered = base.copy()
    jittered[:8, :8] = 120
    assert seg.push(jittered) is False
    # Hard cut to a very different scene: flagged.
    assert seg.push(np.full((64, 64), 240, dtype=np.uint8)) is True
