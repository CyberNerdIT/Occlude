"""IoU tracklet association — identity only, the basis for per-track verdicts."""
from occlude.pipeline.track import TrackletBuilder, iou
from occlude.pipeline.tracklets import Detection


def _det(frame, bbox):
    return Detection(frame_idx=frame, bbox=bbox, score=0.9)


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_same_person_links_across_frames():
    b = TrackletBuilder()
    b.add_frame(0, [_det(0, (10, 10, 50, 100))])
    b.add_frame(1, [_det(1, (12, 11, 52, 101))])  # slight motion, high IoU
    tracks = b.finalize()
    assert len(tracks) == 1
    assert len(tracks[0]) == 2


def test_two_people_two_tracklets():
    b = TrackletBuilder()
    b.add_frame(0, [_det(0, (0, 0, 40, 100)), _det(0, (200, 0, 240, 100))])
    b.add_frame(1, [_det(1, (1, 0, 41, 100)), _det(1, (201, 0, 241, 100))])
    assert len(b.finalize()) == 2


def test_shot_cut_splits_one_person_into_two_tracklets():
    # Same screen position across a cut must NOT be linked — a cut can swap
    # who stands there, and carrying a verdict across would mis-blur.
    b = TrackletBuilder()
    b.add_frame(0, [_det(0, (10, 10, 50, 100))])
    b.add_frame(1, [_det(1, (10, 10, 50, 100))], is_cut=True)
    tracks = b.finalize()
    assert len(tracks) == 2


def test_short_gap_bridged_long_gap_split():
    b = TrackletBuilder(max_gap=5)
    b.add_frame(0, [_det(0, (10, 10, 50, 100))])
    # detector misses frames 1-3, person reappears at frame 4 (gap 4 <= 5)
    b.add_frame(4, [_det(4, (11, 10, 51, 100))])
    assert len(b.finalize()) == 1

    b2 = TrackletBuilder(max_gap=5)
    b2.add_frame(0, [_det(0, (10, 10, 50, 100))])
    b2.add_frame(10, [_det(10, (11, 10, 51, 100))])  # gap 10 > 5 -> split
    assert len(b2.finalize()) == 2


def test_min_length_filters_blips():
    b = TrackletBuilder()
    b.add_frame(0, [_det(0, (0, 0, 40, 100))])           # one-frame blip
    b.add_frame(1, [_det(1, (200, 0, 240, 100))])
    b.add_frame(2, [_det(2, (201, 0, 241, 100))])
    assert len(b.finalize(min_length=2)) == 1
