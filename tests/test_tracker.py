"""Unit tests for Tracker — no models required.

All tests inject a ScriptedPerceiver so no model weights are loaded.
Geometry: bbox (10, 10, 90, 110) → bw=80, bh=100, matching the 100×80
synthetic seg_masks produced by the helpers below.
"""
import numpy as np
from PIL import Image

from occlude.pipeline.perception import SEG_LABELS, Perception, Person
from occlude.pipeline.rules import RuleEngine
from occlude.pipeline.video import Tracker

L = {name: i for i, name in enumerate(SEG_LABELS)}
BBOX = (10, 10, 90, 110)   # 80 wide × 100 tall


class ScriptedPerceiver:
    """Replays a fixed list of (gender, score) classify responses.

    detect_and_segment is unused — callers pass Person lists directly to
    Tracker.update().  classify_calls counts how many times it was invoked.
    """

    def __init__(self, responses: list[tuple]) -> None:
        self._queue = list(responses)
        self.classify_calls = 0

    def detect_and_segment(self, image):
        return []

    def classify(self, crop):
        self.classify_calls += 1
        r = self._queue.pop(0) if self._queue else (None, 0.0)
        # Accept legacy 2-tuples (gender, score); age defaults to None.
        return r if len(r) == 3 else (r[0], r[1], None)


def _mask_female_blurs() -> np.ndarray:
    """Mask → blur=True for F (uncovered hair), blur=False for M (covered).

    Face rows 20–40, cols 20–60.  Hair 10×10 px in head region (3.1% CC,
    no Hat/Scarf → female hair rule fires).  Upper-clothes fills the
    torso so the male shirtless rule stays silent.
    """
    m = np.zeros((100, 80), dtype=np.int32)
    m[20:41, 20:61] = L["Face"]
    m[41:81, :] = L["Upper-clothes"]
    m[5:15, 30:40] = L["Hair"]
    return m


def _mask_covered() -> np.ndarray:
    """Mask that passes both male and female rules (no uncovered hair/arms/legs)."""
    m = np.zeros((100, 80), dtype=np.int32)
    m[20:41, 20:61] = L["Face"]
    m[41:81, :] = L["Upper-clothes"]
    return m


def _person(bbox=BBOX, seg_mask=None) -> Person:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mask = seg_mask if seg_mask is not None else _mask_female_blurs()
    return Person(
        bbox=bbox,
        det_conf=0.9,
        crop=Image.fromarray(np.zeros((bh, bw, 3), dtype=np.uint8)),
        seg_mask=mask,
        gender=None,
        face_det_score=0.0,
        label_masks=Perception.make_label_masks(mask),
    )


# ---------------------------------------------------------------------------
# New track behaviour
# ---------------------------------------------------------------------------

def test_new_track_calls_classify():
    perceiver = ScriptedPerceiver([("F", 0.9)])
    tracker = Tracker(perceiver, RuleEngine())
    tracker.update(0, [_person()])
    assert perceiver.classify_calls == 1


def test_no_blur_for_covered_person():
    tracker = Tracker(ScriptedPerceiver([("F", 0.9)]), RuleEngine())
    targets = tracker.update(0, [_person(seg_mask=_mask_covered())])
    assert targets == []


def test_blur_target_returned_for_uncovered_female():
    tracker = Tracker(ScriptedPerceiver([("F", 0.9)]), RuleEngine())
    targets = tracker.update(0, [_person()])
    assert len(targets) == 1
    assert targets[0][0] is not None   # smoothed bbox
    assert targets[0][1] is not None   # seg_mask


# ---------------------------------------------------------------------------
# Carry-forward
# ---------------------------------------------------------------------------

def test_carry_forward_returns_targets_after_detection_loss():
    tracker = Tracker(ScriptedPerceiver([("F", 0.9)]), RuleEngine())
    tracker.update(0, [_person()])           # frame 0: detection
    targets = tracker.update(1, [])          # frame 1: no detection
    assert len(targets) == 1


def test_carry_forward_expires_after_k_frames():
    k = 2
    tracker = Tracker(ScriptedPerceiver([("F", 0.9)]), RuleEngine(), carry_forward_frames=k)
    tracker.update(0, [_person()])

    for frame in range(1, k + 1):
        t = tracker.update(frame, [])
        assert len(t) == 1, f"carry should still fire at frame {frame}"

    t = tracker.update(k + 1, [])
    assert t == [], "track should be dropped after carry window"


# ---------------------------------------------------------------------------
# Classification reuse within the burst window
# ---------------------------------------------------------------------------

def test_existing_track_reuses_classify_once_burst_window_fills():
    burst = 3
    # Burst reclassifies on frames 0, 1, 2 (len(votes) < burst=3).
    # Frame 3 should reuse the cached votes without calling classify again.
    responses = [("F", 0.9)] * (burst + 5)
    perceiver = ScriptedPerceiver(responses)
    tracker = Tracker(perceiver, RuleEngine(), gender_vote_burst=burst)
    mask = _mask_covered()

    for frame_idx in range(burst + 1):
        tracker.update(frame_idx, [_person(seg_mask=mask)])

    # Frame 0: new track (1 call); frames 1 and 2: burst reclassify (1 call each);
    # frame 3: len(votes)=burst, not below threshold → no call.
    # Total = burst calls.
    assert perceiver.classify_calls == burst


# ---------------------------------------------------------------------------
# Gender vote correction
# ---------------------------------------------------------------------------

def test_gender_vote_corrects_wrong_first_call():
    # First call returns M (wrong); subsequent calls return F.
    # After enough F votes accumulate and blur vote reaches majority, blur fires.
    scripted = ScriptedPerceiver([("M", 0.8)] + [("F", 0.8)] * 20)
    tracker = Tracker(scripted, RuleEngine())
    mask = _mask_female_blurs()

    found_blur = False
    for frame_idx in range(15):
        targets = tracker.update(frame_idx, [_person(seg_mask=mask)])
        if targets:
            found_blur = True
            break

    assert found_blur, "blur should fire after gender vote corrects M→F"


def test_immature_window_forces_blur_on_first_f_vote():
    # Over-blur bias: a confident wrong "M" first call must not lock a
    # woman unblurred. As soon as one high-conf F vote appears while the
    # vote window is still immature, she blurs — no ~2 s reclassify wait.
    scripted = ScriptedPerceiver([("M", 0.9), ("F", 0.9)] + [("F", 0.9)] * 10)
    tracker = Tracker(scripted, RuleEngine())
    mask = _mask_female_blurs()

    assert tracker.update(0, [_person(seg_mask=mask)]) == [], "frame 0: only M vote"
    targets = tracker.update(1, [_person(seg_mask=mask)])
    assert len(targets) == 1, "frame 1: first F vote (immature) must force blur"


# ---------------------------------------------------------------------------
# Child exemption (perception → tracker → rules age flow)
# ---------------------------------------------------------------------------

def test_child_age_exempts_from_blur_through_tracker():
    # A female-uncovered-hair mask that would normally blur, but the
    # classifier reports a child age — must not blur.
    scripted = ScriptedPerceiver([("F", 0.9, 8.0)] * 6)
    tracker = Tracker(scripted, RuleEngine())
    mask = _mask_female_blurs()

    targets = []
    for frame_idx in range(4):
        targets = tracker.update(frame_idx, [_person(seg_mask=mask)])
    assert targets == [], "detected child must never be blurred"


def test_adult_age_still_blurs_through_tracker():
    scripted = ScriptedPerceiver([("F", 0.9, 30.0)] * 6)
    tracker = Tracker(scripted, RuleEngine())
    targets = tracker.update(0, [_person(seg_mask=_mask_female_blurs())])
    assert len(targets) == 1, "adult control must still blur"


# ---------------------------------------------------------------------------
# Out-latency: edge-exit carry kill + alpha fade
# ---------------------------------------------------------------------------

def test_edge_exit_kills_carry_forward():
    frame_shape = (120, 100)
    edge_bbox = (40, 10, 98, 110)   # x2=98 ≥ w-EDGE_TOUCH_PX → flush to right edge
    tracker = Tracker(ScriptedPerceiver([("F", 0.9)]), RuleEngine())
    tracker.update(0, [_person(bbox=edge_bbox)], frame_shape)
    assert tracker.update(1, [], frame_shape) == [], (
        "subject flush to frame edge must not be carried forward"
    )


def test_non_edge_dropout_still_carries():
    # Same dropout but no frame_shape → edge logic disabled, carry intact.
    tracker = Tracker(ScriptedPerceiver([("F", 0.9)]), RuleEngine())
    tracker.update(0, [_person()])
    assert len(tracker.update(1, [])) == 1


def test_carry_forward_alpha_fades_out():
    k = 4
    tracker = Tracker(ScriptedPerceiver([("F", 0.9)]), RuleEngine(), carry_forward_frames=k)
    tracker.update(0, [_person()])           # active frame, alpha 1.0
    alphas = []
    for frame in range(1, k + 1):
        t = tracker.update(frame, [])
        assert len(t) == 1
        alphas.append(t[0][2])
    assert alphas == [1.0, 0.75, 0.5, 0.25], alphas
    assert tracker.update(k + 1, []) == []
