"""End-to-end 3-pass orchestration with fake models (no GPU, no ffmpeg).

Verifies the passes wire together: a detected person who is judged "blur"
gets blurred across the whole tracklet, and one judged "clear" does not.
The heavy models are injected as fakes so this runs anywhere.
"""
import cv2
import numpy as np
import pytest

from occlude.pipeline.tracklets import Detection, Verdict
from occlude.pipeline.video import VideoProcessor

PERSON_BOX = (16, 16, 48, 48)
N_FRAMES = 6
SIZE = 64


def _checkerboard(size: int) -> np.ndarray:
    g = np.indices((size, size)).sum(axis=0) % 2
    return (g * 255).astype(np.uint8)


def _make_video(path, person=True):
    # High-frequency checkerboard inside the person box so blur is detectable
    # as a variance drop that lossy compression alone wouldn't cause.
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (SIZE, SIZE)
    )
    if not writer.isOpened():
        pytest.skip("mp4v VideoWriter unavailable in this environment")
    board = _checkerboard(SIZE)
    for _ in range(N_FRAMES):
        frame = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)
        if person:
            x1, y1, x2, y2 = PERSON_BOX
            frame[y1:y2, x1:x2, :] = board[y1:y2, x1:x2, None]
        writer.write(frame)
    writer.release()


class _FakeDetector:
    def detect(self, frame_bgr, frame_idx):
        return [Detection(frame_idx=frame_idx, bbox=PERSON_BOX, score=0.95)]


class _FakeJudge:
    def __init__(self, blur):
        self._blur = blur
        self.calls = 0

    def judge_crops(self, images):
        self.calls += len(images)
        return [Verdict(blur=self._blur, is_human=True) for _ in images]


def _region_variance(path):
    cap = cv2.VideoCapture(str(path))
    ret, frame = cap.read()
    cap.release()
    assert ret, "output video had no readable frame"
    x1, y1, x2, y2 = PERSON_BOX
    return float(frame[y1:y2, x1:x2].var())


def test_blur_verdict_blurs_the_person(tmp_path):
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _make_video(src, person=True)
    in_var = _region_variance(src)

    judge = _FakeJudge(blur=True)
    vp = VideoProcessor(
        blur_kernel=7, use_sam2=False, detector=_FakeDetector(), judge=judge
    )
    vp.process(src, out, skip_mux=True)

    assert out.exists()
    assert judge.calls > 0  # the tracklet was actually judged
    # Blur collapses the checkerboard's high-frequency variance.
    assert _region_variance(out) < 0.5 * in_var


def test_clear_verdict_leaves_person(tmp_path):
    src = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _make_video(src, person=True)
    in_var = _region_variance(src)

    vp = VideoProcessor(
        blur_kernel=7, use_sam2=False, detector=_FakeDetector(),
        judge=_FakeJudge(blur=False),
    )
    vp.process(src, out, skip_mux=True)

    # No blur applied: the person region keeps most of its detail.
    assert _region_variance(out) > 0.5 * in_var


def test_segmenter_falls_back_when_sam2_missing(monkeypatch, capsys):
    """pip install occlude has no sam2; render must degrade to box blur."""
    from occlude.pipeline import video as video_mod

    class _MissingSam2:
        def __init__(self, *args, **kwargs):
            raise ImportError("No module named 'sam2'")

    monkeypatch.setattr(video_mod, "SAM2Segmenter", _MissingSam2)
    vp = VideoProcessor(use_sam2=True)
    assert vp.segmenter is None
    assert vp.segmenter is None  # sticky: no retry, no second warning
    err = capsys.readouterr().err
    assert err.count("sam2 is not installed") == 1
