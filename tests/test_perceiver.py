"""Tests for the Perceiver protocol and VideoProcessor injection.

Critically: these tests do NOT load any model weights.  A FakePerceiver
satisfying the protocol is injected into VideoProcessor so the
constructor completes in milliseconds.
"""
import numpy as np
import pytest
from PIL import Image

from occlude.pipeline.perception import TARGET_LABELS, Perceiver, Perception, Person
from occlude.pipeline.video import VideoProcessor


class FakePerceiver:
    """Returns scripted detections; never touches disk or a model."""

    def __init__(self, people: list[Person] | None = None) -> None:
        self._people = people or []

    def detect_and_segment(self, image: Image.Image) -> list[Person]:
        return list(self._people)

    def classify(
        self, crop: Image.Image
    ) -> tuple[str | None, float, float | None]:
        return "F", 0.9, None


def test_fake_perceiver_satisfies_protocol():
    assert isinstance(FakePerceiver(), Perceiver)


def test_perception_class_has_required_methods():
    # Structural check only — does not instantiate Perception (would load models).
    from occlude.pipeline.perception import Perception
    assert hasattr(Perception, "detect_and_segment")
    assert hasattr(Perception, "classify")


def test_video_processor_accepts_fake_perceiver():
    # VideoProcessor should construct in <1 ms when a fake perceiver is injected.
    fake = FakePerceiver()
    vp = VideoProcessor(blur_kernel=11, perception_batch=1, perceiver=fake)
    assert vp.perception is fake


def test_video_processor_forwards_detector_model(monkeypatch):
    """The --detector-model knob must reach Perception's constructor.

    Patches the heavy Perception with a recorder so no weights load:
    the test verifies wiring intent (the flag is plumbed through), not
    detection behaviour.
    """
    import occlude.pipeline.video as video

    captured: dict[str, object] = {}

    class _RecorderPerception:
        def __init__(self, *, device=None, detector_model=None) -> None:
            captured["device"] = device
            captured["detector_model"] = detector_model

        def detect_and_segment(self, image):  # pragma: no cover - protocol stub
            return []

        def classify(self, crop):  # pragma: no cover - protocol stub
            return None, 0.0, None

    monkeypatch.setattr(video, "Perception", _RecorderPerception)

    VideoProcessor(blur_kernel=11, perception_batch=1, detector_model="yolov8m.pt")
    assert captured["detector_model"] == "yolov8m.pt"

    VideoProcessor(blur_kernel=11, perception_batch=1)
    assert captured["detector_model"] is None


def test_perception_defaults_detector_model_to_yolov8n():
    """Default must stay yolov8n.pt so the locked H1 benchmark hash holds."""
    from occlude.pipeline.perception import YOLO_MODEL_ID

    assert YOLO_MODEL_ID == "yolov8n.pt"


def test_pick_device_prefers_cuda_over_mps(monkeypatch):
    import occlude.pipeline.perception as perception

    monkeypatch.setattr(perception.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(perception.torch.backends.mps, "is_available", lambda: True)

    assert perception._pick_device().type == "cuda"


def test_pick_device_cuda_request_fails_without_cuda(monkeypatch):
    import occlude.pipeline.perception as perception

    monkeypatch.setattr(perception.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        perception._pick_device("cuda")


def test_require_cuda_io_respects_disable_env(monkeypatch):
    import occlude.pipeline.io_cuda as io_cuda

    monkeypatch.setenv("OCCLUDE_DISABLE_CUDA_IO", "1")

    with pytest.raises(RuntimeError, match="CUDA video I/O required"):
        io_cuda.require_cuda_io_available()


def test_make_label_masks_only_materializes_rule_labels():
    mask = np.zeros((4, 4), dtype=np.uint8)

    label_masks = Perception.make_label_masks(mask)

    assert set(label_masks) == TARGET_LABELS
    assert all(value.dtype == np.bool_ for value in label_masks.values())
