"""Unit tests for blur_region — no models required.

Verifies that:
  - pixels inside the bbox are modified
  - pixels outside the padded work region are untouched
  - a None/zero-size seg_mask falls back to a rectangle blur without crashing
  - a zero-size bbox is a no-op
"""
import numpy as np
import torch

from occlude.pipeline.video import (
    SILHOUETTE_DILATE_FRAC,
    SILHOUETTE_DILATE_MIN_PX,
    SILHOUETTE_FEATHER_FRAC,
    SILHOUETTE_FEATHER_MIN_PX,
    apply_blur_mask,
    blur_region,
    prepare_blur_mask,
)


def _to_numpy(mask):
    """Normalize a blur mask to np.ndarray for type-agnostic assertions.

    prepare_blur_mask returns np.ndarray on Mac/CPU and torch.Tensor on
    CUDA; the legacy tests below predate the dispatch and pin their
    assertions to numpy types/attrs. This helper keeps them honest on
    both paths without rewriting each assertion."""
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy()
    return mask

# Kernel small enough to run fast in tests but large enough to visibly
# change a random frame.
_KERNEL = 11

# Frame large enough that the padded region doesn't touch the border.
FH, FW = 400, 400


def _random_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (FH, FW, 3), dtype=np.uint8)


def _pad_for(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    short = min(bh, bw)
    pad_frac = SILHOUETTE_DILATE_FRAC + SILHOUETTE_FEATHER_FRAC
    pad_min = SILHOUETTE_DILATE_MIN_PX + SILHOUETTE_FEATHER_MIN_PX
    return max(pad_min, int(short * pad_frac))


def test_pixels_inside_bbox_are_changed():
    frame = _random_frame()
    original = frame.copy()
    bbox = (100, 100, 200, 200)
    seg = np.ones((100, 100), dtype=np.int32)  # all foreground

    blur_region(frame, bbox, seg, _KERNEL)

    x1, y1, x2, y2 = bbox
    inner = frame[y1 + 5:y2 - 5, x1 + 5:x2 - 5]
    inner_orig = original[y1 + 5:y2 - 5, x1 + 5:x2 - 5]
    assert not np.array_equal(inner, inner_orig), "bbox interior should be blurred"


def test_pixels_outside_padded_region_are_untouched():
    frame = _random_frame()
    original = frame.copy()
    bbox = (100, 100, 200, 200)
    seg = np.ones((100, 100), dtype=np.int32)

    pad = _pad_for(bbox)
    x1, y1, x2, y2 = bbox
    px1 = max(0, x1 - pad)
    py1 = max(0, y1 - pad)
    px2 = min(FW, x2 + pad)
    py2 = min(FH, y2 + pad)

    blur_region(frame, bbox, seg, _KERNEL)

    # Rows fully above the padded region
    assert np.array_equal(frame[:py1, :], original[:py1, :])
    # Rows fully below
    assert np.array_equal(frame[py2:, :], original[py2:, :])
    # Cols fully left
    assert np.array_equal(frame[:, :px1], original[:, :px1])
    # Cols fully right
    assert np.array_equal(frame[:, px2:], original[:, px2:])


def test_none_seg_mask_falls_back_to_rectangle_blur():
    frame = _random_frame()
    original = frame.copy()
    bbox = (100, 100, 200, 200)

    blur_region(frame, bbox, None, _KERNEL)

    x1, y1, x2, y2 = bbox
    inner = frame[y1 + 5:y2 - 5, x1 + 5:x2 - 5]
    inner_orig = original[y1 + 5:y2 - 5, x1 + 5:x2 - 5]
    assert not np.array_equal(inner, inner_orig)


def test_background_only_seg_mask_falls_back_to_rectangle_blur():
    # seg_mask all zeros means no silhouette pixels were found.
    # The correct behaviour is to fall back to a feathered rectangle
    # so the person is still blurred when the rule engine said blur.
    frame = _random_frame()
    original = frame.copy()
    bbox = (100, 100, 200, 200)
    seg = np.zeros((100, 100), dtype=np.int32)  # all background

    blur_region(frame, bbox, seg, _KERNEL)

    # Interior of the bbox should be blurred (not pixel-exact).
    x1, y1, x2, y2 = bbox
    inner = frame[y1 + 5:y2 - 5, x1 + 5:x2 - 5]
    inner_orig = original[y1 + 5:y2 - 5, x1 + 5:x2 - 5]
    assert not np.array_equal(inner, inner_orig)


def test_zero_size_bbox_is_noop():
    frame = _random_frame()
    original = frame.copy()

    blur_region(frame, (100, 100, 100, 100), None, _KERNEL)  # zero width
    blur_region(frame, (100, 100, 50, 100), None, _KERNEL)   # negative width

    np.testing.assert_array_equal(frame, original)


# --- prepare_blur_mask / apply_blur_mask ---


def _expected_region_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Return (rh, rw) for the padded work region, assuming the frame is FH×FW."""
    pad = _pad_for(bbox)
    x1, y1, x2, y2 = bbox
    px1 = max(0, x1 - pad)
    py1 = max(0, y1 - pad)
    px2 = min(FW, x2 + pad)
    py2 = min(FH, y2 + pad)
    return py2 - py1, px2 - px1


def test_prepare_blur_mask_shape_with_seg_mask():
    bbox = (100, 100, 200, 200)
    seg = np.ones((100, 100), dtype=np.int32)
    mask = _to_numpy(prepare_blur_mask(bbox, seg, (FH, FW)))
    assert mask.shape == _expected_region_shape(bbox)


def test_prepare_blur_mask_shape_fallback():
    bbox = (100, 100, 200, 200)
    mask = _to_numpy(prepare_blur_mask(bbox, None, (FH, FW)))
    assert mask.shape == _expected_region_shape(bbox)


def test_prepare_blur_mask_values_in_range():
    bbox = (100, 100, 200, 200)
    seg = np.ones((100, 100), dtype=np.int32)
    mask = _to_numpy(prepare_blur_mask(bbox, seg, (FH, FW)))
    assert mask.dtype == np.float32
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0


def test_prepare_blur_mask_nonzero_coverage():
    # With a fully-foreground seg_mask, the centre of the mask should be 1.
    bbox = (100, 100, 200, 200)
    seg = np.ones((100, 100), dtype=np.int32)
    mask = _to_numpy(prepare_blur_mask(bbox, seg, (FH, FW)))
    rh, rw = mask.shape
    centre = mask[rh // 2, rw // 2]
    assert centre > 0.9, f"centre of silhouette mask should be near 1, got {centre}"


def test_prepare_blur_mask_degenerate_bbox():
    mask = _to_numpy(prepare_blur_mask((100, 100, 100, 100), None, (FH, FW)))
    assert mask.size == 0


def test_apply_blur_mask_matches_blur_region():
    # apply_blur_mask(prepare_blur_mask(...), ...) must produce byte-identical
    # output to the combined blur_region call.
    bbox = (100, 100, 200, 200)
    seg = np.ones((100, 100), dtype=np.int32)
    frame_shape = (FH, FW)

    frame_combined = _random_frame()
    blur_region(frame_combined, bbox, seg, _KERNEL)

    frame_split = _random_frame()  # same seed=0 default, so identical
    mask = prepare_blur_mask(bbox, seg, frame_shape)
    apply_blur_mask(frame_split, bbox, mask, _KERNEL)

    np.testing.assert_array_equal(frame_split, frame_combined)


def test_apply_blur_mask_degenerate_is_noop():
    frame = _random_frame()
    original = frame.copy()
    apply_blur_mask(frame, (100, 100, 200, 200), np.zeros((0, 0), dtype=np.float32), _KERNEL)
    np.testing.assert_array_equal(frame, original)


# --- Torch path (exercised directly on CPU so Mac dev boxes validate it
# without needing CUDA; in production the same code runs on _BLUR_DEVICE) ---


def test_prepare_blur_mask_torch_matches_numpy_shape_and_range():
    """The torch mask builder must produce the same shape and stay in
    [0, 1] as the numpy reference. Bit-exact equality isn't expected:
    cv2.dilate(MORPH_ELLIPSE) and conv-with-ellipse-kernel-threshold are
    visually equivalent but not pixel-identical at the elliptical boundary,
    which is then masked by the wide Gaussian feather anyway."""
    import torch

    from occlude.pipeline.video import _prepare_blur_mask_torch

    bbox = (100, 100, 200, 200)
    seg = np.ones((100, 100), dtype=np.int32)
    np_mask = prepare_blur_mask(bbox, seg, (FH, FW))
    t_mask = _prepare_blur_mask_torch(bbox, seg, (FH, FW), torch.device("cpu"))

    assert t_mask.shape == np_mask.shape
    assert t_mask.dtype == torch.float32
    assert float(t_mask.min()) >= 0.0
    assert float(t_mask.max()) <= 1.0
    rh, rw = t_mask.shape
    assert float(t_mask[rh // 2, rw // 2]) > 0.9


def test_prepare_blur_mask_torch_fallback_path():
    """No seg_mask → feathered rectangle, on the torch path."""
    import torch

    from occlude.pipeline.video import _prepare_blur_mask_torch

    bbox = (100, 100, 200, 200)
    np_mask = prepare_blur_mask(bbox, None, (FH, FW))
    t_mask = _prepare_blur_mask_torch(bbox, None, (FH, FW), torch.device("cpu"))
    assert t_mask.shape == np_mask.shape
    rh, rw = t_mask.shape
    assert float(t_mask[rh // 2, rw // 2]) > 0.9


def test_prepare_blur_mask_torch_degenerate_returns_empty():
    import torch

    from occlude.pipeline.video import _prepare_blur_mask_torch

    t_mask = _prepare_blur_mask_torch(
        (100, 100, 100, 100), None, (FH, FW), torch.device("cpu")
    )
    assert t_mask.numel() == 0


def test_apply_blur_mask_accepts_torch_mask_on_cpu_path():
    """When the CPU/OpenCV blur path is active (Mac/no-CUDA), a torch
    tensor mask must still be accepted — apply_blur_mask converts it to
    numpy on the way into the cv2 path. Guards against a regression
    where the dispatch forgets the torch→numpy conversion."""
    import torch

    frame = _random_frame()
    original = frame.copy()
    bbox = (100, 100, 200, 200)
    seg = np.ones((100, 100), dtype=np.int32)

    np_mask = prepare_blur_mask(bbox, seg, (FH, FW))
    if isinstance(np_mask, torch.Tensor):
        # CUDA box: we already validate the torch path elsewhere.
        return
    t_mask = torch.from_numpy(np_mask)

    apply_blur_mask(frame, bbox, t_mask, _KERNEL)
    # Interior must have been blurred (not bit-equal but not all original).
    x1, y1, x2, y2 = bbox
    assert not np.array_equal(
        frame[y1 + 5:y2 - 5, x1 + 5:x2 - 5],
        original[y1 + 5:y2 - 5, x1 + 5:x2 - 5],
    )


def test_prepare_blur_mask_torch_accepts_tensor_seg():
    """Passing a torch tensor seg_mask must work — this is the path
    used once Person.seg_mask migrates to GPU tensors in step 2/3."""
    import torch

    from occlude.pipeline.video import _prepare_blur_mask_torch

    bbox = (100, 100, 200, 200)
    seg = torch.ones((100, 100), dtype=torch.int32)
    t_mask = _prepare_blur_mask_torch(bbox, seg, (FH, FW), torch.device("cpu"))
    rh, rw = t_mask.shape
    assert float(t_mask[rh // 2, rw // 2]) > 0.9
