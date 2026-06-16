# PRD — OCCLUDE Performance & Architecture Improvements

## Problem Statement

OCCLUDE processes video at roughly 1 fps baseline (6-person 1280×720 on Apple Silicon), bottlenecked by SegFormer inference. Several concrete improvements — both architectural (reducing coupling, improving testability) and performance (fp16, compilation, mask caching) — have been identified from code inspection. None require changes to the modesty rules or blur quality.

## Solution

A focused set of targeted changes that reduce inference time, remove accumulated technical debt, and make the tracker testable with configurable parameters — without touching the rule layer or the blur visual output.

## User Stories

1. As a user processing long videos, I want SegFormer to run faster, so that a 90-minute video takes hours instead of days.
2. As a user on CUDA/Colab, I want fp16 inference, so that NVIDIA tensor cores are used efficiently.
3. As a user on CUDA/Colab, I want torch.compile on the segmentation model, so that kernel fusion reduces forward-pass latency.
4. As a user with frame stride > 1, I want skip frames to be cheaper, so that carry-forward blur doesn't recompute the silhouette mask unnecessarily.
5. As a developer writing tracker tests, I want to inject carry-forward and vote-window sizes, so that I can probe 2-frame carry-forward without waiting 10 frames.
6. As a developer swapping the segmentation model, I want the rule layer to receive named label maps, so that reordering model labels doesn't silently corrupt blur decisions.
7. As a developer running `occlude --help`, I want startup to be instant and side-effect-free, so that diagnostic file writes don't fail in restricted environments.
8. As a developer reading the codebase, I want dead code removed, so that I don't study a `_segment` method that is never called.
9. As a developer debugging with `OCCLUDE_TRACK_LOG`, I want the log file held open for the duration of processing, so that 180 file-open syscalls per second don't add latency.
10. As a developer on any future accelerator, I want the memory cleanup path to be a single context manager, so that adding a new device means editing one place.

## Implementation Decisions

### Performance

- **fp16 SegFormer**: Cast the segmentation model to half-precision after loading on CUDA/MPS. The output logits are immediately reduced via argmax; intermediate float values are never used, so fp16 and fp32 produce identical label maps in practice. CUDA/Colab is the primary target.

- **torch.compile**: Wrap the segmentation model with `torch.compile` after the fp16 cast. Applies kernel fusion to the transformer attention blocks. Warm-up cost on first batch; all subsequent batches benefit. CUDA/Colab runs use `--device cuda` so missing CUDA support fails before a long job starts.

- **Precompute blur masks on perception frames**: Split `blur_region` into a prepare step (dilation + feather → returns a float alpha mask) and an apply step (composite using a precomputed mask). On perception frames, compute and cache the mask alongside the smoothed bbox in `last_blur_targets`. On skip frames, call only the apply step. Removes redundant `cv2.dilate` + `cv2.GaussianBlur` on every carry-forward frame.

- **Reduce `malloc_zone_pressure_relief` frequency**: Currently called every frame (`MEMORY_CLEANUP_EVERY = 1`). The original problem (heap fragmentation by frame ~90) was caused by InsightFace arena + MPS allocator; both are now fixed independently. Evaluate calling every 10–25 frames to recover 5–20 ms/frame on macOS without reopening the fragmentation issue.

### Architecture

- **Tracker constructor tunables**: Add `carry_forward_frames`, `gender_vote_burst`, `gender_vote_window`, `blur_vote_window`, and `bbox_smoothing_frames` as keyword arguments to `Tracker.__init__`, with defaults matching the current module constants. Tests can pass small values. The module-level constants become the defaults only, not baked-in behaviour.

- **Named label map seam**: Add a method to `Perception` that converts a raw integer segmentation mask into a `dict[str, np.ndarray]` keyed by canonical label names (hair, face, arms, etc.). The rule layer receives this named map and never indexes into `SEG_LABELS` directly. Swapping the segmentation model requires updating only `Perception`, not `RuleEngine`.

- **Remove diagnostic instrumentation from `cli.py`**: The `(TEMPORARY)` block (monkey-patching `os._exit`, `sys.exit`, opening two `/tmp` files) runs unconditionally at module import. Move behind an `OCCLUDE_DEBUG=1` environment variable check, or remove entirely if the original crash has been identified. `--help` must produce no side effects.

- **Delete `_segment`**: The single-image `_segment` method in `Perception` is never called from the production path — `detect_and_segment` uses `_segment_batch` exclusively. Delete it.

- **Buffer `TRACK_LOG_PATH` writes**: Replace the per-track `open(path, "a")` inside `Tracker.update` with a module-level or instance-level file handle opened once at the start of `VideoProcessor.process` and closed in `finally`. Eliminates per-track syscall overhead when the log is active.

- **Context manager for device cleanup**: Extract the `gc.collect()` + `torch.{mps,cuda}.empty_cache()` pattern from `_segment_batch` (and the now-deleted `_segment`) into a `_cleanup_device_memory` context manager. One location to update if a new device is added.

## Testing Decisions

- Good tests verify external behaviour through the module's public interface — not internal state, not private methods. A passing test must survive an internal refactor without modification.
- The `Perceiver` protocol is the test seam for `VideoProcessor` and `Tracker`. Tests inject a `FakePerceiver`; no model weights are loaded.
- Tracker tunability: once constructor arguments are added, test `carry_forward_frames=2` to verify blur ghost lasts exactly 2 frames after detection loss — currently untestable without patching globals.
- Named label map: test `RuleEngine.decide` with hand-crafted `dict[str, np.ndarray]` inputs. No model involved; rule logic is pure.
- Blur mask prepare/apply split: test that the prepared mask for a known seg_mask matches the expected float alpha array (shape, value range, silhouette coverage). Test that apply with a precomputed mask produces the same composited output as the current single-call path.
- Prior art: `tests/test_tracker.py` (ScriptedPerceiver pattern), `tests/test_rules.py` (synthetic mask arrays), `tests/test_blur.py` (boundary conditions on `blur_region`).

## Tasks

- [x] Delete `_segment` (dead code, 37 lines, `perception.py`)
- [x] Add `carry_forward_frames`, `gender_vote_burst`, `gender_vote_window`, `blur_vote_window`, `bbox_smoothing_frames` as `Tracker.__init__` keyword args; update tracker tests to pass small values
- [x] Move diagnostic block in `cli.py` behind `OCCLUDE_DEBUG=1`; verify `--help` produces no file writes
- [x] Open `TRACK_LOG_PATH` file handle once in `VideoProcessor.process`; pass to `Tracker.update` or store as instance state
- [x] Extract device-memory cleanup into a `_cleanup_device_memory()` helper or context manager in `perception.py`
- [x] Cast segmentation model to fp16 after load; verify label map argmax matches fp32 baseline on all 7 test images
- [x] Apply `torch.compile` to segmentation model; document expected warm-up frames in a comment; keep CUDA/Colab runs explicit with `--device cuda`
- [x] Split `blur_region` into `prepare_blur_mask(bbox, seg_mask) -> np.ndarray` and `apply_blur_mask(frame_bgr, bbox, mask)`; update `last_blur_targets` to store precomputed masks; update tests
- [x] Add named-label-map method to `Perception`; refactor `RuleEngine` helpers to consume `dict[str, np.ndarray]`; remove `SEG_LABELS` import from `rules.py`; update rule tests
- [x] Profile `malloc_zone_pressure_relief` call frequency on a known video; raise `MEMORY_CLEANUP_EVERY` if memory stays bounded

## Out of Scope

- Changes to the modesty rules or rule thresholds.
- Changes to blur visual quality (kernel size, feather shape, pixelation block size).
- Real-time (streaming) processing — OCCLUDE is a batch processor by design.
- Multi-video batch processing.
- Model accuracy improvements or model swaps.

## Further Notes

The fp16 + torch.compile changes should be applied together and benchmarked on the same video with the same `--perception-batch` setting to isolate their combined effect. The named-label-map refactor is the most architecturally significant change and should be done independently so its diff is reviewable in isolation.
