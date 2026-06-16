# Changelog

## [1.2.0] — 2026-05-20

### Fixed
- **Blur late to engage (~2 s lag on a woman at the center of frame).**
  Root cause: InsightFace exposes no calibrated gender probability
  (docs/04, Finding 2), so a confidently-wrong "M" on a woman locked her
  unblurred until the 30-frame reclassify cadence flipped the majority.
  Tracker now applies an asymmetric vote during the early life of a track
  (while the vote window isn't yet full): a single high-confidence "F"
  vote forces F immediately. False-F (extra blur on a man) is the
  accepted, self-correcting cost of the over-blur bias the tool already
  commits to elsewhere; once the window matures, majority rules again.
- **Blur silhouette stays one frame after the subject leaves the frame.**
  Two composing fixes in `Tracker.update` carry-forward:
  - **Edge-exit kill.** A track whose smoothed bbox sits within 6 px of
    any frame border is treated as having walked off-screen; carry-forward
    is killed on that frame instead of holding the silhouette inward for
    `CARRY_FORWARD_FRAMES` more frames.
  - **Alpha-fade.** Carried silhouettes now decay linearly across the
    carry window (`alpha = carry_remaining / CARRY_FORWARD_FRAMES`) so
    legitimate mid-frame dropouts fade out smoothly instead of holding a
    crisp ghost at the last position. The blur application multiplies the
    prepared float mask by the per-target alpha.

### Changed
- `Tracker.update` signature: now accepts an optional `frame_shape` arg
  (required to enable the edge-exit kill; omitted in unit tests that only
  exercise carry-forward timing) and returns
  `list[(bbox, seg_mask, alpha)]` (was `list[(bbox, seg_mask)]`).

## [1.1.5] — 2026-05-19

### Fixed
- Crash when YOLO emits a sub-pixel-thin detection. `int()` truncation of
  the box coordinates collapsed it to zero width or height; the empty crop
  then aborted the run inside the segmenter (`F.interpolate` / `cv2.resize`
  reject a zero spatial dimension). `detect_and_segment` now drops boxes
  that floor to zero area before they reach segmentation. Regression test
  added (`tests/test_degenerate_bbox.py`), no model weights required.

## [1.1.4] — 2026-05-19

### Fixed
- Hang during GPU preprocessing when PIL crop produces a non-writable buffer.
  `np.asarray()` returned a read-only view; torch.compile's fused kernels could
  attempt in-place writes on it, causing undefined behavior and a process hang.
  Changed to `np.array()` which always returns a writable copy.

## [1.1.3] — 2026-05-19

### Fixed
- Crash when a person's bounding box is narrower than the Gaussian blur kernel
  (padding 99 ≥ region width). Both the CUDA and CPU blur paths now clamp the
  kernel to `min(k, 2 * min(rh, rw) - 1)` before applying, so very narrow
  crops (e.g. a person partially off-frame) no longer abort the run.

## [1.1.2] — 2026-05-19

### Changed
- **Full CUDA pipeline.** Every model and the per-frame blur now run on
  the GPU when CUDA is selected; the only mandatory CPU work left is
  video decode/encode.
  - **SegFormer image preprocessing** moved off the CPU HF
    `SegformerImageProcessor` onto on-device torch ops (resize →
    rescale → normalize). Argmax-mask agreement vs the processor is
    ~99.2% on the test images; the sub-pixel resampling differences are
    absorbed by the downstream ≥13 px dilation + ≥25 px feather.
  - **Per-frame blur** (pixelate + 199-kernel Gaussian + composite)
    ported from OpenCV/CPU to a torch CUDA path (separable Gaussian
    matching `cv2.GaussianBlur`'s sigma, area/nearest pixelate). Output
    matches the OpenCV reference within ±1/255. Non-CUDA installs keep
    the OpenCV path, which the unit tests still pin.
  - **YOLO** now runs fp16 inference on CUDA (output is thresholded, so
    the precision loss is benign — same argument as the SegFormer
    `.half()`).
- **Device routing generalized.** YOLO and InsightFace also route onto
  Apple Silicon when present (YOLO → MPS; InsightFace → CoreML EP, with
  a loud CPU-fallback warning when CoreML can't bind the buffalo_l
  graph). CUDA behaviour is unchanged from the intent of 1.1.1.

### Fixed
- **Colab `occlude[gpu]` no longer silently runs InsightFace on CPU.**
  Plain `onnxruntime` is a core dependency and shadows
  `onnxruntime-gpu`, so the `[gpu]` extra alone left the face/gender
  ONNX sessions on the CPU EP (~6 fps). The notebook's validation and
  profile cells now apply the same uninstall-both +
  force-reinstall-`onnxruntime-gpu`-last repair the full-run cell
  already used, and the CPU-fallback warning now explains the shadowing
  and prints the exact fix.

### Notes
- The Colab notebook pins `occlude[gpu]==1.1.2`; this release must be
  published to TestPyPI before the notebook picks the changes up.
- `torch.compile` is left ON for the full movie run (cell 6) where the
  one-time warm-up amortizes, and OFF for the short validation/profile
  cells so their timings reflect steady state.

## [1.1.1] — 2026-05-19

### Fixed
- **Per-frame `gc.collect()` removed from the hot path.**
  `_cleanup_device_memory` ran a full `gc.collect()` on every
  perception frame; with torch+transformers resident each collect is
  ~0.2 s, which profiled as ~36% of total runtime — the single
  largest cost, dwarfing the GPU models. It now collects every
  `GC_EVERY` (50) calls; MPS cache cleanup still runs every call, while
  CUDA cache cleanup is now opt-in via `OCCLUDE_CUDA_EMPTY_CACHE_EVERY`. Memory
  stays bounded (video.py also cleans periodically at the loop level);
  throughput improves dramatically.

### Added
- `occlude[gpu]` optional extra (pulls `onnxruntime-gpu`) to run
  InsightFace's face/gender ONNX sessions on CUDA.

### Changed
- Perception now routes InsightFace onto the CUDA execution provider
  and YOLO onto the GPU when the selected device is CUDA. Previously
  InsightFace ran under the CPU execution provider; on CUDA runs this
  added per-track CPU cost (secondary to the gc.collect issue above).
- When a CUDA device is selected but the CUDA ONNX provider is
  unavailable (no `onnxruntime-gpu`, or a CUDA/cuDNN mismatch),
  InsightFace now emits a loud stderr warning and falls back to CPU
  instead of silently crawling.

### Notes
- macOS/MPS and CPU-only installs are unaffected: the base dependency
  stays `onnxruntime`, and behaviour on non-CUDA devices is unchanged.
- If you previously ran `pip install occlude`, plain `onnxruntime` is
  already present and pip will **not** remove it when adding the extra.
  Run `pip uninstall -y onnxruntime` first, then
  `pip install 'occlude[gpu]'`, so the CUDA-enabled runtime is the one
  that loads.

## [1.0.0] — 2026-05-01

First public release.
