# OCCLUDE — Product Requirements Document

## Problem Statement

Muslim families, educators, and content consumers regularly watch documentaries, lectures, and
educational videos that contain useful information but also footage of immodestly dressed people
— shirtless men, women with uncovered hair, people in shorts or sleeveless clothing. Existing
tools (HaramBlur, PordaAI) are browser extensions that work in real-time, which forces them into
brutal accuracy trade-offs: they blur incorrectly, miss content, or produce flickering results that
are distracting rather than helpful. There is no offline tool that processes a video file once,
produces a clean permanent copy, and lets the viewer watch it without distraction.

## Solution

OCCLUDE is an open-source command-line tool that takes a local video file as input, detects
every person who is immodestly dressed according to Islamic modesty rules, blurs their entire
silhouette (not just the offending body part), preserves the original audio track, and writes a
clean output video. Processing is offline and one-shot: run it once, get a clean video you can
watch forever.

The tool targets Apple Silicon Macs as the primary platform and is designed to be extended by
contributors who want to improve accuracy, add GPU support, or build higher-level interfaces on
top.

## User Stories

### Core CLI Workflow

1. As a Muslim content consumer, I want to run a single command on a video file, so that I get
   a clean version without having to learn video editing.
2. As a user, I want the output file placed next to the input with a predictable name suffix
   (`_occluded.mp4`), so that I do not have to specify an output path for the common case.
3. As a user, I want to see a progress bar with current frame, total frames, and estimated time
   remaining while the video is processing, so that I know how long to wait.
4. As a user, I want the tool to fail with a clear error message if the input file does not
   exist, so that I catch typos immediately.
5. As a user, I want the tool to fail with a clear error message if `ffmpeg` is not installed,
   with install instructions, so that I know exactly what dependency is missing.
6. As a user, I want to override the output path with `--output`, so that I can direct the clean
   video to a different location or naming convention.
7. As a user, I want to override blur intensity with `--blur-strength`, so that I can tune the
   aggressiveness of the blur for my display setup.
8. As a user, I want to control the perception batch size with `--perception-batch`,
   so that CUDA runs can trade peak VRAM for throughput without skipping frames.
9. As a user, I want the original audio track preserved exactly in the output, so that speech,
   music, and sound effects are unchanged.
10. As a user, I want the output video to have the same resolution, frame rate, and codec as the
    input, so that playback behaviour is identical.
11. As a user, I want to cancel processing with Ctrl-C and get a clean exit (exit code 130),
    so that I can stop a long job without a crash trace.

### Modesty Rules — Women

12. As a Muslim user, I want a woman with uncovered hair to be blurred, so that the content
    respects Islamic modesty standards for women.
13. As a Muslim user, I want a woman with bare arms to be blurred, so that exposed arms in
    sleeveless clothing are hidden.
14. As a Muslim user, I want a woman with bare legs to be blurred, so that skirts, shorts, and
    dresses that expose the leg are hidden.
15. As a Muslim user, I want a woman with an exposed neck or chest to be blurred, so that
    low-cut necklines are hidden.
16. As a Muslim user, I want a woman in full hijab, abaya, or other complete Islamic modesty
    attire to remain unblurred, so that modestly dressed women are not incorrectly censored.
17. As a Muslim user, I want a woman wearing a hat or beret but with long hair flowing past her
    face to be blurred, so that a hat does not "cancel out" visible flowing hair.

### Modesty Rules — Men

18. As a Muslim user, I want a shirtless man to be blurred, so that an exposed torso is hidden.
19. As a Muslim user, I want a man wearing shorts that expose the thigh to be blurred, so that
    the area above the knee is hidden.
20. As a Muslim user, I want a man in a t-shirt and full-length trousers to remain unblurred,
    so that normally dressed men are not incorrectly censored.

### Blur Appearance

21. As a viewer, I want the blur to follow the person's body outline (silhouette), not a
    rectangular box, so that it looks natural and does not cover unrelated background area.
22. As a viewer, I want the blur edges to fade softly into the background, so that there is no
    sharp rectangular cut or hard paper-cutout edge.
23. As a viewer, I want the blur to fully obscure interior detail — cleavage lines, button
    placement, fabric patterns — so that the underlying shape is not reconstructable from
    low-frequency luminance gradients.
24. As a viewer, I want the blur to be stable frame-to-frame without flickering on or off, so
    that watching the processed video is not distracting.
25. As a viewer, I want the blur to continue briefly when YOLO temporarily loses a person between
    frames, so that momentary detection failures do not flash the underlying image.

### Accuracy and Edge Cases

26. As a user, I want the gender classifier to recover from a wrong first-frame classification
    within a fraction of a second using a rolling majority vote, so that a bad initial call does
    not leave a person unblurred for a whole second.
27. As a user, I want subjects who appear just after a scene cut to be blurred within 3 frames if
    they should be, so that scene transitions do not create a visible window of uncensored content.
28. As a user, I want the tool to default to the female (stricter) ruleset when the gender
    classifier finds no face or returns low confidence, so that the bias errs on the side of
    modesty.
29. As a user, I want a predicted-male person whose hair extends past the face label to be
    re-evaluated as female, so that women with short or obscured faces are not misclassified
    as male and left unblurred.
30. As a user, I want purely background content — objects, animals, text overlays, landscapes —
    to remain unblurred, so that only people who meet the modesty trigger criteria are affected.
31. As a user, I want partially-occluded people near the frame edge to be evaluated only when
    enough of their body is visible to make a reliable determination, so that the tool avoids
    spurious blurs on partially visible subjects.

### Performance

32. As a user on CUDA/Colab, I want the pipeline to use CUDA for YOLO, SegFormer,
    blur, and InsightFace, so that long videos finish in a practical time.
33. As a user, I want the tool to keep its memory footprint bounded over a long video, so that
    processing a 1-hour file does not grow to fill RAM and get killed by the OS.
34. As a user, I want SegFormer inference to run as a single batched forward pass per frame
    rather than one call per detected person, so that multi-person scenes are processed faster.
35. As a user, I want gender classification to be re-used across frames via a per-track cache,
    so that InsightFace ONNX does not run on every frame and cause memory growth.

### Open Source and Contribution

36. As a contributor, I want a clear README explaining what OCCLUDE is, why it exists, the
    modesty rules it applies, and how to install and use it, so that I can understand the
    project without reading the source code.
37. As a contributor, I want an explicit invitation in the README for accuracy improvements,
    GPU support, alternative segmentation models, and front-end work, so that I know where
    effort is welcome.
38. As a contributor, I want the modesty rule thresholds to be named constants with inline
    comments explaining which test cases constrain them, so that I can tune them without
    breaking known-good behaviour.
39. As a contributor wanting to improve the model, I want a set of labelled test images
    covering modest and immodest cases for both genders, so that I can verify a candidate
    model against them before integrating it into the pipeline.

## Implementation Decisions

### Pipeline Architecture

OCCLUDE is a sequential pipeline of five deep modules that are composed by the CLI entry
point (`occlude.py`) via the `VideoProcessor` orchestrator:

- **Perception** (`pipeline/perception.py`): Wraps YOLOv8 person detection, SegFormer
  semantic segmentation, and InsightFace gender classification as a single callable. Accepts
  a PIL Image, returns a list of `Person` dataclasses. The video pipeline calls
  `detect_and_segment` and `classify` separately to allow gender caching across frames.

- **RuleEngine** (`pipeline/rules.py`): Maps a `Person` (segmentation mask + gender + face
  detection score) to a binary `Decision` (blur: bool, reason: str). Contains all modesty
  logic as a pure function with no video or I/O dependencies. Three-level female head
  check (region → connected component shape → Hat/Scarf tiebreaker), shirtless fast path
  for men, thigh-region check for men, hair-below-face check for the hat+ponytail case,
  and a male→female gender cross-check override.

- **VideoProcessor** (`pipeline/video.py`): Reads frames via OpenCV, runs the
  Perception + RuleEngine pipeline per frame, maintains per-person IoU-matched tracks
  with bbox smoothing, temporal blur voting, and gender voting, applies the silhouette blur
  via `_apply_blur`, writes a silent video with OpenCV, and muxes the original audio back
  with FFmpeg.

### Segmentation Model

`mattmdjaga/segformer_b2_clothes` (HuggingFace) — 18-class fashion parsing model. Outputs
labels: Background, Hat, Hair, Sunglasses, Upper-clothes, Skirt, Pants, Dress, Belt,
Left/Right-shoe, Face, Left/Right-leg, Left/Right-arm, Bag, Scarf. The rule layer only
consumes: Hair, Hat, Scarf, Upper-clothes, Pants, Skirt, Dress, Face, legs, arms.

### Gender Classifier

InsightFace `buffalo_l` with only `detection` and `genderage` modules loaded. The ONNX
session has CPU memory arena and mem_pattern disabled to bound per-frame heap growth on
macOS. Gender is cached per track and re-evaluated every 30 frames (burst-reclassify every
frame until 3 votes exist on a new track).

### Person Detector

YOLOv8 nano (`yolov8n.pt`) run at 0.40 confidence threshold on the `person` class only.

### Blur Algorithm

For each flagged person per frame:
1. Pad the work region outward by `dilate + feather` pixels.
2. Pixelate: downsample with `INTER_AREA` to `~18` blocks across the short side; upsample
   with `INTER_NEAREST` back to original size.
3. Gaussian-blur the pixelated result with the configured kernel (default 199, must be odd).
4. Build an alpha mask from the SegFormer silhouette (`seg_mask > 0`), morphologically dilate
   by 5 % of the short side (floor 13 px), Gaussian-feather by 15 % (floor 25 px).
5. Composite blurred image over original via alpha mask in the padded region.

### Temporal Tracking

- IoU threshold: 0.3 to match detections to existing tracks.
- Bbox smoothing: 5-frame moving average per track; window resets if the new raw bbox
  deviates more than 0.6 IoU from the smoothed position (scene-cut signal).
- Carry-forward: blur at last smoothed bbox for 10 frames after YOLO loses a tracked person.
- Blur vote window: majority of last 10 blur decisions determines the frame decision;
  requires at least 3 decisions before voting kicks in.
- Gender vote window: majority of last 5 high-confidence classifications (≥ 0.5 score);
  burst re-classifies until 3 votes exist.

### Memory Management

- `torch.mps.empty_cache()` remains aggressive on the local Mac path.
- `torch.cuda.empty_cache()` is disabled by default and can be enabled with
  `OCCLUDE_CUDA_EMPTY_CACHE_EVERY=N` only when profiling shows it helps.
- `gc.collect()` runs periodically, not every frame.
- `malloc_zone_pressure_relief(0, 0)` runs periodically on macOS only.
- SegFormer logits: argmax on-device (128×128 int) before CPU transfer — avoids the
  18-channel float32 tensor on the CPU heap.
- SegFormer inference batched over all person crops in a frame in one forward pass.
- InsightFace ONNX: arena and mem_pattern disabled; only detection+genderage submodels loaded.

### CLI Flags

| Flag | Default | Constraint |
|---|---|---|
| `--input` | required | must be an existing file |
| `--output` | `<stem>_occluded.mp4` next to input | parent dir created if needed |
| `--blur-strength` | 199 | positive odd integer |
| `--perception-batch` | 4 on CUDA, 1 elsewhere | integer ≥ 1 |
| `--device` | `auto` | one of `auto`, `cuda`, `mps`, `cpu` |
| `--require-cuda-io` | false | fail unless CUDA video decode/encode is available |

### External Dependencies

- Python 3.10+, PyTorch ≥ 2.2, transformers ≥ 4.40, Ultralytics ≥ 8.2,
  insightface ≥ 0.7, onnxruntime ≥ 1.16, OpenCV ≥ 4.9, scipy ≥ 1.10, tqdm ≥ 4.65.
- FFmpeg on PATH (install via `brew install ffmpeg` on macOS).

## Testing Decisions

### What Makes a Good Test

Tests should verify externally observable behaviour, not implementation internals. For this
pipeline, that means:

- **RuleEngine tests**: feed a `Person` with a hand-crafted `seg_mask` and known gender,
  assert the `Decision.blur` value and `reason` string. Do not assert on internal threshold
  names or intermediate computation. The test images in `test_images/` (7 images, 4
  immodest / 3 modest, male and female) are the ground-truth fixture set.

- **Perception tests**: feed a static test image, assert that at least one `Person` is
  returned with the expected gender and that the seg_mask contains the expected label
  classes. Do not assert on exact pixel counts or specific model weights.

- **VideoProcessor tests**: feed a short synthetic video (a few frames of a solid colour or
  a test image repeated), assert that the output file exists and has audio. End-to-end
  correctness is verified by visual inspection of the test clip outputs in `test_output/`.

- **_apply_blur tests**: feed a synthetic BGR frame and a known bbox + seg_mask, assert that
  the pixels inside the silhouette changed and the pixels clearly outside did not.

### Modules to Test

| Module | Test type | Priority |
|---|---|---|
| `RuleEngine.decide` | Unit — per-image ground truth | High |
| `Perception.__call__` | Integration — static test images | Medium |
| `VideoProcessor.process` | Integration — synthetic short video | Medium |
| `_apply_blur` | Unit — synthetic frame | Low |
| `_iou` | Unit | Low |
| `_smoothed_bbox` | Unit | Low |

### Prior Art

`scripts/test_rules.py`, `scripts/test_perception.py`, `scripts/test_detect_and_segment.py`,
and `scripts/test_segmenter.py` are the existing test scripts. Formal pytest tests should
follow the same structure: load a fixture image, instantiate the module under test, assert
on the output.

## Out of Scope

- **GUI or desktop application**: drag-and-drop interface is a v2 contribution item.
- **Real-time stream processing**: OCCLUDE is designed for offline batch processing only.
- **Cloud processing or server deployment**: `scripts/chunked_occlude.py` and
  `notebooks/occlude_colab.ipynb` exist as escape valves for long videos but are not
  packaged as a service.
- **Audio filtering**: out of scope by design — that is a separate project (ELUATE).
- **Automatic video downloading**: users supply their own video files.
- **Direct streaming platform integration**: no YouTube/Netflix/etc. hooks.
- **Windows support**: not currently targeted.
- **Linux/CUDA support**: supported for Colab-style long runs; the CUDA path is the primary
  performance target.
- **CoreML model conversion**: noted as a performance opportunity in the spec but not
  implemented in v1.

## Further Notes

### Known v1 Limitations and Planned v2 Work

**Accuracy improvements (v2 priority 1):**
- Baseball cap + ponytail case: Hat/Scarf:Hair tiebreaker can incorrectly read a hat over
  uncovered ponytail as "covered". Documented in `docs/03-rule-design.md`.
- Below-knee male skin: the thigh-region check (2.5–4.0 face-height units below the chin)
  intentionally excludes calves to avoid blurring men in ankle-length trousers with rolled
  cuffs; this means men in knee-length shorts may only trigger the shirtless fast path.
- False negatives when face is occluded: gender defaults to Female (safe), but the
  segmenter then must catch hair/arm/leg — if those are also occluded, the person may
  escape blur. Over-blur bias covers the no-face case explicitly.
- Model alternatives: `FASHN-Human-Parser` and `DeepLabV3+ ResNet-50` were listed in the
  spec as candidates over SegFormer. They were not tested because SegFormer proved sufficient
  for v1 but may provide better granularity for edge cases.

**Performance (v2 priority 2):**
- Tune `--perception-batch` defaults on A100/L4 with long-video benchmarks.
- Move more of decode -> crop -> segmentation -> blur through CUDA tensors to reduce
  host/device transfers.

**Platform support (v2 priority 3):**
- Docker image: would make cross-platform installation trivial and remove the ffmpeg/Python
  dependency management burden from users.
- Windows: OpenCV and FFmpeg are available on Windows; InsightFace ONNX is the likely
  friction point.

### Relationship to ELUATE

OCCLUDE and ELUATE are sibling tools with the same UX pattern (one input, one output, no
intermediate steps). ELUATE handles audio content filtering; OCCLUDE handles visual modesty
filtering. They are designed to be composable: a user can run ELUATE then OCCLUDE on the
same file.

### Design Principle

The blur is binary and whole-person: either a person's entire silhouette is blurred or they
are untouched. There is no partial body-part blur. This is a deliberate UX and accuracy
decision — partial blurs are harder to tune correctly and more distracting to watch than a
clean silhouette blur.
