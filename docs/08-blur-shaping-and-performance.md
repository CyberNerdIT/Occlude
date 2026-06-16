# Stage 7 — Blur shaping, performance, Colab path

> Status: in progress. Visual changes verified on still test images and a 30s laughing_people clip; longer-video verification pending. Performance improvements implemented but not yet end-to-end-benchmarked on a multi-person clip.

## What changed

Stage 6 shipped a working but visually-rough blur — a feathered rectangle over the dilated bbox — and a runtime profile of ~1 fps end-to-end on multi-person 1280×720 video. Stage 7 was three threads of work, all driven by user feedback after watching real output:

1. **Blur shape**: rectangle → silhouette traced from the SegFormer mask, with a wide soft feather. Round 2 added a pixelation pre-pass to fully obscure interior detail (cleavage line, button placement) that pure Gaussian preserves as low-frequency luminance gradients.
2. **Performance**: batched YOLO + SegFormer across consecutive frames, GPU-side preprocessing/blur, and CUDA video I/O where available. Every frame is still analysed; the speedup comes from batching and keeping the hot path on CUDA.
3. **Colab runner**: `notebooks/occlude_colab.ipynb` for processing long-form video (documentaries, full-length TV episodes) on a Pro+ GPU runtime in 1–3 hours instead of 30+ hours on a Mac Mini.

## 1. Blur shape

### Round 1 — silhouette alpha

The Stage 6 blur applied a Gaussian to the dilated bbox region and composited back through a feathered alpha mask whose 1.0 region was a rectangle. User feedback after the v6 output: *"the blur is just a big rectangle on top of each person"*. Right read — the rectangle's edge feathered, but the interior was opaque-rectangular, not body-shaped.

Fix: the SegFormer mask we already compute for rule evaluation **is** the silhouette. `(seg_mask > 0)` flags every pixel labelled as any body part (Background = 0). Resize that to the smoothed bbox dimensions, dilate by a few pixels, Gaussian-feather, and use it as the alpha. Result on the woman-in-jumpsuit test image:

- Background sky and beach stay sharp.
- The dilated silhouette traces the body outline, fading softly into the background.
- No rectangle artifact.

`BBOX_DILATION_PCT` (the 25 % bbox dilation that existed only to hide the rectangle edge) was removed — nothing rectangular left to hide.

Code: `pipeline/video.py::_apply_blur` now takes a `seg_mask` argument and builds the alpha from it. `_TrackedPerson.last_seg_mask` caches the most recent silhouette so carry-forward frames also blur on the body shape rather than falling back to a rectangle.

### Round 2 — paper-cutout look

User feedback on the round-1 output: *"too much like a paper cutting thing — corners are too sharp, expand by 10–15 px and soften the falloff"*. Two parameters tuned:

- `SILHOUETTE_DILATE_FRAC` 0.03 → 0.05 with a **13 px floor**. Even on a 200-px-tall background subject the alpha now extends ~13 px past the segmentation edge.
- `SILHOUETTE_FEATHER_FRAC` 0.06 → 0.15 with a **25 px floor**. The feather kernel is now visibly wide — soft fade over tens of pixels rather than a near-cutout.

### Round 2 — intensity

User feedback after the softer mask: *"still too much scene visible — the goal is to not differentiate where the cleavage is"*.

The advisor's earlier note mattered here: **stacking Gaussian passes does not increase opacity, it widens the kernel**. Even at K=199 the Gaussian preserves the low-frequency luminance pattern — the eye reconstructs body detail from the blurred contrast. The standard redaction technique is **pixelation** (downsample with `INTER_AREA`, upsample with `INTER_NEAREST`) which destroys positional information at scale below the block size. A small Gaussian on top hides the discrete block edges so the output reads as "soft blur" rather than "censorship mosaic".

Implementation in `_apply_blur`:

```python
block = max(PIXELATE_MIN_BLOCK_PX, min(rh, rw) // PIXELATE_BLOCKS_ACROSS)
small = cv2.resize(region, (rw // block, rh // block), interpolation=cv2.INTER_AREA)
pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
blurred = cv2.GaussianBlur(pixelated, (k, k), 0)
```

`PIXELATE_BLOCKS_ACROSS = 18` means ~18 mosaic blocks across the short side of the bbox region. The Gaussian then smooths block edges into a continuous-looking blur. Visual: interior is now uniformly obscured — fabric stripes, button lines, cleavage edges all gone.

### Round 3 — edge cuts and scene-change unblur

User feedback after v8 on the 30s laughing_people clip: *"some top parts are too clean straight cut, and when scene changes there is like a second of subject unblurred"*. Two distinct bugs.

**Bug 1 — sharp horizontal cut at the top of the silhouette.** When YOLO's bbox tightly contains the head, the segmentation mask reaches row 0 of the work region. The Gaussian feather operates in the same `(rh, rw)` shape, with `BORDER_DEFAULT` (reflect) at the boundary — so the alpha values at row 0 stay near 1.0 and the composite shows a hard horizontal seam at `frame[y1, x1:x2]`. Stage 6's `BBOX_DILATION_PCT` accidentally hid this by giving the rectangle alpha the room to feather; removing the dilation in round 1 exposed it.

Fix: pad the work region outward by `dilate + feather` pixels (clipped to frame bounds), build the silhouette in the padded coordinate system with the seg_mask placed at the original-bbox offset (zeros around it), and blur+composite in the padded region. The dilation then grows into the padding zone and the feather fades to 0 over real frame pixels, no seam.

```python
pad = max(SILHOUETTE_DILATE_MIN_PX + SILHOUETTE_FEATHER_MIN_PX,
          int(short * (SILHOUETTE_DILATE_FRAC + SILHOUETTE_FEATHER_FRAC)))
silhouette = np.zeros((rh, rw), dtype=np.uint8)
silhouette[oy:oy+bh, ox:ox+bw] = (seg_mask > 0).astype(np.uint8)
```

Cost: the 199-kernel Gaussian now runs on a region with ~30–150 px more on each side (~1.3–1.5× more pixels). Acceptable.

**Bug 2 — new tracks unblurred for ~1 sec after a scene cut.** Stage 6 Finding 9 introduced per-track gender voting, but a track only votes once on creation and then again every `RECLASSIFY_INTERVAL_FRAMES = 30`. If the first vote is wrong-with-confidence (motion blur on a scene cut, partial face turn, partially-occluded face) and returns `M @ ≥ 0.5`, the rule layer reads "man → don't blur" for the full 30-frame window — exactly 1 second at 30 fps.

There's also a related candidate: a new subject appearing where a previous-scene subject's track is still IoU-matchable (`CARRY_FORWARD_FRAMES = 10`) inherits the stale track's cached gender. Same symptom, different root cause; same fix.

Fix: burst-reclassify on every frame until the vote window has at least `GENDER_VOTE_BURST = 3` entries. After that, fall back to the periodic interval. A wrong first vote gets corrected within ~3 frames (≈0.1 s) by majority. Cost: ~3 InsightFace calls per new track instead of 1 — negligible at typical track counts. The arena/mem_pattern fixes from Stage 6 Finding 6 keep the per-call heap bounded.

```python
needs_reclassify = (
    len(votes) < GENDER_VOTE_BURST
    or frame_idx - last_classify >= RECLASSIFY_INTERVAL_FRAMES
)
```

If the symptom persists after this fix, the diagnostic path is `OCCLUDE_TRACK_LOG=/tmp/track.log` on a re-run — `track_origin=new` at scene-change frames means candidate A (bad first vote), `track_origin=reused` with a stale gender means candidate B (carry-forward inheritance). Both should be addressed by burst, but the log distinguishes them if needed.

### Findings (rounds 2–3)

**Finding A — silhouette tightness traded for buffer width.** A 3 % dilation hugs the segmentation outline and reads as a paper cutout. A 5 % dilation with a 13 px floor hides the segmenter's tendency to under-segment hair/clothing edges, and the eye stops reading the alpha as a hand-traced cutout. There is a real upper bound — push further and you're back to "rectangle blur" — but the round-2 numbers are well inside it.

**Finding B — Gaussian alone cannot make the interior unrecognizable.** At kernel 199 on 1280×720 the Gaussian is already saturated; another pass widens the spread but doesn't obscure local positional detail. Pixelation is the fix. The Gaussian on top is purely cosmetic (block-edge smoothing).

**Finding C — pixelation block size scales by short-side, not absolute.** Using a fixed block size made small subjects look fine and large ones look pixel-art. Expressing it as `short_side // 18` with an 8 px floor produces consistent visual coverage across subject sizes.

**Finding D — silhouette feather needs a padded work region.** A feather kernel applied in-place at bbox dimensions produces a hard cut wherever the silhouette touches the bbox boundary — the Gaussian's reflect border preserves alpha=1 at the edge. The fix is to enlarge the working canvas, not the silhouette. Cost is a moderately bigger Gaussian region; the alternative (grow the silhouette inward) would shrink the obscured area, which is worse.

**Finding E — once-per-track gender vote is fragile on scene cuts.** Stage 6 Finding 9 added vote majority but didn't change the *frequency* of voting on new tracks. A single high-confidence wrong vote at track creation locks the rule output for a full reclassification interval. Burst-reclassifying the first N frames is the corrective; it's strictly an addition (existing periodic cadence still applies after the burst).

### Round 4 — bbox-smoothing lag at scene cuts

User feedback after the round-3 fixes: *"edges are great, but the scene-change unblur is still happening"*. Burst reclassify did its job (no more 30-frame gender lock-in), but the visible symptom — new subject unblurred for ~1 sec on cuts — persisted.

Diagnostic: ran with `OCCLUDE_TRACK_LOG=/tmp/track.log`, identified the major scene cut at frame 321 (10.7 s) via `ffmpeg scdet`, grepped frames 315–360.

Frame 319 (last frame of old shot, 6 tracks): `tid=1` was a left-side person at bbox `(0, 115, 353, 715)`.

Frame 320 (first frame of new shot, single closeup): YOLO returned a single bbox at `(77, 88, 752, 693)` — **center-frame, 1.93× the old area**. IoU with old `tid=1` = 0.346, just above `IOU_THRESHOLD = 0.3`. The matcher reused `tid=1`. The track's `bbox_history` deque has 4 stale (left-side) entries + 1 fresh (centered) entry. `_smoothed_bbox = mean(history)` → mostly the **old** position. Blur is drawn at the smoothed (old) position; the new subject's actual face/body is uncovered for the duration of the smoothing-window flush (~5 frames).

Add to that the carry-forward ghost blurs: `tid=0`, `tid=3` etc. that didn't IoU-match in frame 320 keep drawing blur ovals at their old positions for `CARRY_FORWARD_FRAMES = 10` more frames. On a fresh shot with a different layout, those land in random places. Net visible weirdness: ~5 frames of mis-positioned active blur + ~10 frames of ghost blurs from carry-forward = ~300 ms; subjectively reads as "around a second" once the brain registers the unblur.

**Fix**: per-track bbox-history reset on big jumps. When a track is IoU-matched (≥ 0.3) but its new raw bbox has IoU < `BBOX_RESET_IOU = 0.6` with the *current smoothed* bbox, the smoothing window is mostly stale — clear `bbox_history` and start fresh with just the new bbox. Gender votes, blur_history, and the rest of the track state are kept; only the position-smoothing window resets. Cheap, surgical, only fires on the scene-cut signature.

```python
if _iou(bbox, prev.last_smoothed) < BBOX_RESET_IOU:
    history = deque(maxlen=BBOX_SMOOTHING_FRAMES)
else:
    history = prev.bbox_history
```

The carry-forward ghost-blur issue is a separate symptom — the cleanest fix is to detect scene cuts at the frame level (frame-difference threshold) and drop all unmatched carry-forwards on a cut. Deferred to next round if round 4 doesn't fully resolve the user's complaint.

**Finding F — bbox smoothing assumes continuous motion; it has no notion of cuts.** Stage 6's `_smoothed_bbox = mean(last_K_bboxes)` is correct for jitter on real motion (a person walking → small frame-to-frame deltas) but pathological at scene boundaries (huge delta in one step). The 5-frame moving average is wrong for that frame even though IoU still nominally matches. Detecting the discontinuity per-track via "smoothed-IoU below threshold" is enough — no need for a video-level scene-cut detector for this specific symptom.

## 2. Performance

Stage 6 baseline: ~1.04 fps end-to-end on the laughing_people 30s clip (4–6 people on screen continuously). Per-frame breakdown:

- SegFormer-b2 forward pass: ~250–400 ms × N people = **~1.5–2 s** ← dominant
- YOLOv8n detection: ~50–100 ms
- InsightFace gender: amortized to near-zero (cached per track + reclassified every 30 frames per Stage 6 Findings 6+9)
- Blur application: ~10–30 ms

So the path forward is "make SegFormer cheaper or call it less often."

### Batched SegFormer — `Perception._segment_batch`

The processor always resizes input to 512×512, which means the batch tensor is `(N, 3, 512, 512)` regardless of crop dimensions. One forward pass over N people instead of N sequential passes. The current CUDA path also batches consecutive frames, so one YOLO call and one SegFormer call cover the union of crops from the batch. On A100/L4, `--perception-batch` is the main throughput knob.

`detect_and_segment` delegates to `detect_and_segment_batch([image])`, so single-image callers keep the same API while the video pipeline uses cross-frame batching.

### Perception batch — `--perception-batch N`

Batched perception replaced frame skipping. The video loop groups N consecutive frames, converts them to PIL once, calls `detect_and_segment_batch`, then applies tracking and blur in source order. This keeps temporal quality because every frame gets fresh detections and rules; the cost is higher peak VRAM as the crop batch grows.

Defaults: 4 on CUDA, 1 on MPS/CPU. On Colab A100/L4, benchmark 4, 6, and 8 and keep the highest value that does not OOM or reduce throughput.

### Combined effect

| Strategy | Throughput multiplier (multi-person) | Quality cost |
|---|---|---|
| Baseline (Stage 6) | 1× | — |
| + Batched SegFormer | ~1.5–2× | none |
| + Cross-frame perception batch | workload-dependent | none, unless batch OOMs |
| + CUDA I/O + GPU blur | workload-dependent | none |

Realistic ceiling on Mac Mini is still much lower than Colab because InsightFace is not fully CUDA-backed there. Long-form runs should use the CUDA path.

### What was rejected

**CUDA-first optimization.** Mac/MPS remains useful for local smoke tests, but CUDA is the primary long-run target. Production auto-selection now prefers CUDA when available.

**SegFormer-b0 distillation.** ~5× speedup but visibly worse hair / scarf / arm boundaries. The rule layer is sensitive to those (`docs/06-rule-implementation.md`). Rejected on quality.

## 3. Colab path

For 1.5-hour videos the user's tolerance ceiling is ~5 hr processing time ("less than my sleep cycle"). Mac Mini even at the optimized 3–4× ceiling is 12–15 hr. That's the boundary that pushes toward Colab.

Throughput estimates per hardware tier on this pipeline:

| Hardware | SegFormer-b2 throughput | 90 min video |
|---|---|---|
| Mac Mini MPS (Stage 6) | ~3–4 calls/sec | ~45 hr |
| Mac Mini MPS (Stage 7 optimized) | ~10 calls/sec | ~12–15 hr |
| Colab T4 (Pro tier) | ~25–40 calls/sec | ~4–6 hr |
| Colab L4 / A100 (Pro+ tier) | ~80–150 calls/sec | **~1–2 hr** |

Pro+ at 900 TL/mo is justified once the user processes long-form content (full-length TV episodes, documentaries) with any regularity.

### Architecture

The codebase is CUDA-first:
- `Perception._pick_device` returns `torch.device("cuda")` when CUDA is available; `--device cuda` hard-fails if CUDA cannot bind.
- The macOS-specific `malloc_zone_pressure_relief` block is gated by `sys.platform == "darwin"` — no-op on Linux.
- InsightFace ONNX needs `onnxruntime-gpu` instead of `onnxruntime` to use CUDA. CUDA fallback now raises instead of warning.
- The `[gpu]` extra includes `torchcodec` on non-Darwin platforms for the NVDEC fast path.
- CUDA video I/O reports its unavailable reasons when `--device cuda` is used; `--require-cuda-io` makes NVDEC/NVENC mandatory.

The notebook is still a setup wrapper, but it now invokes the CLI with `--device cuda` so accidental CPU/MPS compute fallback cannot pass silently. CUDA video I/O is runtime-dependent; use `--require-cuda-io` on GPU/FFmpeg combinations where NVDEC/NVENC must be mandatory.

### `notebooks/occlude_colab.ipynb`

Eight cells:
1. `nvidia-smi` — verify the runtime is a GPU.
2. `apt-get install ffmpeg libgl1` — system deps for opencv and the audio mux.
3. Clone the repo (placeholder URL — user replaces with their fork).
4. `pip install -r requirements.txt` + `onnxruntime-gpu insightface ultralytics psutil`.
5. `drive.mount('/content/drive')`.
6. Cache HF / InsightFace / YOLO model dirs to Drive so subsequent sessions skip the ~750 MB download. Done via `HF_HOME`, `INSIGHTFACE_HOME`, and a YOLO weight symlink.
7. Run `python -m occlude --input <drive-path> --output <drive-path> --device cuda --perception-batch 4`.
8. (One-time) copy YOLO weights to Drive after first run so cell 6's symlink resolves on next session.

### Recommended Drive layout

```
MyDrive/
  occlude/
    inputs/   <- source videos
    outputs/  <- processed results
    models/   <- auto-cached weights (HF, InsightFace, YOLO)
```

### Open questions for next session

- **Repo hosting.** Notebook currently has a placeholder `REPO_URL`. Need to push to GitHub (private fine) or document the zip-upload alternative.
- **Long-video memory profile on CUDA.** Stage 6's MPS-specific memory mitigations (per-frame `empty_cache`, ONNX arena disable, `malloc_zone_pressure_relief`) were tuned on Mac. CUDA's allocator behaves differently — should profile RSS on a 30-min run and confirm the mitigations don't actively hurt throughput on Linux.
- **Resumable processing.** A 90-min Pro+ run that crashes 80 min in is painful. Worth adding a chunk-and-checkpoint mode that ffmpeg-splits the input and runs `occlude.py` per chunk, like Stage 6's `scripts/chunked_occlude.py` but Drive-aware.

## Files this stage produced / modified

```
OCCLUDE/
├── pipeline/
│   ├── perception.py           # added _segment_batch, batched detect_and_segment
│   └── video.py                # silhouette+pixelation blur, CUDA batching/I/O
├── occlude/
│   └── cli.py                  # --device and --perception-batch flags
├── notebooks/
│   └── occlude_colab.ipynb   # GPU-runtime setup wrapper (new)
├── scripts/
│   └── test_silhouette_blur.py # before/after preview on a single image (new)
└── docs/
    ├── 07-video-pipeline.md    # tunable-constants section refreshed
    └── 08-blur-shaping-and-performance.md   # this file (new)
```

## How to reproduce

Mac, baseline current pipeline:
```bash
.venv/bin/python occlude.py --input /tmp/laughing_30s.mp4 \
  --output test_output/laughing_30s_v8.mp4
```

CUDA, with explicit batch/device:
```bash
.venv/bin/python -m occlude --input /tmp/laughing_30s.mp4 \
  --output test_output/laughing_30s_cuda.mp4 --device cuda --perception-batch 4
```

Single-image silhouette-blur preview (writes a 3-panel before/after):
```bash
.venv/bin/python scripts/test_silhouette_blur.py
```

Colab: open `notebooks/occlude_colab.ipynb` in Colab, set runtime to GPU, replace `REPO_URL`, set `INPUT` and `OUTPUT`, run all cells.
