# Stage 6 — Video Pipeline + CLI

> Status: shipped, multi-person memory-bounded and correctness-verified
> on `laughing_people.mp4` 30s clip. Both originally-failing subjects
> (bere/long-hair woman, short-hair/cleavage woman) now blur correctly.
> Memory peak ~8 GB with per-track gender voting enabled (well under
> 24 GB Jetsam threshold). The full chain of findings below — most of
> the early ones turned out to be wrong tracks; the load-bearing
> fixes are **6** (gender cache), **8** (cross-check threshold), and
> **9** (per-track voting).

## What was built

`pipeline/video.py` exports `VideoProcessor`. The class loads
`Perception` and `RuleEngine` once, then `process(input_path,
output_path)` runs the per-frame loop:

1. `cv2.VideoCapture` reads BGR ndarray.
2. Convert to RGB → PIL.Image → `Perception(pil)` → list of `Person`.
3. For each `Person`: `RuleEngine.decide(person)` → `Decision`.
4. **IoU-matched track update** with 5-frame bbox moving average per
   track (`_smoothed_bbox`).
5. **Apply blur** to the smoothed bbox: dilate by 15 %, Gaussian-
   blur the dilated region with kernel 99 (default), then composite
   back through a feathered alpha mask. Removes the rectangular
   bbox edge that the kernel-51 hard-clamped version produced.
6. **Carry-forward**: unmatched tracks blur at last smoothed bbox
   for K = 2 frames before being dropped (per `OCCLUDE_SPEC.md`
   §Step 7).
7. `cv2.VideoWriter` (mp4v) writes a silent intermediate.
8. After the loop, ffmpeg muxes the original audio onto the silent
   processed video. Optional-stream syntax (`-map 1:a:0?`) handles
   no-audio inputs cleanly without a fallback dance.

`occlude.py` is the CLI: `--input`, `--output` (default
`<stem>_occluded.mp4`), `--blur-strength` (default 199). Validates
input exists, kernel is odd, ffmpeg is on PATH. Catches
KeyboardInterrupt for exit 130.

`scripts/chunked_occlude.py` orchestrates subprocess chunking:
ffmpeg-splits the input into N-second chunks, runs `occlude.py` in
a fresh Python interpreter per chunk, then ffmpeg-concats the
results. Each subprocess pays ~5 s of model-init overhead but in
exchange the heap and macOS VM accounting reset between chunks.

## Tunable constants (pipeline/video.py)

```
IOU_THRESHOLD             = 0.3   # match a current detection to a previous track
CARRY_FORWARD_FRAMES      = 10    # K — bumped from spec's 2 (laughing_people flicker)
BBOX_SMOOTHING_FRAMES     = 5     # per-track moving average window
BBOX_RESET_IOU            = 0.6   # smoothed-IoU below this → reset history (scene-cut signal)
DEFAULT_BLUR_KERNEL       = 199   # ~15 % of frame width at 1280p
BLUR_VOTE_WINDOW          = 10    # per-track blur-decision smoothing window
BLUR_VOTE_MIN_HISTORY     = 3     # require this many votes before majority kicks in
RECLASSIFY_INTERVAL_FRAMES= 30    # InsightFace re-runs at this cadence per track
GENDER_VOTE_WINDOW        = 5     # rolling vote over last K classifications
GENDER_VOTE_BURST         = 3     # new tracks reclassify every frame until this many votes
MEMORY_CLEANUP_EVERY      = 1     # every frame, after Finding 1
SILHOUETTE_DILATE_FRAC    = 0.05  # see docs/08
SILHOUETTE_DILATE_MIN_PX  = 13
SILHOUETTE_FEATHER_FRAC   = 0.15
SILHOUETTE_FEATHER_MIN_PX = 25
PIXELATE_BLOCKS_ACROSS    = 18    # ~18 mosaic blocks across the short side
PIXELATE_MIN_BLOCK_PX     = 8
```

`BBOX_DILATION_PCT` was removed in Stage 7 (silhouette mask replaces it; nothing left to hide). `DEFAULT_BLUR_KERNEL` was 51 → 99 → 199 across rounds of user feedback. See `docs/08-blur-shaping-and-performance.md` for the shape/intensity redesign and the perf work that came with it.

## Findings

### Finding 1 — SegFormer leaks ~40 MB per call on MPS

The Stage 4 `_segment()` originally read:

```python
def _segment(self, crop):
    inputs = self.seg_processor(images=crop, return_tensors="pt").to(self.device)
    with torch.no_grad():
        outputs = self.seg_model(**inputs)
    logits = outputs.logits.cpu()
    upsampled = torch.nn.functional.interpolate(logits, size=crop.size[::-1], ...)
    return upsampled.argmax(dim=1)[0].numpy().astype(np.int32)
```

`outputs` is a HuggingFace `SemanticSegmenterOutput` dataclass.
`outputs.logits` is an MPS-resident tensor. `.cpu()` returns a CPU
*copy* but the original MPS tensor stays alive — referenced by
`outputs` until the next call rebinds the local. On Apple Silicon's
MPS allocator, the underlying GPU memory is only released when the
tensor is garbage-collected *and* `torch.mps.empty_cache()` is
called.

For a 1280×720 input, `outputs.logits` is `[1, 18, H_eff, W_eff]`
floats = ~66 MB. Without the `del`, this grows monotonically across
frames. macOS jetsam SIGKILL'd the process at frame 1290–1311 of a
1831-frame video on a 24 GB Mac mini in three consecutive runs.

**Fix in `pipeline/perception.py:_segment`:**

```python
logits_cpu = outputs.logits.detach().to("cpu")
del outputs, inputs
# ... interpolate, argmax, numpy ...
del logits_cpu, upsampled
return result
```

Combined with `gc.collect() + torch.mps.empty_cache()` every frame
(`pipeline/video.py:MEMORY_CLEANUP_EVERY = 1`), this kept RSS
bounded at 3.5 GB peak across the 451-frame, single-person, 15s
test clip. **Cost:** ~70 % slower per frame (0.36 s → 0.60 s) due
to per-call empty_cache + dilation/feather composite. Acceptable for
v1 where "real-time is not a goal" per spec.

### Finding 2 — Activity Monitor "Memory" ≠ RSS

During debugging, `psutil.Process().memory_info().rss` reported
2.5–8 GB while macOS Activity Monitor's "Memory" column showed
42 GB and "VM (Compressed)" showed 39 GB.

Activity Monitor's "Memory" is the process's *memory footprint* —
RSS plus the process's contribution to compressed-memory pages.
macOS aggressively compresses inactive memory pages to keep more
"alive" in physical RAM. On 24 GB hardware running a long ML
workload, compressed VM can swell past 4–5 × the actual working
set.

This matters because:
- A bounded RSS does not mean a bounded *footprint*.
- Compression itself has CPU cost — the per-frame timing degraded
  from 1.5 s to 3.6 s as compression overhead grew, even before the
  process died.
- Jetsam considers footprint, not RSS, when deciding what to kill.

**The implication:** psutil-only memory diagnostics are misleading
on macOS. The real signal was Activity Monitor reporting GBs of
compressed VM while psutil RSS looked healthy.

### Finding 3 — Multi-person frames amplify the leak ~6×

The single-person 15s test (`human_reactions_15s.mp4`) showed at
most 1–2 tracks per frame and stayed at ~3.5 GB peak RSS. The
1-min `laughing_people.mp4` showed 6–9 tracks per frame (stock
footage of multiple laughing people) and hit 42 GB footprint at
frame ~200.

Per-frame SegFormer cost scales with detected persons:

| Tracks/frame | Logits per frame | Allocations per 1831 frames |
|---|---:|---:|
| 1 | 66 MB | 121 GB total cumulative |
| 6 | 396 MB | 725 GB total cumulative |

Even with proper `del` and `empty_cache`, the *allocator* keeps the
address space mapped. Each new allocation finds free space inside
already-mapped pages, but the address space high-water mark only
grows. macOS sees the high-water mark, compresses the inactive
range, and eventually thrashes.

### Finding 4 — Subprocess chunking helps but doesn't suffice at full resolution

`scripts/chunked_occlude.py` splits the input into N-second chunks
and runs `occlude.py` in a fresh subprocess per chunk. On the 15s
single-person clip with 5s chunks → 3 subprocesses → clean output
with intact AAC audio.

On `laughing_people.mp4` with 10s chunks → first chunk (320 frames)
**still blew up to 30+ GB at frame 200**. Subprocess isolation
prevents *cross-chunk* drift but doesn't help if a single chunk's
working set is too big for the hardware.

Going to smaller chunks (5s → ~150 frames, 2s → ~60 frames) would
eventually fit, but 1) Python startup is ~5 s per chunk, so 30+
chunks for a 1-min video starts to dominate runtime, 2) the per-
frame allocation rate of ~150 MB on multi-person frames means even
60-frame chunks risk hitting 9 GB resident before exit.

### Finding 6 — Actual root cause: InsightFace ONNX, not MPS / not allocator

**ELI5.** Imagine your video has 6 people in it, and for each person
in each frame you ask a face-detection helper "is this a man or a
woman?" That helper has a memory leak — every time you ask, it
quietly keeps a small pile of scratch paper on its desk and never
throws any away. After thousands of questions the desk is buried
under 30 GB of paper and your computer kills the program. The old
code asked the helper the same question about the same person 1,800
times (once per frame). The fix: ask once per person, write the
answer on a sticky note (the IoU-matched track), and just read the
sticky note for every following frame. Same 6 people, same answer
in every frame — no need to re-ask. The helper now gets ~6
questions total instead of ~5,400, the scratch paper stays small,
and the program finishes cleanly.


`vmmap` on a live multi-person run revealed the leak's actual nature.
At frame 25 of `laughing_people.mp4`:

```
MALLOC_LARGE   4.6G virtual   278M dirty   4.3G compressed   607 regions
MALLOC_SMALL   3.4G virtual   127M dirty   3.2G compressed   862 regions
IOAccelerator  17M           (MPS — fine)
```

Only ~400 MB of unique data; the rest is fragmented compressed
pages from CPU heap allocations that never returned to the kernel.
**The MPS allocator was not the problem.** Finding 1's `del +
empty_cache` change worked at single-person scale only because
allocation rate was low enough not to fragment within 451 frames.

The discriminating test: stub `_classify` to return `("F", 0.0)`
without calling InsightFace. Footprint plateaus at 2.4 GB across
30 frames (vs 13 GB and rising at frame 39 with InsightFace). The
ONNX face-detection sessions inside `buffalo_l` were retaining
~45 MB/person-call across some combination of arena, mem_pattern,
and model-internal bookkeeping that no `SessionOptions` flag we
tried (`enable_cpu_mem_arena=False`, `enable_mem_pattern=False`)
materially reduced.

**Things that did not work** (ranked by how plausible each looked
beforehand):

1. `torch.mps.empty_cache()` per call — addresses the wrong device.
2. `gc.collect()` per call — leak isn't GC-reachable garbage.
3. `malloc_zone_pressure_relief(0, 0)` per frame — returned 0
   bytes; pages are held by ORT, not free-but-cached.
4. `enable_cpu_mem_arena = False` on each InsightFace ORT
   session — verified applied (`session.get_session_options()`
   confirmed False), no measurable effect.
5. `enable_mem_pattern = False` — composes with the above, no
   effect.
6. `allowed_modules=["detection", "genderage"]` (drop landmark +
   recognition models) — modest help: ~265 MB/frame → ~190 MB/frame.
   Kept anyway as cheap optimization.
7. Subprocess chunking via `scripts/chunked_occlude.py` — works
   in principle but `ffmpeg -c copy -segment_time 2` only splits
   at keyframes; on real video that means 4–6 s segments
   (120–180 frames), peaks still 25+ GB.

**The actual fix: cache gender per IoU-tracked person.**

Architecturally: split `Perception.__call__` into
`detect_and_segment(image)` (called every frame) and
`classify(crop)` (called once per new track). `_TrackedPerson`
gains `gender` + `face_det_score` fields. The video loop matches
detection bbox to existing track first; on match, copies cached
gender; on miss, runs `classify` once.

```python
# pipeline/video.py inside the per-frame loop
people = self.perception.detect_and_segment(pil)
for person in people:
    bbox = ...
    best_id = ... (IoU match against tracks)
    if best_id is None:
        person.gender, person.face_det_score = (
            self.perception.classify(person.crop)  # InsightFace ONCE
        )
    else:
        person.gender = tracks[best_id].gender
        person.face_det_score = tracks[best_id].face_det_score
    decision = self.rules.decide(person)
    ...
```

For `laughing_people.mp4` (6 stable persons), this drops InsightFace
calls from ~11,000 to ~6 across the whole video. Footprint plateaus
in the 1.7–2.7 GB range with no upward trend across 60+ frames.

Trade-off: gender determined once per track persists for that
track's lifetime. The cross-check rule (`predicted M + Hair below
Face → F override`) still runs every frame against the cached
gender, so a misclassified male long-hair gets corrected. But a
genuinely uncertain initial classification can't be re-tried later
in the track. Acceptable for v1; a future Stage 7 could add
periodic re-classification (every Nth frame on tracks with
`face_det_score < threshold`) at a bounded leak cost.

The argmax-before-upsample fix in `_segment` (Finding 7 below) is
also kept — it shaves ~22 MB/crop of float32 bilinear-upsample CPU
allocation. Modest win on top of the gender cache; redundant if
you also have the cache.

### Finding 8 — Cross-check threshold was too strict for bere/beanie

The Stage-5 cross-check (`predicted M + Hair below Face → F override`)
fires when `hair_below_face_count > k * face_area`. The doc-original
`k=0.5` was tuned on the 7-image set, where image 05 (bald man) had
near-zero below-face hair. But on `laughing_people.mp4`'s foreground
woman wearing a bere with long hair flowing past her shoulders, the
ratio was 16275/67500 = 0.24 — clearly visible long uncovered hair,
yet just shy of the 0.5 threshold. Cross-check didn't fire → her
track stayed M (InsightFace had returned M @ 0.91 on her) → male
ruleset applied → never blurred.

Lowered to `k=0.20`. Verified against the 7-image set: image 04
(shirtless M, short hair) and image 05 (bald M) both have
essentially no hair below the face label, so they don't trip it.
Trade-off: a hypothetical long-haired *clothed* man (not in the
test set) would now flip to F and over-blur. Acceptable v1 bias —
the spec already over-blurs on uncertainty.

### Finding 9 — Per-track gender voting (the actual classifier-cache fix)

Finding 6 cached one InsightFace classification per IoU-tracked
person to bound memory. That worked, but locked in any wrong first
call for the entire track lifetime. Two failure modes both
manifested on `laughing_people.mp4`:

1. **Bbox stitching.** YOLO at frame 81 returned a single huge
   bbox `(0, 78, 701, 713)` covering ~half the frame width. That
   created `tid=18` with a "magnet" bbox that subsequently matched
   any person in the left half of the frame via IoU > 0.3. The
   tracked content effectively changed identity over time, but the
   cached gender (M @ 0.91) didn't.

2. **Confident misclassification.** The rooftop-scene woman
   (short hair, cleavage) was first classified M at high
   confidence — likely a face-angle / occlusion fluke (drinking
   from a glass on the first detection). High confidence meant
   the cache wouldn't be discarded by any score-threshold logic.

**Fix: rolling vote with periodic re-classify.** Each
`_TrackedPerson` carries a `gender_votes: deque(maxlen=5)`. Every
`RECLASSIFY_INTERVAL_FRAMES = 30` (~1s at 30fps), the track is
re-classified and the result appended. The active gender is
majority of high-confidence votes (ties broken by highest score).
For a 820-frame track, that's ~27 InsightFace calls instead of 1
— still bounded.

Result on `laughing_people.mp4` 30s clip:
- `tid=18` accumulated **60 M votes vs 760 F votes** across its
  lifetime → flipped to F → blur% rose from 0% → 93%.
- Bere woman: blurred from frame 82 onward via the Finding 8
  cross-check (didn't even need the vote to flip).
- Cleavage woman: blurred from her first re-classify (around
  frame 30 of her track) once the vote majority shifted to F.

Cost: peak footprint rose from 4.8 GB (cache-once) to ~8 GB
(with voting). Still well under the 24 GB Jetsam threshold.
Wall time on 30s clip went 12m17s → 13m32s (+10%).

### Finding 7 — Argmax-before-upsample in `_segment`

`SegformerImageProcessor.from_pretrained('mattmdjaga/segformer_b2_clothes')`
defaults to `do_resize=True, size={'height': 512, 'width': 512}`,
so logits are constant-shape `(1, 18, 128, 128)` ≈ 1.2 MB
regardless of input crop size. The pre-fix code bilinear-upsampled
those logits at original crop resolution on CPU, producing an
`18 × H × W × float32` tensor (up to 22 MB on a 600×1500 crop).

Argmax on the small `(1, 18, 128, 128)` tensor *on the device*
collapses 18 channels → 1 before upsampling. `cv2.resize(...,
INTER_NEAREST)` then upsamples a single-channel int map at the
crop size — semantically equivalent for the coverage-ratio +
connected-component rules in `pipeline/rules.py` (no sub-pixel
boundary logic, verified by re-running `scripts/test_rules.py`:
7/7 expected decisions preserved).

This is in `pipeline/perception.py:_segment`. Not load-bearing
after Finding 6 lands, but kept as cheap insurance.

### Finding 5 — Downsize SegFormer input (rejected)

The original plan: resize each crop to ≤512 px before SegFormer
to reduce per-call allocation. **Rejected after probing**:
`SegformerImageProcessor` already resizes input to 512×512
internally (`do_resize=True`), so pre-resizing is a no-op. The
real CPU memory hog was the post-forward upsample tensor
(addressed by Finding 7), not the input pixel buffer.

### (original) Finding 5 — Real fix: downsize SegFormer input (not implemented)

SegFormer (`mattmdjaga/segformer_b2_clothes`) was trained on
512×512 inputs and uses learned positional encodings that
interpolate to other sizes. Feeding it variable-sized full crops
(common case: 600×1500 px for a person silhouette) makes the
logits proportionally large.

Resizing each crop to max 512 px on the longest side before
SegFormer would give logits of ~9 MB instead of 66 MB — a **7×
reduction in per-call allocation**. With 6 persons/frame this
brings the per-frame budget from 396 MB to 56 MB, comfortably under
any hardware threshold.

Sketch (not yet in the codebase):

```python
def _segment(self, crop):
    max_dim = 512
    scale = min(1.0, max_dim / max(crop.size))
    work = crop if scale == 1.0 else crop.resize(
        (int(crop.width * scale), int(crop.height * scale)),
        Image.BILINEAR,
    )
    inputs = self.seg_processor(images=work, return_tensors="pt").to(self.device)
    with torch.no_grad():
        outputs = self.seg_model(**inputs)
    logits_cpu = outputs.logits.detach().to("cpu")
    del outputs, inputs
    # Upsample directly to ORIGINAL crop size (not the resized work size)
    upsampled = torch.nn.functional.interpolate(
        logits_cpu, size=crop.size[::-1], mode="bilinear", align_corners=False,
    )
    result = upsampled.argmax(dim=1)[0].numpy().astype(np.int32)
    del logits_cpu, upsampled
    return result
```

Trade-off: segmentation quality on very tall crops (full-body
shots) may degrade because fine details (hair texture, hand vs
arm boundary) are lost at 512 px before being upsampled back. Need
to validate against the 7-image rule test before shipping. None
of the existing test images are stretched enough for this to clearly
hurt; would want to add a tall-figure test image first.

## What was verified end-to-end

- `human_reactions_15s_occluded.mp4` — 451 frames, single person,
  audio preserved (AAC stream copied), duration 15.00s vs input
  15.07s, exit 0, RSS bounded.
- The new dilated/feathered blur visibly removes the hard
  rectangular edges from the kernel-51 hard-clamped version.
- bbox smoothing (5-frame moving average per track) removes the
  per-frame jitter that produced "focus breathing".
- `scripts/chunked_occlude.py` round-trip on the 15s clip with
  5s chunks → 3 chunks processed → concat with audio intact.
- CLI argument validation: missing input, even kernel, missing
  ffmpeg → clean exit codes + stderr messages.

## What did *not* work and why

| Attempt | Outcome | Why it failed |
|---|---|---|
| Periodic `gc.collect() + torch.mps.empty_cache()` every 50 frames | OOM at same frame range | The cleanup *did* run; the issue is allocator-pages-mapped, not GC-reachable garbage |
| Per-frame `gc.collect() + torch.mps.empty_cache()` | OOM at same frame range on multi-person | Same reason; allocator never returns pages to OS |
| `del outputs, inputs` in `_segment` (Finding 1 fix) | Fixed single-person case | Helps but per-frame allocation still huge for multi-person |
| Subprocess chunking (Finding 4) | Helps cross-chunk, fails within-chunk | Chunk's own working set still too big at full resolution |

## Open issues / known v1 gaps

> Stage 7 closed several of these. See `docs/08-blur-shaping-and-performance.md` for the silhouette-blur redesign, batched perception, CUDA processing, and the Colab path for long-form video.

- **Multi-person high-resolution video doesn't process on 24 GB
  hardware** without Finding 5's downsize fix. Documented gap.
- **Per-frame timing is 0.6 s** on 1280×720 single-person at default
  blur kernel. A 1-hour video at 30 fps takes ~9 hours. Spec
  accepts this ("real-time is not a goal") but the user reaction —
  reasonable — is that 12× slower than realtime is too slow even by
  that bar. Frame-stride (run perception every Nth frame, propagate
  decisions through the existing carry-forward) is the deferred
  Stage 7 win — **closed in Stage 7** via batched SegFormer and the CUDA path.
- **Rule threshold fragility** documented in
  `docs/06-rule-implementation.md` Finding 2 surfaced again in the
  test-video pipeline: image 06 (modest abaya woman) flipped from
  *pass* to *blur* purely because the test-video letterboxing
  changed YOLO's bbox proportions enough to alter the SegFormer
  Hat+Scarf:Hair ratio from 0.67 to 0.21. Not a Stage 6 bug — but
  it confirms the existing prediction that thresholds need
  re-tuning on real video data.
- **`pbar.close()` lives inside the same try block as the loop** in
  `pipeline/video.py`; if the loop throws, pbar leaks. Cosmetic.

## Files this stage produced

```
OCCLUDE/
├── occlude.py                      # CLI entry point
├── pipeline/
│   ├── perception.py                 # _segment() updated for Finding 1
│   └── video.py                      # VideoProcessor + tracking + blur + mux
├── scripts/
│   ├── create_test_video.py          # synthetic test-video generator
│   └── chunked_occlude.py          # subprocess-chunking orchestrator
├── requirements.txt                  # added scipy, tqdm
└── docs/
    └── 07-video-pipeline.md          # this file
```

## How to reproduce

Single-person 15s test (works):
```bash
.venv/bin/python occlude.py --input test_video_real/human_reactions_15s.mp4
```

Multi-person 1-min test (will OOM at full resolution; Finding 5
needs to land first):
```bash
.venv/bin/python scripts/chunked_occlude.py \
  --input test_video_real/laughing_people.mp4 --chunk-seconds 10
```

Memory diagnostic mode (writes per-25-frame RSS to a separate log
that survives tqdm's progress-bar redraws):
```bash
OCCLUDE_MEM_LOG=/tmp/mem.log .venv/bin/python occlude.py --input <file>
tail -f /tmp/mem.log
```
