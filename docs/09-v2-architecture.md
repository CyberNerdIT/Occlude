# Stage 9 — v2 Rearchitecture: Offline, Three-Pass, VLM-Judged

> Status: implemented (v0.1.0). Replaces the per-frame streaming pipeline
> (Stages 4–8) with an offline, multi-pass design. The pure-logic layers are
> unit-tested on CPU; the model layers (RT-DETR, SAM2, Qwen2.5-VL) are
> verified on a CUDA runtime, not on the dev laptop.

## Why this stage exists

v1 worked, but `docs/Current-State.md` catalogued failures that were not
incidental bugs — they were structural. Mapping each to its root cause made
the split clear:

| Observed failure | Root cause | Class |
|---|---|---|
| Clean-shaven young man repeatedly blurred | InsightFace flips beardless males → F, then "uncertain → female" | model (semantic) |
| Fully-visible people at frame edges missed | YOLOv8n (nano) recall | model (detector) |
| Person with no visible face blurred anyway | "no face anchor → blur" policy | policy |
| Crowds from behind → muddy whole-frame blur | SegFormer sees hair + no face → default blur | policy + model |
| Head-on shots blur late and flicker | per-frame decision + weak IoU tracker | architecture (temporal) |
| Kids over-blurred | InsightFace age noise | model (age) |
| Non-human CGI figure blurred | YOLO false "person" + default blur | model + policy |

Two families: **temporal/structural** (flicker, late blur, edge misses) and
**semantic** (sex, age, human-ness, scene reasoning). v1's design — a stream
of three brittle single-purpose heads, smoothed after the fact by an IoU
tracker, defaulting to blur on any uncertainty — was wrong for *both*.

The key realisation (the user's): **the pipeline acted real-time, but OCCLUDE
is offline.** It has the whole file. A streaming shape was paying a tax for
nothing.

## The shape: three passes

```
Pass 1  detect + track   →  list[Tracklet]   (boxes only; memory-bounded)
Pass 2  judge            →  one Verdict per Tracklet
Pass 3  render           →  blurred video + original audio
```

- **Pass 1** (`detect.py`, `track.py`, `scenes.py`). RT-DETR detects people
  every frame (offline, we no longer trade recall for speed — and a missed
  detection is a missed blur). `TrackletBuilder` links detections across
  frames by IoU into one identity per person per shot; `ShotSegmenter` splits
  tracklets at shot cuts so no identity (or mask) is carried across a scene
  change. Only boxes are stored.

- **Pass 2** (`judge.py`, `decide.py`). For each tracklet, `Tracklet.best_frames`
  picks its largest, most-confident views — the offline luxury of choosing the
  *clearest* evidence instead of whatever frame a streaming decision fell on.
  A VLM (Qwen2.5-VL) returns structured JSON per crop; `decide.aggregate`
  combines the samples into one verdict with the policy priority
  **non-human > child > over-blur-on-tie**.

- **Pass 3** (`video.py`, `blur.py`, `track.py:SAM2Segmenter`). The verdict is
  applied to *every* frame the tracklet spans. SAM2 cuts a clean silhouette —
  only for the people actually being blurred, computed lazily so the common
  (modest) case never pays for segmentation. ffmpeg muxes the original audio.

## Why each failure is addressed

- **Flicker / blurs late.** Gone by construction. There is no per-frame
  decision to wobble, and the verdict back-applies to the tracklet's whole
  span — a person flagged on frame 200 is blurred from frame 1 of the shot.
  This deletes v1's gender-vote windows, blur-vote smoothing, carry-forward,
  bbox moving-averages, and re-classify cadence — all of which existed only to
  fake global knowledge from a causal view.
- **Edge / back-facing misses.** RT-DETR's global attention recovers the edge
  and partial people YOLO-nano dropped.
- **Sex flip, CGI figure, crowd-from-behind, no-face.** A model that *reasons*
  replaces the clothing-segmenter + gender-head stack. It judges "real human?",
  "male despite no beard?", "crowd from behind with no skin → not a trigger"
  holistically, and the dumb "no face → blur" default is gone.
- **Kids.** The VLM's age bracket is materially better than InsightFace's age
  head; the exemption is gated on a multi-frame majority so a single
  mislabeled frame can't let an immodest adult escape.

## Decisions worth recording

- **RT-DETR, not Co-DETR.** Co-DETR has higher recall but needs MMDetection /
  mmcv, an install that isn't worth it for v1. RT-DETR ships via the existing
  Ultralytics dependency. The detector is swappable (`--detector`); Co-DETR is
  a documented upgrade path.
- **SAM2 image predictor + IoU association, not SAM2-*video* propagation.**
  Full video propagation (chunked, with cross-chunk re-id) is the heavier,
  higher-ceiling option. v1 uses per-frame detect + IoU identity + per-box SAM2
  masks: simpler, robust, and it still delivers clean silhouettes and per-track
  verdicts. SAM2-video is a documented upgrade. Note the distinction from v1:
  IoU here does *identity association only* — the thing we criticised was using
  IoU to smooth *decisions*, which this design doesn't do.
- **Geometric child backstop: rejected.** A small bbox means "far from camera"
  as often as "child", so a size rule would wrongly exempt distant adults — the
  exact error the over-blur policy exists to avoid. We trust the VLM age
  bracket (gated on a majority) instead. See `decide.py`.
- **VLM only for judgment, code for everything else.** Tracking, masking,
  blurring, muxing are deterministic and stay in code; the VLM makes only the
  classification call. This is the project's "use the model for judgment, not
  for what code can answer" rule, satisfied by construction.

## What's verified where

- Pure logic — `tracklets.py`, `scenes.py`, `decide.py`, `track.py`
  (association), `judge.py` (JSON parsing), `detect.py` (result parsing),
  `video.py` (full 3-pass orchestration with fake models) — unit-tested on
  CPU (`tests/`).
- Model behaviour — RT-DETR recall, SAM2 mask quality, Qwen2.5-VL judgment
  accuracy on the real failure clips — must be measured on a CUDA runtime.
  That measurement is the next step, not a claim this stage makes.

## Open / next

- Empirical bake-off on the real failure scenes (Demis gender flip, edge-miss
  lady, crowd-from-behind, CGI figure, kids) to tune `DEFAULT_CONF`,
  `judge_frames`, the shot-cut threshold, and the aggregation fractions.
- Upgrade paths if the bake-off demands them: Co-DETR detector, SAM2-video
  propagation, a newer VLM (Qwen3-VL), or a hybrid where cheap specialist
  models pre-filter and the VLM only adjudicates the uncertain crops.
- Restore a benchmark mode and an eval harness against the new pipeline.
