# OCCLUDE

OCCLUDE is a command-line tool that takes a video file and writes back the same video with immodestly dressed people blurred out, audio left intact. It's for muslims who want to watch otherwise-fine content (documentaries, lectures, talks) without the immodest imagery that gets mixed into it.

I kept skipping videos I was genuinely curious about because of what was in frame, not what was being said. The friction is the imagery, not the subject, so I wanted something that hands me back the same file with the unwanted imagery blurred. Great real-time browser blockers like HaramBlur exist and are really needed, but processing every frame live forces accuracy tradeoffs (flicker, missed frames, false positives). OCCLUDE runs offline on a whole file instead, so it can take its time and get it right.

```bash
occlude --input documentary.mp4
# writes documentary_occluded.mp4 next to the input
```

## What gets blurred

The decision is per person and binary: someone is either fully blurred or not at all. There's no partial blur of a single body part. When a person is blurred, the blur traces their silhouette with a soft feathered edge rather than a hard rectangle.

For men, the whole person is blurred if skin shows between the navel and the knee. Shirtless is the main case; shorts and bare thighs are the secondary one.

For women, the whole person is blurred if any of these is visible: uncovered hair, bare arms, bare legs, or exposed neck and chest.

People dressed modestly are left as is: a woman in full hijab, a man in a t-shirt and full-length trousers. Faces of modestly dressed people, backgrounds, objects, animals, cartoon/CGI figures, and text overlays are not touched. A person who appears to be a child (under 13) is never blurred. When the judgment is genuinely uncertain about a real adult, OCCLUDE leans toward blurring rather than missing.

## How it works (v0.1 — offline, three passes)

OCCLUDE has the whole file on disk, so it does not process frame-by-frame like a live filter. It runs three passes, each free to use the whole video:

```
input.mp4
  Pass 1  detect + track   RT-DETR finds people every frame; IoU association
                           links them into per-person "tracklets", split at
                           shot cuts.
  Pass 2  judge            a vision-language model (Qwen2.5-VL) looks at each
                           tracklet's clearest frames once and returns a
                           structured verdict: real human? male/female?
                           child? modest or not?  One decision per person.
  Pass 3  render           the verdict is applied across the person's entire
                           on-screen span; SAM2 cuts a clean silhouette for
                           the people being blurred; ffmpeg muxes the audio.
output_occluded.mp4
```

Deciding **once per person and applying it across their whole span** is what removes the flicker and the "blurs a second late" problem that a streaming filter has — there is no per-frame decision to wobble, and a person flagged on frame 200 is blurred from frame 1 of that shot. Letting a model that can actually *reason* make the modesty call (instead of stacking a clothing segmenter and a brittle gender classifier) is what fixes the subtler errors: a clean-shaven man read as a woman, a CGI character treated as a person, a crowd shown from behind blurred into mud.

## Install

```bash
pip install occlude
```

You also need **ffmpeg** on your PATH (used to copy the original audio onto the output):

```bash
brew install ffmpeg        # macOS
```

For clean silhouette-shaped blur you also want **SAM2**, which isn't on PyPI:

```bash
pip install "git+https://github.com/facebookresearch/sam2.git"
```

Without SAM2, run with `--no-sam2` and the blur falls back to a soft feathered box around each person.

Python 3.10+. **A CUDA GPU is strongly recommended** — OCCLUDE runs a heavy detector, SAM2, and a 7B vision-language model. On a single A100/H100 a feature-length video takes a few hours; on CPU/Apple Silicon it is only practical for short clips. Model weights download on first run.

## Usage

```bash
occlude --input video.mp4
```

The output lands next to the input as `video_occluded.mp4`. Common options:

```bash
occlude --input video.mp4 --output cleaned.mp4              # custom output path
occlude --input video.mp4 --device cuda                     # force CUDA
occlude --input video.mp4 --blur-strength 99                # Gaussian kernel, odd, default 199
occlude --input video.mp4 --judge-batch 16                  # crops per VLM pass (throughput knob)
occlude --input video.mp4 --judge-frames 5                  # frames judged per person (accuracy knob)
occlude --input video.mp4 --judge-model Qwen/Qwen2.5-VL-7B-Instruct
occlude --input video.mp4 --detector rtdetr-x.pt            # swap the person detector
occlude --input video.mp4 --no-sam2                         # box blur instead of silhouette
```

`--judge-batch` is the main speed knob: it sets how many person-crops the VLM judges in one forward pass. `--judge-frames` trades a little speed for accuracy by judging more views of each person. `--detector` accepts any Ultralytics RT-DETR or YOLO weights.

### Running on Colab

OCCLUDE is compute-heavy, so for anything long run it on a Colab GPU runtime. `notebooks/occlude_colab.ipynb` is set up for that: open it in Colab (GPU runtime), point it at your video, run the cells. It installs `occlude`, SAM2, and the model weights.

## Known limitations

It works and is usable, but it's early (version 0.1.0) and is a ground-up rearchitecture of the earlier per-frame version.

- Accuracy is best on clear cases and worst on hard ones: tiny/distant people, heavy occlusion, low resolution, motion blur.
- It biases toward over-blurring under uncertainty, so expect occasional false positives (a modestly dressed person blurred now and then) more often than misses.
- Age estimation near the under-13 line is the least reliable signal; the child exemption is gated on the model agreeing across multiple frames.
- It's terminal only, takes a local file (no YouTube URLs), and is not a general people-blur tool — the blur only fires when the modesty rules are triggered.
- Hijab vs. uncovered-hair from behind / in profile remains the hardest case.

Start with a short clip to confirm the output is what you want before committing to a full-length file.

## Credits and license

OCCLUDE is glue around three pretrained models:

- person detection: [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) via Ultralytics
- segment + silhouette: [SAM 2](https://github.com/facebookresearch/sam2) (Segment Anything 2) from Meta
- modesty / sex / age judgment: [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

Plus PyTorch, Transformers, OpenCV, NumPy, Pillow, tqdm, and rich.

MIT licensed (note the upstream model weights carry their own licenses). Issues and PRs welcome at <https://github.com/anaxoniclabs/OCCLUDE/issues>. If a fellow muslim developer takes this further than I could, that's the outcome I'd be happiest with.
