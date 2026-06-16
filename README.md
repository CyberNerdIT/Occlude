# OCCLUDE

OCCLUDE is a command-line tool that takes a video file and writes back the same video with immodestly dressed people blurred out, audio left intact. It's for muslims who want to watch otherwise-fine content (documentaries, lectures, talks) without the immodest imagery that gets mixed into it.

I kept skipping videos I was genuinely curious about because of what was in frame, not what was being said. The friction is the imagery, not the subject, so I wanted something that hands me back the same file with the unwanted imagery blurred. Great real-time browser blockers like HaramBlur exist and really needed, but processing every frame live naturally forces accuracy tradeoffs (flicker, missed frames, false positives). OCCLUDE runs offline on a file instead, so my main goal is getting it right as much as possible, even though it takes longer.

```bash
occlude --input documentary.mp4
# writes documentary_occluded.mp4 next to the input
```

## What gets blurred

The decision is per person and binary: someone is either fully blurred or not at all. There's no partial blur of a single body part. When a person is blurred, the blur traces their silhouette with a soft feathered edge rather than a hard rectangle.

For men, the whole person is to be blurred if skin shows between the navel and the knee. Shirtless is the main case (it exposes the navel); shorts and bare thighs are the secondary one.

For women, the whole person is to be blurred if any of these is visible: uncovered hair, bare arms, bare legs, or exposed neck and chest. Visible hair is the dominant trigger in practice.

People dressed modestly are left as is: a woman in full hijab, a man in a t-shirt and full-length trousers. Faces of modestly dressed people, backgrounds, objects, animals, and text overlays are not touched.

When it can't tell (no face detected, or low-confidence gender) it leans toward blurring rather than missing: an unanchored person gets blurred, and uncertain gender falls back to the stricter female ruleset. And a person whose estimated age comes back at 12 or under is never blurred. That age signal is noisy, so the cutoff is deliberately low.

## Install

```bash
pip install occlude
```

You also need ffmpeg on your PATH; it's used to copy the original audio onto the output. On macOS:

```bash
brew install ffmpeg
```

If you have an NVIDIA GPU, install the GPU extra. It adds `onnxruntime-gpu` for the face/gender model and `torchcodec` for the CUDA video decode path:

```bash
pip install 'occlude[gpu]'
```

Python 3.10 or newer. It runs on CPU, Apple Silicon (MPS), and CUDA. On CUDA every stage runs on the GPU. On Apple Silicon the detector and segmenter use Metal but the face/gender model still falls back to CPU, so long videos are slow there too. Plain CPU gets slow fast past a short clip.

## Usage

```bash
occlude --input video.mp4
```

The output lands in the same directory, named `video_occluded.mp4`. Common options:

```bash
occlude --input video.mp4 --output cleaned.mp4         # custom output path
occlude --input video.mp4 --blur-strength 99           # Gaussian kernel size, odd, default 199
occlude --input video.mp4 --device cuda                # force CUDA, fail if unavailable
occlude --input video.mp4 --perception-batch 6         # frames per GPU forward pass
occlude --input video.mp4 --detector-model yolov8m.pt  # heavier detector, better recall
```

`--perception-batch` is the main speed knob for long content on a GPU. It still analyzes every frame, but bundles consecutive frames into one detector + segmenter pass so launch overhead amortizes across the batch. Default is 4 on CUDA, 1 elsewhere. `--detector-model` swaps the YOLO weights: the default `yolov8n.pt` is fastest, while `yolov8m.pt` or `yolov8l.pt` catch more small or odd-angle people at a throughput cost (a missed detection is a missed blur). There's also `--benchmark` (with `--seconds N`), which runs the pipeline on the first N seconds and prints fps, GPU utilization, peak VRAM, and a frame hash without writing a video. It's for tuning, not everyday use.

### Running on Colab

OCCLUDE is compute-heavy, so for anything long I run it on a Colab GPU runtime. `notebooks/occlude_colab.ipynb` is set up for that: open it in Colab, point it at your video, run the cells. It installs `occlude[gpu]`, repairs the `onnxruntime-gpu` install, and fails loudly if a stage can't bind CUDA.

## How it works

```
input.mp4
   -> YOLOv8 detects people, frame by frame
   -> SegFormer segments each person crop into body parts and clothing
   -> InsightFace estimates gender and age per person
   -> rule check decides blur / no-blur for that person
   -> silhouette-shaped pixelate + Gaussian blur if triggered
   -> IoU tracking carries the decision across frames to stop flicker
   -> ffmpeg muxes the original audio back on
output_occluded.mp4
```

The tracker matches people across frames by bounding-box overlap and votes on gender and the blur decision over a short window, so one bad frame doesn't make the blur flicker on and off. If detection briefly drops a flagged person, their blur is carried forward a few frames so it doesn't pop.

## Known limitations

It works and I use it on real content, but it's early (version 0.0.1).

- Accuracy is decent on clear cases (shirtless men, uncovered hair, bare arms in good lighting) and worse on hard ones: unusual angles, partial occlusion, low resolution, busy backgrounds.
- It biases toward over-blurring under uncertainty, so expect false positives (a modestly dressed person blurred now and then) more often than misses.
- It's terminal only. No GUI, no real-time, no browser extension.
- It takes a local file. It won't download YouTube URLs.
- It's not a general people-blur or object-removal tool. The blur is tied to the modesty rules above and only fires when they're triggered.

Start with a short clip to confirm the output is what you want before committing to a full-length file. If you hit a consistent failure mode, an issue describing the content type is more useful than a general accuracy complaint.

## Credits and license

OCCLUDE is glue around three pretrained models:

- person detection: YOLOv8 (`yolov8n` by default) via Ultralytics
- body-part and clothing segmentation: SegFormer, [`mattmdjaga/segformer_b2_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes) on Hugging Face
- face, gender, and age: InsightFace `buffalo_l`

Plus PyTorch, Transformers, OpenCV, NumPy, SciPy, ONNX Runtime, Pillow, tqdm, and rich. Model weights download on first run.

MIT licensed. Issues and PRs welcome at <https://github.com/anaxoniclabs/OCCLUDE/issues>. If a fellow muslim developer takes this further than I could, that's the outcome I'd be happiest with.
