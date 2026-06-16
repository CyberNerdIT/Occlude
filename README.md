# OCCLUDE

As a muslim, there is content I want to watch and don't. Not because it's bad content (documentaries, lectures, educational videos I'm genuinely curious about), but because they have immodestly dressed people in them, and I'd rather not sit through that. The friction is not the content, it's what's mixed into it.

I went looking for a tool that would just take the video and give me back the same video, with those people blurred. Browser extensions like HaramBlur and PordaAI process in real time, which forces brutal accuracy tradeoffs: flickering, missed frames, false positives. I needed something for offline processing, where you trade interactivity for actually getting it right.

OCCLUDE is what I built. You give it a video file, it gives you back a video file, the immodestly dressed people are blurred according to Islamic modesty rules, and the original audio is intact.

```bash
occlude --input documentary.mp4
# → documentary_occluded.mp4
```

This is not a general people-removal or object-removal tool. The blur logic checks what's visible on each person against specific Islamic modesty criteria, not just whether someone is present. I built it for muslims who run into the same friction. If that's you, you're welcome here.

## The modesty rules

The decision is binary: a person either gets fully blurred or they don't. There's no partial blur of a specific body part.

For women, the entire person is blurred if any of the following is visible: uncovered hair, bare arms, bare legs, or exposed neck and chest. Uncovered hair is the primary trigger in practice: if hair is visible, the person is blurred.

For men, the entire person is blurred if any skin in the zone between the navel and the knee is visible. Being shirtless is the primary trigger since it automatically exposes the navel. Shorts or exposed thighs are the secondary case.

People dressed modestly are never blurred: a woman in full hijab and modest clothing, a man in a t-shirt and full-length trousers. Faces of modestly dressed people, backgrounds, objects, animals, and text overlays are not touched.

## How it works

```
 input.mp4 ─▶ frame extraction
                    │
             person detection (YOLOv8)
                    │
          gender classification (per person)
                    │
        body part segmentation (per person crop)
                    │
          modesty rule check (binary decision)
                    │
     blur full bounding box if triggered
                    │
     temporal smoothing (carry flag across frames)
                    │
     video reconstruction (original audio preserved)
                    │
             output_occluded.mp4
```

YOLOv8 finds the people. A gender classifier determines which rule set applies. A body-part segmentation model identifies what's visible on each crop. The modesty rules run on that output, and if triggered, the full bounding box gets a Gaussian blur. Basic tracking carries a flag forward a few frames so the blur doesn't flicker when detection briefly drops between frames.

## Install

```bash
pip install occlude
```

If you have an NVIDIA GPU, install the GPU extra instead. It pulls in `onnxruntime-gpu` for face/gender inference and `torchcodec` for the CUDA video I/O fast path:

```bash
pip install 'occlude[gpu]'
```

Python 3.10+ required. I develop on macOS with Apple Silicon, and for anything long I run the CUDA path, usually on Google Colab. Plain CPU works but gets slow fast on anything past a short clip.

## Usage

```bash
occlude --input /path/to/video.mp4
```

Output is saved in the same directory as the input, named `video_occluded.mp4`.

Optional flags:

```bash
occlude --input video.mp4 --output cleaned.mp4   # custom output path
occlude --input video.mp4 --blur-strength 99     # Gaussian kernel size, odd number (default 199)
occlude --input video.mp4 --perception-batch 6   # batch consecutive frames for faster CUDA inference
occlude --input video.mp4 --device cuda          # require CUDA and fail if it cannot bind
occlude --input video.mp4 --device cuda --require-cuda-io  # also require NVDEC/NVENC video I/O
```

`--perception-batch` is the main long-content speed knob. It still analyses every frame, but groups consecutive frames into one YOLO + SegFormer dispatch so CUDA launch overhead and model work amortize across the batch. The default is 4 on CUDA and 1 elsewhere; on A100/L4, sweep 4, 6, and 8 on a short benchmark clip to find the memory/throughput elbow.

## Running on Google Colab

OCCLUDE is compute-intensive, and where it runs changes the experience a lot. On an NVIDIA GPU every stage runs on CUDA: person detection, body-part segmentation, the per-frame blur, and the face and gender model. On Apple Silicon the detector and segmenter use the GPU through Metal, but the face and gender model still falls back to the CPU, so a feature-length file is slow even there. On a plain CPU it gets painful well before you reach a documentary.

So for anything long, Google Colab with a GPU runtime is the path I actually use. A ready-to-run notebook is at [`notebooks/occlude_colab.ipynb`](notebooks/occlude_colab.ipynb). Open it in Colab, point it at your video, and run the cells. The notebook installs `occlude[gpu]`, repairs the `onnxruntime-gpu` install, hard-stops if any compute stage fails to bind CUDA, and reports whether CUDA video I/O is available on that runtime.

## What OCCLUDE isn't

Before you install it, a few things it doesn't do, so you don't find out the wrong way.

It's not a general people-removal or object-removal tool. If you want to blur all people regardless of dress, or remove specific objects, this is the wrong tool. The blur is tied to the modesty rules above and only triggers when they're violated.

It's not a browser extension and doesn't do real-time processing. OCCLUDE works on local video files only.

It's terminal only. No GUI, no drag-and-drop.

It doesn't take YouTube URLs or download videos. Give it a local file.

## Development stage

It works and I use it on real content, but it's still early. The accuracy is reasonable on clear cases: shirtless men, women with uncovered hair and bare arms in decent lighting. Edge cases will give you trouble: unusual angles, partial occlusion, low-resolution video, complex backgrounds.

A long documentary on a machine without GPU acceleration will take a while. Start with a short clip to confirm the output is what you need before committing to a full-length file.

If you find a consistent failure mode, open an issue with a description of the content type. That's more useful than a general accuracy complaint.

## Contributing

The parts most worth improving right now: segmentation accuracy, gender classification on difficult crops, and speed on long content. NVIDIA is covered now through the CUDA path; AMD and other accelerators aren't yet. If you want to build a frontend or optimise for a different hardware target, there's room for that too.

Open an issue or PR at <https://github.com/anaxoniclabs/OCCLUDE/issues>.

If a fellow muslim developer takes this further than I could, that's honestly the outcome I'd be happiest with.

## License

MIT
