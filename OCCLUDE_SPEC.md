# OCCLUDE — Project Specification

## What Is This

OCCLUDE is a CLI tool that takes a video file as input, detects people who are immodestly dressed according to Islamic modesty rules, blurs them entirely, and outputs a clean video file with the original audio preserved.

It is the visual equivalent of ELUATE. Same UX pattern: one input, one output, progress visible, no intermediate steps the user has to manage.

-----

## Motivation

Existing tools like HaramBlur and PordaAI operate as browser extensions doing real-time processing. Real-time forces brutal accuracy tradeoffs — they blur incorrectly, miss content, or flicker. OCCLUDE solves a different problem: offline processing of video files (documentaries, lectures, educational content) that contain valuable information but also immodest content. Process once, watch cleanly forever.

-----

## CLI Interface

Modeled after ELUATE. Simple, minimal.

```
occlude --input /path/to/video.mp4
```

Output file saved in same directory as input with a suffix:

```
video_occluded.mp4
```

Optional flags (implement after core works):

```
--output /path/to/output.mp4     # custom output path
--blur-strength 25               # gaussian blur kernel size, default 51
```

Show a progress bar during processing indicating current frame / total frames and estimated time remaining.

-----

## Modesty Rules (Core Logic)

This is the decision engine. It is binary — either a person gets fully blurred or they don’t. There is no partial blurring of specific body parts.

### For Women

Blur the entire person if **any** of the following is visible:

- Hair (uncovered)
- Arms (bare)
- Legs (bare)
- Neck / chest area
- Anything beyond the eye area

**Decision rule:** If the model detects any exposed area that is not the eyes on a female-presenting person → blur the whole person bounding box.

In practice: uncovered hair is the primary and most reliable trigger. If hair is visible, blur. Arms and legs are secondary confirmation signals.

### For Men

Blur the entire person if **any** of the following is visible:

- Navel or area below navel (shirtless automatically exposes this)
- Groin area
- Thighs
- Area above the knee

**Decision rule:** The zone between the navel and the end of the knee must be fully covered. Any visible skin in that zone → blur the whole person bounding box.

In practice: shirtless is the primary trigger for men since it automatically exposes the navel. Shorts or exposed thighs are the secondary trigger.

### What Does NOT Get Blurred

- A woman in full hijab and modest clothing
- A man in a t-shirt and full-length trousers
- Faces of modestly dressed people
- Backgrounds, objects, animals, text overlays
- Anyone outside the frame partially (only blur if enough is visible to make a reliable determination)

-----

## Technical Pipeline

### Step 1 — Frame Extraction

Extract frames from the input video using OpenCV or FFmpeg. Process at original frame rate. Store frames in memory or a temp directory.

### Step 2 — Person Detection

Run YOLOv8 nano or small on each frame to get bounding boxes around each detected person. This narrows the region passed to the heavier segmentation model, improving speed.

Model: `yolov8n.pt` or `yolov8s.pt` from Ultralytics.

### Step 3 — Gender Classification

For each detected person bounding box, classify gender (male / female). This determines which rule set to apply.

Use a lightweight gender classification model on the cropped person bounding box.

### Step 4 — Body Part Segmentation

Run a human parsing / semantic segmentation model on each person crop to identify what body parts and clothing are visible.

Models to test in this order:

1. **FASHN-Human-Parser** — fine-tuned for fashion, outputs granular body part + clothing labels
1. **DeepLabV3+ ResNet-50-human** — reliable segmentation for human body parts

The segmentation output should provide pixel-level labels including: hair, face, arms, legs, torso, and clothing categories (top, pants, skirt, dress, sleeve, etc.)

**Important:** Test these models on static images first before integrating into the video pipeline. Confirm their label output is granular enough to support the rule logic above before building anything around them.

### Step 5 — Rule Application

Apply the modesty rules (defined above) to the segmentation output for each detected person.

If the rules are triggered → flag this person’s bounding box for blurring in this frame.

### Step 6 — Blur Application

For each flagged bounding box, apply Gaussian blur using OpenCV:

```python
cv2.GaussianBlur(region, (51, 51), 0)
```

Apply blur to the full bounding box of the person, not just the specific body part that triggered the rule.

### Step 7 — Temporal Smoothing

To prevent flickering (blur appearing and disappearing frame to frame for the same person), apply basic tracking across frames. If a person was flagged in frame N, they should remain flagged in frames N+1 and N+2 even if detection momentarily fails.

Use a simple confidence carry-forward approach — don’t require re-detection every single frame.

### Step 8 — Video Reconstruction

Reconstruct the processed frames back into a video file. Preserve the original audio track exactly. Use FFmpeg for this.

Output: same resolution, same frame rate, same format as input, original audio intact.

-----

## Hardware Context

Primary development machine: M4 Mac mini, 24GB RAM.

The M4’s Neural Engine accelerates CoreML-converted models. Consider CoreML conversion of YOLOv8 and segmentation models for significantly faster inference on Apple Silicon.

Target performance: fast enough to process a 1-hour video in a reasonable offline timeframe. Real-time is not a goal.

-----

## What to Build First

Do not build the full pipeline immediately. Follow this order:

1. **Model testing script** — A simple Python script that takes a single image, runs the segmentation model, and prints/visualizes the output labels. Confirm the model outputs what we need.
1. **Single frame proof of concept** — Take one video frame, run the full detection → rule check → blur logic on it, output a single processed image. Confirm the blur is applied correctly.
1. **Multi-frame video test** — Run the pipeline on a short 30-second clip. Check for flickering, accuracy, and output quality.
1. **Full pipeline** — Integrate progress bar, audio preservation, output file naming, and handle edge cases.

-----

## Project Structure

```
occlude/
├── occlude.py          # main CLI entry point
├── pipeline/
│   ├── detector.py       # person detection (YOLO)
│   ├── classifier.py     # gender classification
│   ├── segmenter.py      # body part segmentation
│   ├── rules.py          # modesty rule logic
│   ├── blur.py           # blur application
│   └── video.py          # frame extraction and video reconstruction
├── models/               # downloaded model weights
├── requirements.txt
└── README.md
```

-----

## README Requirements

The README must clearly explain:

- What OCCLUDE is and why it exists
- Installation steps
- Basic usage example
- The modesty rules it applies (so contributors understand the logic)
- An invitation for contributions — especially from developers who want to improve model accuracy, add GPU support, build a frontend, or optimize for different hardware

-----

## Out of Scope for v1

- GUI or web interface
- Real-time video stream processing
- Cloud processing or server deployment
- Music or audio filtering (that is ELUATE’s job)
- Automatic video downloading
- Support for streaming platforms directly

These are things contributors can build on top of OCCLUDE later.

-----

## Success Criteria for v1

A 10-minute documentary clip containing clearly immodestly dressed people (men shirtless, women with uncovered hair and arms) is processed by OCCLUDE and the output video has those people visibly blurred, audio is intact, and the blur does not flicker significantly between frames.

That is enough to publish and attract contributors.