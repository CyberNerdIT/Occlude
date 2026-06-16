"""Stage 4 — Perception.

Wraps the YOLO person detector, the SegFormer body-part segmenter, and
the InsightFace gender classifier as a single callable that maps an
image to a list of per-person observations. This is the data structure
the Stage 5 rule layer consumes.

The class loads all three models once on construction; calling the
instance on an image runs detection, then segments + classifies each
person crop.
"""
import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Protocol, runtime_checkable

import cv2
import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo as _imz
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO


class _NoArenaPickableSession(_imz.PickableInferenceSession):
    """InsightFace `PickableInferenceSession` with the ONNX Runtime CPU
    memory arena disabled.

    By default ORT keeps a per-session arena that retains freed buffers
    for reuse. Empirically, on multi-person video, each
    `face_app.get(bgr)` call leaks ~45 MB/person into compressed-VM
    pages that macOS's Activity Monitor tallies into the Jetsam
    footprint metric (vmmap MALLOC_LARGE virtual = 4.6 GB at frame 25
    of laughing_people.mp4 with arena on, ~250 MB total with arena
    off). Disabling the arena trades a small inference slowdown for
    bounded steady-state memory.
    """

    def __init__(self, model_path, **kwargs):
        if "sess_options" not in kwargs:
            opts = onnxruntime.SessionOptions()
            opts.enable_cpu_mem_arena = False
            # mem_pattern caches per-input-shape allocation plans.
            # InsightFace receives variable-sized BGR crops per
            # detection call, so each unique shape grew the cache.
            # Disabling forces fresh allocations and bounds the heap.
            opts.enable_mem_pattern = False
            kwargs["sess_options"] = opts
        super().__init__(model_path, **kwargs)

YOLO_MODEL_ID = "yolov8n.pt"
SEG_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
INSIGHT_MODEL_ID = "buffalo_l"
PERSON_CLASS_ID = 0
DEFAULT_PERSON_CONF = 0.40
INSIGHT_DET_SIZE = (640, 640)

# Cadence for the full gc.collect() in _cleanup_device_memory. A full
# collect with torch+transformers resident is ~0.2 s; running it every
# perception frame profiled as ~36% of total runtime. It only exists to
# bound heap growth under macOS memory pressure, and video.py already
# does its own periodic cleanup at the loop level, so collecting every
# Nth call bounds memory without the per-frame tax.
GC_EVERY = 50
CUDA_EMPTY_CACHE_EVERY = int(os.environ.get("OCCLUDE_CUDA_EMPTY_CACHE_EVERY", "0"))

# 18 SegFormer classes from `mattmdjaga/segformer_b2_clothes`. Index
# into this list maps to the integer label in the predicted mask.
SEG_LABELS = [
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes",
    "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe",
    "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm",
    "Bag", "Scarf",
]

# Subset the modesty rule layer cares about. Surfaced here (not buried
# in test scripts) so Stage 5 can import the same set.
TARGET_LABELS = frozenset({
    "Hair", "Hat", "Scarf", "Upper-clothes", "Pants", "Skirt", "Dress",
    "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm",
})
LABEL_TO_ID = {name: i for i, name in enumerate(SEG_LABELS)}


@dataclass
class Person:
    """Everything the rule layer needs to know about one detected person."""

    # YOLO bbox in source-image pixel coordinates.
    bbox: tuple[int, int, int, int]
    # YOLO person-class confidence.
    det_conf: float
    # Cropped RGB image at the bbox.
    crop: Image.Image
    # SegFormer pixel-wise label map, same H×W as `crop`. uint8 values
    # index into SEG_LABELS.
    seg_mask: np.ndarray
    # InsightFace gender: 'M', 'F', or None when no face was found.
    gender: str | None
    # InsightFace *face detection* confidence — answers "is there a
    # face here?", not "is the gender prediction reliable." Stage 3
    # (`docs/04-gender-classifier.md`, Finding 2) is explicit about
    # this distinction. 0.0 when no face was detected.
    face_det_score: float
    # Boolean mask per rule-relevant label, same H×W as seg_mask.
    # Populated by Perception.detect_and_segment so the rule layer
    # never needs to know integer label indices — only string names.
    label_masks: dict[str, np.ndarray]
    # InsightFace estimated age, or None when no face was found. The
    # buffalo_l genderage head already computes this (docs/04 table) —
    # we previously discarded it. Used only by the rule layer's child
    # exemption. Trailing default so existing constructors/tests that
    # don't pass an age keep working unchanged.
    age: float | None = None


@runtime_checkable
class Perceiver(Protocol):
    """Interface the video pipeline depends on.

    One production adapter exists: :class:`Perception`.  Fakes that
    satisfy this protocol can be injected into :class:`VideoProcessor`
    for testing the tracking / temporal-smoothing logic without loading
    any model weights.
    """

    def detect_and_segment(self, image: Image.Image) -> list[Person]: ...
    def classify(
        self, crop: Image.Image
    ) -> tuple[str | None, float, float | None]: ...


def _pick_device(requested: str | None = None) -> torch.device:
    choice = (requested or os.environ.get("OCCLUDE_DEVICE") or "auto").lower()
    if choice == "auto":
        # Colab/A100 is the primary long-run target. Prefer CUDA whenever
        # it exists; MPS remains a local-development fallback only.
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device '{requested}'; expected auto, cuda, mps, or cpu")


class Perception:
    def __init__(
        self,
        person_conf: float = DEFAULT_PERSON_CONF,
        *,
        device: str | None = None,
        detector_model: str | None = None,
    ) -> None:
        self.person_conf = person_conf
        self.device = _pick_device(device)
        self.detector_model = detector_model or YOLO_MODEL_ID

        # Ultralytics device selector for predict(). MPS was previously
        # pinned to CPU here, leaving the detector as the only CPU-bound
        # model on Apple Silicon while SegFormer ran on MPS. 8.4.46
        # accepts "mps" directly.
        self._yolo_device = (
            0 if self.device.type == "cuda"
            else "mps" if self.device.type == "mps"
            else "cpu"
        )
        # fp16 detection on CUDA: ~1.5-2× the throughput, and YOLO's
        # output is thresholded + argmax'd (same benign-precision-loss
        # argument as the SegFormer .half() below). Ultralytics only
        # supports half on CUDA; MPS/CPU keep fp32.
        self._yolo_half = self.device.type == "cuda"

        self.detector = YOLO(self.detector_model)

        self.seg_processor = SegformerImageProcessor.from_pretrained(SEG_MODEL_ID)

        # GPU-side image preprocessing. The HF SegformerImageProcessor
        # resizes + rescales + normalizes every person crop on the CPU
        # (PIL/NumPy) before the tensor ever reaches the GPU — pure CPU
        # work feeding the pipeline's dominant model. Replicate it with
        # torch ops on-device. Only enabled when every transform we
        # replicate is actually on (do_resize/rescale/normalize); any
        # non-standard processor config falls back to the HF path so we
        # never silently mis-preprocess. resample==2 is PIL BILINEAR;
        # F.interpolate(bilinear) is the on-device equivalent (argmax +
        # the dilated/feathered silhouette downstream absorb the
        # sub-pixel resampling differences — verified by mask-agreement
        # check, see docs/08).
        _sp = self.seg_processor
        _size = _sp.size
        _sh = (
            (int(_size.height), int(_size.width))
            if hasattr(_size, "height") and _size.height is not None
            else (int(_size["height"]), int(_size["width"]))
        )
        self._gpu_prep = (
            self.device.type in ("cuda", "mps")
            and bool(getattr(_sp, "do_resize", False))
            and bool(getattr(_sp, "do_rescale", False))
            and bool(getattr(_sp, "do_normalize", False))
        )
        self._seg_hw = _sh
        self._seg_rescale = float(getattr(_sp, "rescale_factor", 1 / 255))
        self._seg_mean = torch.tensor(
            _sp.image_mean, dtype=torch.float32
        ).view(1, 3, 1, 1).to(self.device)
        self._seg_std = torch.tensor(
            _sp.image_std, dtype=torch.float32
        ).view(1, 3, 1, 1).to(self.device)

        self.seg_model = (
            AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL_ID)
            .to(self.device)
            .eval()
        )
        # fp16: logits are immediately argmax'd so precision loss is benign.
        # CPU fp16 support in PyTorch is incomplete; restrict to MPS/CUDA.
        if self.device.type in ("mps", "cuda"):
            self.seg_model = self.seg_model.half()

        # torch.compile fuses transformer attention kernels on top of fp16.
        # dynamic=True treats the batch dim as symbolic so frames with
        # different person counts (1 person, then 3, then 2 …) don't each
        # trigger a recompile. First batch incurs one-time JIT compilation
        # (~10-30 frames of wall-time warm-up at 1 fps baseline); all
        # subsequent batches run fused. On MPS, unsupported ops fall back to
        # eager silently — no correctness risk. Silently skip if the backend
        # raises (e.g. an inductor limitation on this torch build).
        if self.device.type in ("mps", "cuda"):
            try:
                self.seg_model = torch.compile(self.seg_model, dynamic=True)
            except Exception:
                pass

        # Two scoped tweaks for memory bounding:
        #
        # 1. `allowed_modules` filters the buffalo_l bundle from 5
        #    sub-models down to the 2 we actually consume. We only
        #    read `face.gender` and `face.det_score`, so landmarks
        #    (2d_106, 3d_68) and recognition embeddings are pure
        #    overhead — fewer models per call = less per-call CPU
        #    allocation.
        # 2. Swap insightface's PickableInferenceSession for our
        #    arena/mem_pattern-disabled subclass during construction,
        #    then restore — keeps the patch scoped to this instance.
        # InsightFace runs under ONNX Runtime, not torch. We want its
        # detection + genderage sessions on the accelerator too, else this
        # CPU-pinned model dominates the per-track cost. CUDA needs the
        # onnxruntime-gpu wheel; Apple Silicon uses the CoreML EP that
        # ships in stock onnxruntime. Both keep CPUExecutionProvider last
        # so ORT falls back per-op if the accelerator can't run a node.
        _use_cuda = self.device.type == "cuda"
        _use_coreml = self.device.type == "mps"
        if _use_cuda:
            _providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif _use_coreml:
            _providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            _providers = ["CPUExecutionProvider"]
        _orig = _imz.PickableInferenceSession
        _imz.PickableInferenceSession = _NoArenaPickableSession
        try:
            self.face_app = FaceAnalysis(
                name=INSIGHT_MODEL_ID,
                providers=_providers,
                allowed_modules=["detection", "genderage"],
            )
            self.face_app.prepare(
                ctx_id=0 if _use_cuda else -1, det_size=INSIGHT_DET_SIZE
            )
        finally:
            _imz.PickableInferenceSession = _orig

        # If an accelerator was requested but ORT couldn't load its EP
        # (onnxruntime-gpu missing / CUDA-cuDNN mismatch, or CoreML
        # failing to compile the buffalo_l graph), it silently uses the
        # CPU EP that's also in the providers list. On CUDA that turns
        # a Colab run into a crawl, so fail hard. MPS/CoreML remains a
        # local-development fallback and only warns.
        _accel_ep = (
            "CUDAExecutionProvider" if _use_cuda
            else "CoreMLExecutionProvider" if _use_coreml
            else None
        )
        if _accel_ep is not None:
            _active: set[str] = set()
            for _m in self.face_app.models.values():
                _sess = getattr(_m, "session", None)
                if _sess is not None:
                    _active.update(_sess.get_providers())
            if _accel_ep not in _active:
                import sys
                _hint = (
                    "Plain `onnxruntime` (a core dep) shadows "
                    "`onnxruntime-gpu` even with the [gpu] extra — "
                    "installing the extra alone is NOT enough. Fix: "
                    "pip uninstall -y onnxruntime onnxruntime-gpu && "
                    "pip install --force-reinstall --no-deps "
                    "onnxruntime-gpu  (must match this CUDA/cuDNN)."
                    if _use_cuda else
                    "The CoreML EP couldn't bind buffalo_l on this "
                    "onnxruntime build."
                )
                msg = (
                    "InsightFace is running on CPU "
                    f"despite a {self.device.type.upper()} device; "
                    f"active EPs: {sorted(_active)}. {_hint}"
                )
                if _use_cuda:
                    raise RuntimeError(msg)
                print(f"\n[occlude] WARNING: {msg} Continuing on CPU.\n",
                      file=sys.stderr, flush=True)

    @staticmethod
    def make_label_masks(seg_mask: np.ndarray) -> dict[str, np.ndarray]:
        """Convert an integer seg_mask to named boolean arrays.

        Isolates the model's integer label convention here so the rule
        layer only deals with string keys and never imports SEG_LABELS.
        """
        return {
            name: (seg_mask == LABEL_TO_ID[name])
            for name in TARGET_LABELS
        }

    def __call__(self, image: Image.Image) -> list[Person]:
        """Single-image entry point: detect, segment, classify all in one.

        Used by test scripts. The video pipeline uses
        :meth:`detect_and_segment` + :meth:`classify` separately so it
        can cache gender per IoU-tracked person and avoid the
        ~200 MB/frame heap growth that running InsightFace ONNX every
        frame produces (see docs/07-video-pipeline.md Finding 6).
        """
        people = self.detect_and_segment(image)
        for person in people:
            person.gender, person.face_det_score, person.age = self.classify(
                person.crop
            )
        return people

    def detect_and_segment(self, image: Image.Image) -> list[Person]:
        """Detect persons, segment each, return Person list with
        ``gender=None`` and ``face_det_score=0.0``.

        Single-image entry point — kept stable for the Perceiver
        protocol and for tests. Delegates to
        :meth:`detect_and_segment_batch` so there's one implementation
        path.
        """
        return self.detect_and_segment_batch([image])[0]

    def detect_and_segment_batch(
        self, images: list[Image.Image]
    ) -> list[list[Person]]:
        """Detect + segment across ``len(images)`` frames in two batched
        forward passes (one YOLO predict, one SegFormer forward).

        Returns ``[[Person, ...] per image, ...]``. The flatten/scatter
        is bookkeeping: YOLO is called once with a list, all crops
        across the batch are concatenated for a single SegFormer call,
        then results are scattered back per source image. This collapses
        B frames' worth of GPU pipeline overhead and Python/torch
        launch latency into one call apiece.
        """
        if not images:
            return []

        dets = self.detector.predict(
            source=images,
            classes=[PERSON_CLASS_ID],
            conf=self.person_conf,
            verbose=False,
            device=self._yolo_device,
            half=self._yolo_half,
        )

        # Flatten all crops across the batch; remember which frame each
        # came from so we can scatter SegFormer outputs back.
        all_crops: list[Image.Image] = []
        meta_per_frame: list[list[tuple[int, int, int, int, float]]] = []
        crop_slice_per_frame: list[tuple[int, int]] = []
        for image, det in zip(images, dets):
            start = len(all_crops)
            frame_meta: list[tuple[int, int, int, int, float]] = []
            boxes = det.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for (x1f, y1f, x2f, y2f), conf in zip(xyxy, confs):
                    x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
                    # Sub-pixel-thin detections floor to a zero-area box;
                    # the resulting empty crop crashes the segmenter
                    # (F.interpolate / cv2.resize reject a zero spatial
                    # dim). Drop them.
                    if x2 <= x1 or y2 <= y1:
                        continue
                    all_crops.append(image.crop((x1, y1, x2, y2)))
                    frame_meta.append((x1, y1, x2, y2, float(conf)))
            meta_per_frame.append(frame_meta)
            crop_slice_per_frame.append((start, len(all_crops)))

        # One SegFormer forward over the union of crops from all frames.
        # The processor resizes every crop to 512×512 so the batch shape
        # is (sum_crops, 3, 512, 512) regardless of frame count. On
        # CUDA + torch.compile(dynamic=True) this amortizes JIT'd kernel
        # launch overhead across the full batch.
        seg_masks = self._segment_batch(all_crops) if all_crops else []

        results: list[list[Person]] = []
        for frame_meta, (lo, hi) in zip(meta_per_frame, crop_slice_per_frame):
            frame_crops = all_crops[lo:hi]
            frame_seg = seg_masks[lo:hi]
            frame_people: list[Person] = []
            for (x1, y1, x2, y2, conf), crop, seg in zip(
                frame_meta, frame_crops, frame_seg
            ):
                frame_people.append(Person(
                    bbox=(x1, y1, x2, y2),
                    det_conf=conf,
                    crop=crop,
                    seg_mask=seg,
                    gender=None,
                    face_det_score=0.0,
                    label_masks=Perception.make_label_masks(seg),
                ))
            results.append(frame_people)
        return results

    def classify(
        self, crop: Image.Image
    ) -> tuple[str | None, float, float | None]:
        """Run face detection + gender + age classification on a person crop."""
        return self._classify(crop)

    @contextmanager
    def _cleanup_device_memory(self) -> Generator[None, None, None]:
        try:
            yield
        finally:
            # Periodic, not per-call: a full gc.collect() here was the
            # single largest runtime cost (profiled ~36%). MPS cache
            # cleanup remains aggressive for the local Mac path; CUDA
            # cache eviction is disabled by default because it can force
            # synchronization and allocator churn on Colab.
            self._cleanup_calls = getattr(self, "_cleanup_calls", 0) + 1
            if self._cleanup_calls % GC_EVERY == 0:
                gc.collect()
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif (
                self.device.type == "cuda"
                and CUDA_EMPTY_CACHE_EVERY > 0
                and self._cleanup_calls % CUDA_EMPTY_CACHE_EVERY == 0
            ):
                torch.cuda.empty_cache()

    def _preprocess_crops_gpu(self, crops: list[Image.Image]) -> torch.Tensor:
        """SegFormer preprocessing (resize → rescale → normalize) done
        with torch ops on-device instead of the CPU HF processor.

        Returns a ``(N, 3, H, W)`` tensor in the model's dtype. The
        resize runs on 0-255 floats then rescale/normalize follow, which
        is linear-equivalent to the processor's resize-then-rescale order
        (only normalize must come last).
        """
        H, W = self._seg_hw
        dtype = next(self.seg_model.parameters()).dtype
        tensors: list[torch.Tensor] = []
        for crop in crops:
            arr = np.array(crop.convert("RGB"))  # HWC uint8 RGB, writable copy
            t = torch.from_numpy(arr).to(self.device)
            t = t.permute(2, 0, 1).unsqueeze(0).float()  # (1,3,h,w) 0-255
            t = F.interpolate(
                t, size=(H, W), mode="bilinear", align_corners=False
            )
            tensors.append(t)
        batch = torch.cat(tensors, dim=0)
        batch = batch * self._seg_rescale
        batch = (batch - self._seg_mean) / self._seg_std
        return batch.to(dtype=dtype)

    def _segment_batch(self, crops: list[Image.Image]) -> list[np.ndarray]:
        """One forward pass for all person crops in a frame. Argmax on
        device, INTER_NEAREST upsample on CPU, per-batch empty_cache.
        """
        if not crops:
            return []
        with self._cleanup_device_memory():
            if self._gpu_prep:
                pixel_values = self._preprocess_crops_gpu(crops)
                with torch.no_grad():
                    outputs = self.seg_model(pixel_values=pixel_values)
                del pixel_values
            else:
                inputs = self.seg_processor(
                    images=list(crops), return_tensors="pt"
                ).to(self.device)
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    dtype=next(self.seg_model.parameters()).dtype
                )
                with torch.no_grad():
                    outputs = self.seg_model(**inputs)
                del inputs
            # (B, 128, 128) int64 on device — argmax before CPU move keeps
            # the 18-channel logits tensor off CPU heap (Finding 7).
            pred_small = outputs.logits.argmax(dim=1)
            del outputs
            pred_small_cpu = (
                pred_small.detach().to("cpu").numpy().astype(np.uint8, copy=False)
            )
            del pred_small
            results: list[np.ndarray] = []
            for i, crop in enumerate(crops):
                seg = cv2.resize(
                    pred_small_cpu[i], crop.size, interpolation=cv2.INTER_NEAREST
                )
                results.append(seg.astype(np.uint8, copy=False))
        return results

    def _classify(
        self, crop: Image.Image
    ) -> tuple[str | None, float, float | None]:
        bgr = np.array(crop)[:, :, ::-1]
        faces = self.face_app.get(bgr)
        if not faces:
            return None, 0.0, None
        face = max(faces, key=lambda f: f.det_score)
        gender = "M" if int(face.gender) == 1 else "F"
        age = getattr(face, "age", None)
        return gender, float(face.det_score), (None if age is None else float(age))
