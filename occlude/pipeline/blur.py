"""Blur-mask preparation and application for OCCLUDE video frames.

This module owns the silhouette-shaped alpha mask, pixelation, Gaussian
blur, and CUDA/CPU dispatch used by the video pipeline.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Default Gaussian kernel (overridable via --blur-strength). 199 on a
# 1280×720 frame is a thoroughly opaque smudge. Was 99 (still
# acceptable for v1) but user feedback wanted a much higher radius.
# Silhouette mask shaping. The blur alpha is built from the SegFormer
# silhouette (any non-background label) so the blur traces the person
# outline instead of a rectangle. The silhouette is dilated outward
# (rounding sharp segmentation corners and adding a generous buffer
# zone) then Gaussian-feathered for a wide soft falloff into the
# background — avoids the "paper-cutout" look a tight mask produces.
# Sizes are a fraction of the crop's short side, with a floor in raw
# pixels so even small/distant subjects keep a visible buffer.
SILHOUETTE_DILATE_FRAC = 0.05    # ~10 px on a 200 px crop, ~50 px on a 1000 px crop
SILHOUETTE_DILATE_MIN_PX = 13    # floor — ensures ≥~13 px buffer past the silhouette
SILHOUETTE_FEATHER_FRAC = 0.15   # wide feather kernel for a soft, gradual falloff
SILHOUETTE_FEATHER_MIN_PX = 25

# Pixelation block size. Before the Gaussian pass we downsample the
# region to ~PIXELATE_BLOCKS_ACROSS blocks across the short side and
# nearest-upsample back, then Gaussian-blur the result. Pixelation
# destroys local positional detail (button lines, cleavage edges,
# facial features) that a pure Gaussian preserves as low-frequency
# luminance gradients — even at kernel 199 a Gaussian leaves a
# recognizable silhouette of contrast within the blur. The follow-up
# Gaussian softens the block edges so the output reads as a soft
# blur, not visible mosaic. Lower value = bigger blocks = more
# obscured.
PIXELATE_BLOCKS_ACROSS = 18
PIXELATE_MIN_BLOCK_PX = 8

def _blur_region_bounds(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int, int, int, int, int] | None:
    """Return (px1, py1, px2, py2, ox, oy, bh, bw) for the padded work region.

    Returns None if the bbox is degenerate (zero or negative area after clamping).
    Used by both prepare_blur_mask and apply_blur_mask so their region math stays
    identical — divergence here would produce mask/region shape mismatches.
    """
    h, w = frame_shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    short = min(bh, bw)
    pad_frac = SILHOUETTE_DILATE_FRAC + SILHOUETTE_FEATHER_FRAC
    pad_min = SILHOUETTE_DILATE_MIN_PX + SILHOUETTE_FEATHER_MIN_PX
    pad = max(pad_min, int(short * pad_frac))
    px1 = max(0, x1 - pad)
    py1 = max(0, y1 - pad)
    px2 = min(w, x2 + pad)
    py2 = min(h, y2 + pad)
    ox, oy = x1 - px1, y1 - py1
    return px1, py1, px2, py2, ox, oy, bh, bw


def prepare_blur_mask(
    bbox: tuple[int, int, int, int],
    seg_mask: np.ndarray | "torch.Tensor" | None,
    frame_shape: tuple[int, int],
) -> np.ndarray | "torch.Tensor":
    """Compute the float32 alpha mask (dilation + feather) for *bbox*.

    Returns an array of shape (rh, rw) — the padded work-region dimensions
    derived from *bbox* and *frame_shape*. Values are in [0, 1]. Returns a
    zero-area array if the bbox is degenerate.

    When CUDA is available the mask is built and returned as a
    ``torch.Tensor`` on the blur device, so the downstream
    :func:`apply_blur_mask` torch path doesn't have to bounce a numpy
    array through ``torch.from_numpy().to(device)`` per frame. On Mac /
    CPU the function returns a ``np.ndarray`` exactly as before — every
    existing test pins this path.

    The mask is valid only for the *frame_shape* it was prepared against;
    OCCLUDE processes one fixed-resolution video at a time so this is safe.
    """
    if _BLUR_DEVICE is not None:
        return _prepare_blur_mask_torch(bbox, seg_mask, frame_shape, _BLUR_DEVICE)
    if isinstance(seg_mask, torch.Tensor):
        seg_mask = seg_mask.detach().cpu().numpy()
    bounds = _blur_region_bounds(bbox, frame_shape)
    if bounds is None:
        return np.zeros((0, 0), dtype=np.float32)
    px1, py1, px2, py2, ox, oy, bh, bw = bounds
    rh, rw = py2 - py1, px2 - px1
    short = min(bh, bw)

    if seg_mask is not None and seg_mask.size > 0 and seg_mask.max() > 0:
        sil = (seg_mask > 0).astype(np.uint8)
        if sil.shape != (bh, bw):
            sil = cv2.resize(sil, (bw, bh), interpolation=cv2.INTER_NEAREST)
        silhouette = np.zeros((rh, rw), dtype=np.uint8)
        silhouette[oy:oy + bh, ox:ox + bw] = sil

        d = max(SILHOUETTE_DILATE_MIN_PX, int(short * SILHOUETTE_DILATE_FRAC))
        if d % 2 == 0:
            d += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        silhouette = cv2.dilate(silhouette, kernel)

        mask = silhouette.astype(np.float32)
        f = max(SILHOUETTE_FEATHER_MIN_PX, int(short * SILHOUETTE_FEATHER_FRAC))
        if f % 2 == 0:
            f += 1
        mask = cv2.GaussianBlur(mask, (f, f), 0)
    else:
        # Fall back to a feathered rectangle when no silhouette is available
        # or when the segmentation mask is entirely background.
        mask = np.zeros((rh, rw), dtype=np.float32)
        mask[oy:oy + bh, ox:ox + bw] = 1.0
        f = max(3, min(bh, bw) // 8)
        if f % 2 == 0:
            f += 1
        mask = cv2.GaussianBlur(mask, (f, f), 0)

    return mask


# The pixelate + 199-kernel Gaussian + composite runs on EVERY output
# frame — it is the only
# remaining per-frame compute that isn't already on the accelerator. On
# CUDA the OpenCV/CPU path is replaced with a torch implementation that
# does the resize, separable Gaussian, and composite on the GPU. Mac/CPU
# (no CUDA) keep the OpenCV path, which is also the reference the unit
# tests pin. Detected once at import.
_BLUR_DEVICE: torch.device | None = (
    torch.device("cuda") if torch.cuda.is_available() else None
)
# Cache of 1-D Gaussian kernels keyed by (k, device, dtype). The kernel
# is rebuilt only when the blur strength changes (it never does within a
# run), so this is effectively a one-time cost.
_GAUSS_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _cv2_gaussian_kernel1d(k: int, device: torch.device) -> torch.Tensor:
    """1-D kernel matching cv2.GaussianBlur(..., sigmaX=0).

    OpenCV derives sigma from the kernel size when sigma is 0:
    ``sigma = 0.3*((k-1)*0.5 - 1) + 0.8``. Replicating that keeps the
    GPU blur visually identical to the OpenCV reference.
    """
    key = (k, device.type)
    cached = _GAUSS_CACHE.get(key)
    if cached is not None:
        return cached
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    xs = torch.arange(k, dtype=torch.float32, device=device) - (k - 1) / 2.0
    g = torch.exp(-(xs**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    _GAUSS_CACHE[key] = g
    return g


# Cache of (k, device) → elliptical {0,1} kernel matching
# cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)). Used to dilate
# the silhouette on-device without the cv2.dilate CPU round-trip.
_ELLIPSE_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _ellipse_kernel(k: int, device: torch.device) -> torch.Tensor:
    """Elliptical {0,1} kernel of shape (1, 1, k, k) matching cv2's
    ``getStructuringElement(MORPH_ELLIPSE, (k, k))``.

    Used as the conv kernel inside the dilation step: we conv the
    binary silhouette with this stamp and threshold ``> 0`` to get the
    same "any pixel within an ellipse-radius distance is foreground"
    result that cv2.dilate produces. The math is the per-pixel "is any
    point in the kernel footprint inside the binary mask" check —
    convolution turns it into a single op the GPU can fuse.
    """
    key = (k, device.type)
    cached = _ELLIPSE_CACHE.get(key)
    if cached is not None:
        return cached
    r = (k - 1) / 2.0
    coord = torch.arange(k, dtype=torch.float32, device=device) - r
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    # r==0 only happens for k==1, where the ellipse is just the center pixel.
    if r <= 0:
        kern = torch.ones_like(yy)
    else:
        kern = ((yy * yy + xx * xx) <= (r * r)).to(torch.float32)
    kern = kern.view(1, 1, k, k)
    _ELLIPSE_CACHE[key] = kern
    return kern


def _prepare_blur_mask_torch(
    bbox: tuple[int, int, int, int],
    seg_mask: np.ndarray | "torch.Tensor" | None,
    frame_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Torch/GPU equivalent of :func:`prepare_blur_mask`.

    Returns a float32 alpha tensor of shape ``(rh, rw)`` on ``device``,
    matching the numpy version's shape and value range. The dilation
    uses a conv with the cv2 elliptical kernel + threshold (visually
    equivalent to cv2.dilate); the feather uses the same 1-D Gaussian
    kernel as the existing torch blur path (bit-equivalent sigma).
    """
    bounds = _blur_region_bounds(bbox, frame_shape)
    if bounds is None:
        return torch.zeros((0, 0), dtype=torch.float32, device=device)
    px1, py1, px2, py2, ox, oy, bh, bw = bounds
    rh, rw = py2 - py1, px2 - px1
    short = min(bh, bw)

    # Resolve "is there a usable silhouette?" without forcing a host
    # sync — if the caller passed a CUDA tensor we'd rather not pull
    # .max() back to the CPU when a cheap shape check is enough.
    have_sil: bool
    if seg_mask is None:
        have_sil = False
    elif isinstance(seg_mask, np.ndarray):
        have_sil = seg_mask.size > 0 and bool(seg_mask.max() > 0)
    else:
        have_sil = seg_mask.numel() > 0 and bool((seg_mask > 0).any().item())

    if have_sil:
        if isinstance(seg_mask, np.ndarray):
            sil = torch.from_numpy((seg_mask > 0).astype(np.uint8)).to(
                device=device, dtype=torch.float32
            )
        else:
            sil = (seg_mask > 0).to(device=device, dtype=torch.float32)
        sil = sil.view(1, 1, *sil.shape[-2:])
        if sil.shape[-2:] != (bh, bw):
            sil = F.interpolate(sil, size=(bh, bw), mode="nearest")

        canvas = torch.zeros((1, 1, rh, rw), dtype=torch.float32, device=device)
        canvas[:, :, oy:oy + bh, ox:ox + bw] = sil

        d = max(SILHOUETTE_DILATE_MIN_PX, int(short * SILHOUETTE_DILATE_FRAC))
        if d % 2 == 0:
            d += 1
        ellipse = _ellipse_kernel(d, device)
        pad_d = d // 2
        # Constant (zero) padding matches cv2.dilate's default BORDER_CONSTANT
        # with borderValue=0 — the silhouette doesn't bleed in from outside.
        dil = F.conv2d(
            F.pad(canvas, (pad_d, pad_d, pad_d, pad_d), mode="constant", value=0.0),
            ellipse,
        )
        canvas = (dil > 0).to(torch.float32)

        f = max(SILHOUETTE_FEATHER_MIN_PX, int(short * SILHOUETTE_FEATHER_FRAC))
        if f % 2 == 0:
            f += 1
    else:
        canvas = torch.zeros((1, 1, rh, rw), dtype=torch.float32, device=device)
        canvas[:, :, oy:oy + bh, ox:ox + bw] = 1.0
        f = max(3, min(bh, bw) // 8)
        if f % 2 == 0:
            f += 1

    # Separable Gaussian feather with reflect padding (cv2.GaussianBlur
    # default is BORDER_REFLECT_101 == torch 'reflect').
    g = _cv2_gaussian_kernel1d(f, device)
    pad_f = f // 2
    kh = g.view(1, 1, 1, f)
    kv = g.view(1, 1, f, 1)
    mask = F.pad(canvas, (pad_f, pad_f, 0, 0), mode="reflect")
    mask = F.conv2d(mask, kh)
    mask = F.pad(mask, (0, 0, pad_f, pad_f), mode="reflect")
    mask = F.conv2d(mask, kv)
    return mask.view(rh, rw)


def _apply_blur_mask_torch(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray | "torch.Tensor",
    k: int,
    px1: int,
    py1: int,
    px2: int,
    py2: int,
    device: torch.device,
) -> None:
    region = frame_bgr[py1:py2, px1:px2]
    rh, rw = region.shape[:2]

    # reflect padding requires pad < dim; pad = k//2, so k < 2*min(rh,rw).
    k = min(k, 2 * min(rh, rw) - 1)
    if k % 2 == 0:
        k -= 1
    k = max(1, k)

    block = max(PIXELATE_MIN_BLOCK_PX, min(rh, rw) // PIXELATE_BLOCKS_ACROSS)
    small_w = max(2, rw // block)
    small_h = max(2, rh // block)

    t = (
        torch.from_numpy(np.ascontiguousarray(region))
        .to(device)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )  # (1, 3, rh, rw) BGR

    # Pixelate: area-downsample then nearest-upsample (cv2 INTER_AREA /
    # INTER_NEAREST equivalents).
    small = F.interpolate(t, size=(small_h, small_w), mode="area")
    pix = F.interpolate(small, size=(rh, rw), mode="nearest")

    # Separable Gaussian, depthwise, with reflect padding (torch
    # 'reflect' == cv2 BORDER_REFLECT_101, OpenCV's GaussianBlur default).
    g = _cv2_gaussian_kernel1d(k, device)
    pad = k // 2
    kh = g.view(1, 1, 1, k).expand(3, 1, 1, k)
    kv = g.view(1, 1, k, 1).expand(3, 1, k, 1)
    pix = F.pad(pix, (pad, pad, 0, 0), mode="reflect")
    pix = F.conv2d(pix, kh, groups=3)
    pix = F.pad(pix, (0, 0, pad, pad), mode="reflect")
    blurred = F.conv2d(pix, kv, groups=3)

    # Mask may already be a torch tensor on device (the CUDA path of
    # prepare_blur_mask produces one). Skip the numpy round-trip in
    # that case — it's the per-frame copy this overhaul is targeting.
    if isinstance(mask, torch.Tensor):
        m = mask.to(device=device, dtype=torch.float32).view(1, 1, rh, rw)
    else:
        m = (
            torch.from_numpy(np.ascontiguousarray(mask))
            .to(device)
            .view(1, 1, rh, rw)
        )
    reg_f = t
    composite = blurred * m + reg_f * (1.0 - m)
    out = (
        composite.clamp_(0, 255)
        .squeeze(0)
        .permute(1, 2, 0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    frame_bgr[py1:py2, px1:px2] = out


def _apply_blur_mask_cv2(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    k: int,
    px1: int,
    py1: int,
    px2: int,
    py2: int,
) -> None:
    region = frame_bgr[py1:py2, px1:px2]
    rh, rw = region.shape[:2]

    # cv2.GaussianBlur also fails when k >= region dimension; clamp to be safe.
    k = min(k, 2 * min(rh, rw) - 1)
    if k % 2 == 0:
        k -= 1
    k = max(1, k)

    block = max(PIXELATE_MIN_BLOCK_PX, min(rh, rw) // PIXELATE_BLOCKS_ACROSS)
    small_w = max(2, rw // block)
    small_h = max(2, rh // block)
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    blurred = cv2.GaussianBlur(pixelated, (k, k), 0)

    mask3 = mask[:, :, None]
    composite = blurred.astype(np.float32) * mask3 + region.astype(np.float32) * (1.0 - mask3)
    frame_bgr[py1:py2, px1:px2] = np.clip(composite, 0, 255).astype(np.uint8)


def apply_blur_mask(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray | "torch.Tensor",
    blur_kernel: int,
) -> None:
    """Apply pixelate+Gaussian blur composite to *frame_bgr* in place using a precomputed *mask*.

    *mask* is the float32 alpha array returned by
    :func:`prepare_blur_mask` for the same *bbox* and frame shape. May
    be either ``np.ndarray`` (CPU path) or ``torch.Tensor`` (CUDA path)
    — the CUDA composite accepts the tensor directly so no per-frame
    numpy round-trip is required. Silently skips degenerate inputs.
    Runs on CUDA via torch when available, otherwise OpenCV on the CPU.
    """
    mask_numel = mask.numel() if isinstance(mask, torch.Tensor) else mask.size
    if mask_numel == 0:
        return
    bounds = _blur_region_bounds(bbox, frame_bgr.shape[:2])
    if bounds is None:
        return
    px1, py1, px2, py2, _ox, _oy, _bh, _bw = bounds

    k = blur_kernel
    if k % 2 == 0:
        k -= 1
    if k < 3:
        return

    if _BLUR_DEVICE is not None:
        _apply_blur_mask_torch(
            frame_bgr, bbox, mask, k, px1, py1, px2, py2, _BLUR_DEVICE
        )
    else:
        # The cv2 fallback path takes numpy only. The dispatch in
        # prepare_blur_mask guarantees a numpy mask here, but accept a
        # tensor too in case a caller built one manually for tests.
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        _apply_blur_mask_cv2(frame_bgr, mask, k, px1, py1, px2, py2)


def _scale_blur_mask(
    mask: np.ndarray | "torch.Tensor",
    alpha: float,
) -> np.ndarray | "torch.Tensor":
    """Scale a NumPy or torch blur mask without forcing CUDA tensors to host."""
    if isinstance(mask, torch.Tensor):
        return mask * float(alpha)
    return mask * np.float32(alpha)


def blur_region(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    seg_mask: np.ndarray | None,
    blur_kernel: int,
) -> None:
    """Apply pixelate+Gaussian blur with a silhouette-shaped alpha to *frame_bgr* in place."""
    mask = prepare_blur_mask(bbox, seg_mask, frame_bgr.shape[:2])
    apply_blur_mask(frame_bgr, bbox, mask, blur_kernel)


