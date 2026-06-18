"""Shot-cut detection for the offline tracking pass.

SAM2 propagates a mask forward assuming visual continuity. Across a hard cut
that assumption breaks: the pixels under the old mask now belong to a
different scene, so propagation smears one person's silhouette onto whatever
replaced them. So Pass 1 splits the video at shot cuts — a cut ends every
open tracklet and forces a fresh detector seed on the next frame.

The signal is intentionally cheap and model-free: the total-variation
distance between consecutive frames' grayscale histograms. A hard cut
swaps almost all content at once, spiking this metric far above the
within-shot baseline; a histogram (rather than per-pixel diff) is robust to
camera pans and subject motion, which barely move the intensity
distribution. This is a coarse gate, not a learned shot detector:
over-splitting (a false cut) costs only one extra detector seed, which is
cheap, while the tracking pass also re-seeds periodically so a *missed* cut
self-heals within a second.
"""
from __future__ import annotations

import numpy as np

# Total-variation distance between normalized grayscale histograms (0..1)
# above which two consecutive frames are treated as a shot cut. 0.35 fires
# on hard cuts while clearing within-shot motion (pans, walking people),
# which rarely pushes histogram TV past ~0.15 even on busy footage.
DEFAULT_CUT_THRESHOLD = 0.35

# Histogram resolution. 64 bins is enough to separate distinct scenes
# without making the metric jittery to small lighting shifts.
HIST_BINS = 64


def gray_histogram(gray: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """Normalized intensity histogram of a uint8 grayscale frame.

    Returns a `bins`-length float array summing to 1.0 (or all-zeros for an
    empty frame), so the distance below is resolution-independent — the same
    threshold works at 480p or 4K.
    """
    hist, _ = np.histogram(gray, bins=bins, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return hist.astype(np.float32)
    return hist.astype(np.float32) / float(total)


def histogram_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Total-variation distance between two normalized histograms, in 0..1.

    0 means identical distributions, 1 means disjoint. TV is half the L1
    distance; for probability vectors it lands cleanly in [0, 1], which is
    what makes DEFAULT_CUT_THRESHOLD interpretable.
    """
    return float(0.5 * np.abs(h1 - h2).sum())


class ShotSegmenter:
    """Stateful per-frame cut detector for the decode loop.

    Feed it grayscale frames in order; ``push`` returns True on the frame
    that *starts a new shot* (i.e. the cut happened between the previous
    frame and this one). The first frame is always a shot start.
    """

    def __init__(self, threshold: float = DEFAULT_CUT_THRESHOLD, bins: int = HIST_BINS) -> None:
        self.threshold = threshold
        self.bins = bins
        self._prev_hist: np.ndarray | None = None

    def push(self, gray: np.ndarray) -> bool:
        hist = gray_histogram(gray, self.bins)
        if self._prev_hist is None:
            self._prev_hist = hist
            return True
        cut = histogram_distance(self._prev_hist, hist) >= self.threshold
        self._prev_hist = hist
        return cut
