"""Scene cut detection (F-06).

Cuts matter for two reasons: a gain curve must not be smoothed across one, and
the frame-to-frame exposure ratio is meaningless across one (the two frames show
different scenes, so ``median(I_t / I_{t-1})`` measures the edit, not the AE).

The detector compares normalised per-channel histograms of consecutive analysis
frames.  A histogram distance is deliberately insensitive to motion (panning
barely changes the histogram) while a hard cut changes it a lot.
"""

from __future__ import annotations

import numpy as np


def histogram_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """L1/2 distance between two sum-to-one histograms -> [0, 1]."""
    return float(0.5 * np.abs(np.asarray(h1) - np.asarray(h2)).sum())


#: Mean absolute log2 difference between two thumbnails that scores as 1.0.
#: Two unrelated shots typically differ by more than a stop per pixel; two
#: consecutive frames of the same shot, even during a brisk pan, by far less.
SPATIAL_SCALE = 1.0

#: Half-width of the thumbnail shift search, in thumbnail pixels.  At the
#: default 16x16 thumbnail, +/-3 absorbs a pan of up to ~19% of the frame width
#: per frame - far more than any handheld move, but far less than the
#: rearrangement a cut produces.
SPATIAL_SEARCH = 3


def _shifted_absdiff(t1: np.ndarray, t2: np.ndarray, dy: int, dx: int) -> float:
    """Mean |t1 - t2| over the region the two frames still have in common.

    Cropping rather than wrapping matters: a pan genuinely brings new content in
    at one edge, and that content should not be counted as a mismatch.
    """
    h, w = t1.shape[:2]
    y1a, y1b = max(0, dy), h + min(0, dy)
    x1a, x1b = max(0, dx), w + min(0, dx)
    y2a, y2b = max(0, -dy), h + min(0, -dy)
    x2a, x2b = max(0, -dx), w + min(0, -dx)
    a = t1[y1a:y1b, x1a:x1b]
    b = t2[y2a:y2b, x2a:x2b]
    if a.size == 0:
        return float("inf")
    return float(np.mean(np.abs(a - b)))


def spatial_distance(t1: np.ndarray, t2: np.ndarray, search: int = SPATIAL_SEARCH) -> float:
    """Motion-tolerant layout difference between two gain-invariant thumbnails.

    Without the shift search a 12 px/frame pan scored as a cut on 33% of its
    frames; taking the best alignment within a small window removes that while
    leaving a real cut an order of magnitude above the noise floor.
    """
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    best = float("inf")
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            best = min(best, _shifted_absdiff(t1, t2, dy, dx))
    return min(best / SPATIAL_SCALE, 1.0)


def cut_scores(
    histograms: np.ndarray, thumbnails: np.ndarray | None = None
) -> np.ndarray:
    """Per-frame dissimilarity to the previous frame (index 0 is always 0).

    The two descriptors fail in opposite directions - the histogram is blind to
    layout, the thumbnail is sensitive to motion - so the score is the larger of
    the two and a shot boundary only has to trip one of them.
    """
    h = np.asarray(histograms, dtype=np.float64)
    scores = np.zeros(len(h), dtype=np.float64)
    for i in range(1, len(h)):
        score = histogram_distance(h[i], h[i - 1])
        if thumbnails is not None:
            score = max(score, spatial_distance(thumbnails[i], thumbnails[i - 1]))
        scores[i] = score
    return scores


def detect_cuts(scores: np.ndarray, threshold: float, min_gap: int = 2) -> list[int]:
    """Indices where a new shot starts.

    ``min_gap`` suppresses a burst of detections around a dissolve, keeping only
    the strongest frame of the burst.
    """
    scores = np.asarray(scores, dtype=np.float64)
    candidates = [i for i in range(1, len(scores)) if scores[i] >= threshold]
    cuts: list[int] = []
    for idx in candidates:
        if cuts and idx - cuts[-1] < min_gap:
            if scores[idx] > scores[cuts[-1]]:
                cuts[-1] = idx
            continue
        cuts.append(idx)
    return cuts


def times_to_indices(times: list[float], timestamps: np.ndarray) -> list[int]:
    """Map manual ``--cuts`` timestamps to the nearest analysis frame index."""
    ts = np.asarray(timestamps, dtype=np.float64)
    out = []
    for t in times:
        if ts.size == 0:
            continue
        idx = int(np.argmin(np.abs(ts - float(t))))
        if idx > 0:
            out.append(idx)
    return sorted(set(out))
