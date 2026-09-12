"""Region of interest and exclusion masks (F-07).

The statistics that drive the whole correction are only as good as the pixels
they are computed over.  Two knobs are exposed:

* ``--roi``            - restrict measurement to part of the frame
                         (``center:0.6`` or ``x,y,w,h`` in normalised units).
* ``--exclude-mask``   - drop dome-port reflections / dive-light hotspots,
                         either as a rectangle or a PNG (white = exclude).

Both are resolved once, at analysis resolution, into a single boolean array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _parse_rect(spec: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"rectangle must be 'x,y,w,h' (got {spec!r})")
    x, y, w, h = (float(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError(f"rectangle must have positive size (got {spec!r})")
    return x, y, w, h


def _rect_to_mask(rect, width: int, height: int) -> np.ndarray:
    x, y, w, h = rect
    if max(x + w, y + h) > 1.5:  # absolute pixel coordinates
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + w)), int(round(y + h))
    else:
        x0, y0 = int(round(x * width)), int(round(y * height))
        x1, y1 = int(round((x + w) * width)), int(round((y + h) * height))
    mask = np.zeros((height, width), dtype=bool)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    mask[y0:y1, x0:x1] = True
    return mask


def build_roi_mask(
    width: int,
    height: int,
    roi: str | None = None,
    exclude_mask: str | None = None,
) -> np.ndarray:
    """Return a HxW boolean array: True = this pixel counts towards statistics."""
    mask = np.ones((height, width), dtype=bool)

    if roi:
        spec = roi.strip()
        if spec.startswith("center:"):
            frac = float(spec.split(":", 1)[1])
            if not 0 < frac <= 1:
                raise ValueError("--roi center:F requires 0 < F <= 1")
            w = frac * width
            h = frac * height
            x0 = int(round((width - w) / 2))
            y0 = int(round((height - h) / 2))
            sub = np.zeros_like(mask)
            sub[y0 : y0 + int(round(h)), x0 : x0 + int(round(w))] = True
            mask &= sub
        else:
            mask &= _rect_to_mask(_parse_rect(spec), width, height)

    if exclude_mask:
        spec = str(exclude_mask).strip()
        if "," in spec and not Path(spec).exists():
            mask &= ~_rect_to_mask(_parse_rect(spec), width, height)
        else:
            mask &= ~load_png_mask(spec, width, height)

    if not mask.any():
        raise ValueError("ROI/exclusion mask leaves no pixels to measure")
    return mask


def load_png_mask(path: str, width: int, height: int) -> np.ndarray:
    """Load a PNG mask and resize it to the analysis resolution (white = True)."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read mask image: {path}")
    if img.shape != (height, width):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img >= 128
