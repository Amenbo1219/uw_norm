"""Segment-aware temporal filtering (F-03.4, F-05, F-06).

Three rules govern every filter in this module:

1. **Never filter across a scene cut.**  A cut is a genuine discontinuity; a
   filter that straddles it drags the colour of the previous shot into the next
   one for half a window length (acceptance criterion A-4).  Every series is
   therefore split at the cuts and each segment filtered independently.
2. **Reflect at the edges.**  Zero-padding a log-gain series would pull the
   first and last seconds of each segment towards a gain of 1.0, producing a
   visible ramp.
3. **Zero phase.**  A causal filter would delay the correction relative to the
   flicker it is correcting, which makes the flicker worse, not better.  Every
   filter here is symmetric (``filtfilt`` / Gaussian / Savitzky-Golay).

The low/high split uses a zero-phase Butterworth rather than a Gaussian: the
Gaussian's roll-off is so gentle that with a 1.5 s cut-off it still removes a
quarter of a genuine 5 s brightness change, which would flatten exactly the
scene dynamics requirement F-03.4 asks to preserve.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, filtfilt, savgol_filter

Segments = list[tuple[int, int]]

#: Order of the Butterworth prototype; ``filtfilt`` doubles it to 4th order.
BUTTER_ORDER = 2

# A Gaussian of temporal sigma s attenuates a sinusoid of frequency f by
# exp(-2*pi^2*f^2*s^2).  With s = K*tau, K = sqrt(ln2/2)/pi, a sinusoid whose
# period equals tau comes out at half amplitude.  Used only as the fallback for
# segments too short for filtfilt.
GAUSS_TAU_K = 0.18739


def segments_from_cuts(n: int, cuts: list[int]) -> Segments:
    """Turn cut indices (first frame of each new shot) into [start, end) ranges."""
    bounds = [0] + sorted({int(c) for c in cuts if 0 < int(c) < n}) + [n]
    return [
        (bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i + 1] > bounds[i]
    ]


def _apply_per_segment(x: np.ndarray, segments: Segments, fn) -> np.ndarray:
    out = np.array(x, dtype=np.float64, copy=True)
    for start, end in segments:
        chunk = out[start:end]
        if chunk.size:
            out[start:end] = fn(chunk)
    return out


def tau_to_sigma(tau_seconds: float, fps: float) -> float:
    """Time constant in seconds -> Gaussian sigma in frames."""
    return max(GAUSS_TAU_K * float(tau_seconds) * float(fps), 0.5)


def tau_to_savgol_window(tau_seconds: float, fps: float, order: int = 2) -> int:
    """Time constant -> Savitzky-Golay window length in frames.

    Uses Schafer's approximation of the SG cut-off, fc ~ (order+1)/(3.2*W-4.6)
    cycles/sample, solved for W at fc = 1/(tau*fps).
    """
    fc = 1.0 / max(float(tau_seconds) * float(fps), 1e-6)
    window = ((order + 1) / max(fc, 1e-9) + 4.6) / 3.2
    return max(3, int(round(window)) | 1)


def lowpass(
    x: np.ndarray,
    tau_seconds: float,
    fps: float,
    segments: Segments | None = None,
) -> np.ndarray:
    """Keep only what changes more slowly than ``tau`` seconds.

    Zero-phase Butterworth at fc = 1/tau.  Segments too short to support the
    filter's padding fall back to a Gaussian, and very short ones to their own
    median (a two-frame shot has no meaningful frequency content).
    """
    x = np.asarray(x, dtype=np.float64)
    if segments is None:
        segments = [(0, len(x))]

    nyquist = 0.5 * float(fps)
    fc = 1.0 / max(float(tau_seconds), 1e-6)
    wn = min(max(fc / nyquist, 1e-6), 0.99)
    b, a = butter(BUTTER_ORDER, wn, btype="low")
    padlen = 3 * max(len(a), len(b))
    sigma = tau_to_sigma(tau_seconds, fps)

    def fn(chunk):
        if chunk.size <= padlen:
            if chunk.size < 3:
                return np.full_like(chunk, float(np.median(chunk)))
            s = min(sigma, max(chunk.size / 3.0, 0.5))
            return gaussian_filter1d(chunk, sigma=s, mode="reflect", truncate=4.0)
        return filtfilt(b, a, chunk, padtype="odd", padlen=padlen)

    return _apply_per_segment(x, segments, fn)


def highpass(
    x: np.ndarray,
    tau_seconds: float,
    fps: float,
    segments: Segments | None = None,
) -> np.ndarray:
    """The complement of :func:`lowpass` - the part treated as AE hunting."""
    x = np.asarray(x, dtype=np.float64)
    return x - lowpass(x, tau_seconds, fps, segments)


def highpass_energy(
    x: np.ndarray,
    tau_seconds: float,
    fps: float,
    segments: Segments | None = None,
) -> float:
    """Std-dev of the components faster than ``tau``.

    A-1 / A-2 are phrased as "the high-frequency variation must drop by >= 80%",
    and this is the number both sides of that comparison are measured with.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return 0.0
    return float(np.std(highpass(x, tau_seconds, fps, segments)))


def savgol(
    x: np.ndarray, window: int, segments: Segments | None = None, order: int = 2
) -> np.ndarray:
    """Savitzky-Golay smoothing: preserves ramps (a real lighting change) better
    than a moving average of the same width, which is why it is the default for
    the WB series where slow drift must survive."""
    x = np.asarray(x, dtype=np.float64)
    if segments is None:
        segments = [(0, len(x))]

    def fn(chunk):
        w = int(window) | 1
        if chunk.size % 2 == 0:
            w = min(w, max(chunk.size - 1, 1))
        else:
            w = min(w, chunk.size)
        if w <= order + 1 or chunk.size < 3:
            return np.full_like(chunk, float(np.median(chunk)))
        return savgol_filter(chunk, window_length=w, polyorder=order, mode="nearest")

    return _apply_per_segment(x, segments, fn)


def median_gauss(
    x: np.ndarray, window: int, sigma: float, segments: Segments | None = None
) -> np.ndarray:
    """Median filter (kills one-frame outliers) followed by a Gaussian."""
    x = np.asarray(x, dtype=np.float64)
    if segments is None:
        segments = [(0, len(x))]

    def fn(chunk):
        if chunk.size < 3:
            return np.full_like(chunk, float(np.median(chunk)))
        w = max(3, int(window) | 1)
        w = min(w, chunk.size if chunk.size % 2 == 1 else chunk.size - 1)
        med = median_filter(chunk, size=max(w, 1), mode="reflect")
        s = min(sigma, max(med.size / 3.0, 0.5))
        return gaussian_filter1d(med, sigma=s, mode="reflect", truncate=4.0)

    return _apply_per_segment(x, segments, fn)


def smooth_series(
    x: np.ndarray,
    method: str,
    tau_seconds: float,
    fps: float,
    segments: Segments | None = None,
) -> np.ndarray:
    """Dispatch to the configured smoother (``--smoother``)."""
    if method == "savgol":
        return savgol(x, tau_to_savgol_window(tau_seconds, fps), segments)
    if method == "median-gauss":
        sigma = tau_to_sigma(tau_seconds, fps)
        return median_gauss(x, max(3, int(round(sigma)) | 1), sigma, segments)
    raise ValueError(f"unknown smoother: {method!r}")


def segment_constant(x: np.ndarray, segments: Segments, robust: bool = True) -> np.ndarray:
    """Replace each segment by its (robust) centre."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    for start, end in segments:
        chunk = x[start:end]
        out[start:end] = np.median(chunk) if robust else np.mean(chunk)
    return out


def global_constant(x: np.ndarray, robust: bool = True) -> np.ndarray:
    """Replace the whole series by its (robust) centre - the ``fixed`` modes."""
    x = np.asarray(x, dtype=np.float64)
    value = np.median(x) if robust else np.mean(x)
    return np.full_like(x, value)
