"""Temporal exposure normalisation (F-03).

The problem
-----------
An MP4 carries no per-frame ISO/shutter metadata (premise P-1), so the exposure
has to be *measured from the pixels*.  Two measurements are available and each
is wrong in a different way:

``L_t`` - the robust log-luminance of frame *t*.
    Absolute and drift-free, but contaminated by scene content: swim towards a
    dark wall and L drops even though the camera never changed exposure.

``D_t`` - the running sum of log(median(I_t / I_{t-1})).
    Immune to scene content (the median ratio ignores anything that is not a
    global scale change), but it is an integral of noisy increments, so it
    slowly walks away from the truth over thousands of frames (risk table:
    "drift from accumulating frame ratios").

The fix: a complementary filter
-------------------------------
Take the *high* frequencies from D (where it is accurate) and the *low*
frequencies from L (where it is drift-free)::

    E_t = D_t - lowpass(D_t - L_t)

The correction ``D - L`` is by construction slowly varying, so this re-anchors
the accumulated curve onto the absolute measurement without importing L's
frame-to-frame scene noise.  ``E_t`` is then a drift-free log-exposure curve on
an absolute scale, comparable across the whole clip.

Correcting it
-------------
``deflicker`` removes only what changes faster than ``--exposure-tau`` and keeps
the rest (the dive's real lighting arc).  ``lock`` flattens the entire curve to
one target.  Both produce a log-gain that is clamped to +/- ``--max-exposure-gain``
EV before it ever touches a pixel.
"""

from __future__ import annotations

import numpy as np

from . import smooth

LN2 = float(np.log(2.0))


def cumulative_log_exposure(log_ratios: np.ndarray, segments: smooth.Segments) -> np.ndarray:
    """Integrate per-frame log ratios into a curve, restarting at every cut.

    Restarting matters: across a cut ``median(I_t/I_{t-1})`` compares two
    unrelated scenes, so its value is meaningless and must not enter the sum.
    """
    log_ratios = np.asarray(log_ratios, dtype=np.float64)
    out = np.zeros_like(log_ratios)
    for start, end in segments:
        chunk = log_ratios[start:end].copy()
        if chunk.size:
            chunk[0] = 0.0  # the first frame of a shot has no predecessor
            out[start:end] = np.cumsum(chunk)
    return out


def anchor(
    cumulative: np.ndarray,
    log_luma: np.ndarray,
    fps: float,
    segments: smooth.Segments,
    anchor_tau: float = 4.0,
) -> np.ndarray:
    """Complementary filter: high frequencies from D, low frequencies from L."""
    cumulative = np.asarray(cumulative, dtype=np.float64)
    log_luma = np.asarray(log_luma, dtype=np.float64)
    residual = cumulative - log_luma
    return cumulative - smooth.lowpass(residual, anchor_tau, fps, segments)


def exposure_log_gain(
    curve: np.ndarray,
    mode: str,
    tau: float,
    fps: float,
    segments: smooth.Segments,
    max_ev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-gain to apply to each frame, plus the per-frame clamp flags.

    Returns ``(log_gain, clamped)`` where ``log_gain`` is natural-log domain
    (multiply linear light by ``exp(log_gain)``).
    """
    curve = np.asarray(curve, dtype=np.float64)
    if mode == "off":
        return np.zeros_like(curve), np.zeros_like(curve, dtype=bool)

    if mode == "deflicker":
        target = smooth.lowpass(curve, tau, fps, segments)
    elif mode == "lock":
        # One target for the whole clip: the robust centre of the curve.  Using
        # a global (not per-segment) median keeps separate shots consistent with
        # each other, which is the point of "lock".
        target = smooth.global_constant(curve, robust=True)
    else:
        raise ValueError(f"unknown exposure mode: {mode!r}")

    log_gain = target - curve
    limit = float(max_ev) * LN2
    clamped = np.abs(log_gain) > limit
    return np.clip(log_gain, -limit, limit), clamped


def predicted_curve(curve: np.ndarray, log_gain: np.ndarray) -> np.ndarray:
    """What the exposure curve becomes once the gains are applied.

    Exact, because the gain is a pure multiplication in linear light: the
    robust log-luminance of the corrected frame is ``E_t + log_gain_t``.  This
    is what the report and the A-1 check compare against the original.
    """
    return np.asarray(curve, dtype=np.float64) + np.asarray(log_gain, dtype=np.float64)
