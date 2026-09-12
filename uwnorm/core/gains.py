"""Gain composition and pixel application (F-03.6, F-03.7, F-04 last bullet).

The whole point of this module is that a pixel is touched **once**.  Exposure
and white balance are both multiplications in linear light, so they are folded
into a single per-frame, per-channel triplet::

    gain_R = exp(exposure_log_gain) * wb_gain_R
    gain_G = exp(exposure_log_gain)
    gain_B = exp(exposure_log_gain) * wb_gain_B

Applying them separately would mean two round trips through the 8-bit
quantiser; folding them keeps the pipeline at one linearise / one gain / one
re-encode (premise P-2: there is no headroom to waste).

Highlights get a soft shoulder rather than a hard clamp.  A hard ``min(x, 1)``
turns a gained highlight into a flat plateau and, because the three channels
clip at different gains, tints it (the risk table's "colour shift on saturated
highlights").  The roll-off below is C1-continuous at the knee, so no contour
appears where it starts.
"""

from __future__ import annotations

import numpy as np

from . import ops


def compose(
    exposure_log_gain: np.ndarray,
    wb_gain_r: np.ndarray,
    wb_gain_b: np.ndarray,
) -> np.ndarray:
    """Per-frame (N, 3) linear gain table."""
    exp_gain = np.exp(np.asarray(exposure_log_gain, dtype=np.float64))
    gr = exp_gain * np.asarray(wb_gain_r, dtype=np.float64)
    gg = exp_gain
    gb = exp_gain * np.asarray(wb_gain_b, dtype=np.float64)
    return np.stack([gr, gg, gb], axis=-1)


def soft_clip(x, knee: float):
    """Compress ``[knee, inf)`` into ``[knee, 1)`` with a tanh shoulder.

    ``y = k + (1-k)*tanh((x-k)/(1-k))`` has value ``k`` and slope 1 at ``x = k``,
    so it joins the identity smoothly and is monotonic everywhere above it.
    """
    k = float(knee)
    if k >= 1.0:
        return ops.clip(x, 0.0, 1.0)
    over = (x - k) / (1.0 - k)
    rolled = k + (1.0 - k) * ops.tanh(over)
    return ops.where(x > k, rolled, ops.clip(x, 0.0, None))


def apply_gain(linear_rgb, gain3, knee: float):
    """Multiply linear RGB by a 3-vector gain and roll off the highlights."""
    g = ops.asarray_like(linear_rgb, gain3)
    out = linear_rgb * g
    out = soft_clip(out, knee)
    return ops.clip(out, 0.0, 1.0)


def build_lut(
    gain3,
    knee: float,
    trc: str,
    out_bits: int = 8,
    in_bits: int = 8,
) -> np.ndarray:
    """Collapse the whole per-pixel chain into one lookup table per channel.

    For a given frame the chain is

        encoded -> EOTF -> x gain_c -> soft clip -> OETF -> quantise

    and, crucially, *every step is a scalar function of that channel's own
    value*.  The input is 8-bit, so each channel has only 256 possible inputs:
    the entire pipeline is therefore 256 evaluations per channel per frame
    instead of one per pixel, and applying it is three array lookups.

    On a 1080p frame that replaces ~37 million transcendental operations with
    768, which is what brings the CPU-only path inside the N-02 budget.  The
    result is *identical*, not approximated: the table is built with the same
    functions, in the same dtype, on the same input values.

    The one case this cannot express is ``--red-compensation``, where red
    depends on the green pixel next to it; that path stays element-wise.
    """
    from . import color

    levels = 1 << in_bits
    peak_in = float(levels - 1)
    peak_out = float((1 << out_bits) - 1)
    dtype = np.uint8 if out_bits <= 8 else np.uint16

    encoded = (np.arange(levels, dtype=np.float32) / peak_in).astype(np.float32)
    linear = color.eotf(encoded, trc)
    lut = np.empty((3, levels), dtype=dtype)
    for c in range(3):
        gained = soft_clip(linear * np.float32(gain3[c]), knee)
        out = color.oetf(np.clip(gained, 0.0, 1.0), trc)
        lut[c] = np.clip(out * peak_out + 0.5, 0.0, peak_out).astype(dtype)
    return lut


def to_cv_lut(lut: np.ndarray) -> np.ndarray:
    """(3, 256) -> the (1, 256, 3) interleaved layout cv2.LUT expects."""
    return np.ascontiguousarray(lut.T.reshape(1, lut.shape[1], 3))


def apply_lut(frame, lut):
    """Map an HxWx3 integer frame through a per-channel lookup table.

    For numpy frames this goes through ``cv2.LUT``, which reads the interleaved
    RGB layout natively.  The obvious ``np.take(lut[c], frame[..., c])`` has to
    walk a stride-3 view three times and manages only 8.6 fps on a 4K frame;
    cv2.LUT does the same work at 113 fps, which is what moves the bottleneck
    off this stage and onto the encoder where it belongs.
    """
    if ops.is_torch(frame):
        import torch

        out = torch.empty(frame.shape, dtype=lut.dtype, device=frame.device)
        for c in range(3):
            out[..., c] = lut[c][frame[..., c].long()]
        return out

    import cv2

    cv_lut = lut if lut.ndim == 3 else to_cv_lut(lut)
    return cv2.LUT(np.ascontiguousarray(frame), cv_lut)


def interpolate_to_frames(
    values: np.ndarray, source_index: np.ndarray, n_frames: int
) -> np.ndarray:
    """Spread a strided analysis series over every frame of the apply pass.

    With ``--analysis-stride N`` the statistics only exist every Nth frame; the
    gains in between are linearly interpolated.  Gains are smooth by
    construction (they are low-pass filtered), so linear interpolation between
    measured points introduces no visible error.
    """
    values = np.asarray(values, dtype=np.float64)
    source_index = np.asarray(source_index, dtype=np.float64)
    target = np.arange(n_frames, dtype=np.float64)
    if values.ndim == 1:
        return np.interp(target, source_index, values)
    cols = [np.interp(target, source_index, values[:, c]) for c in range(values.shape[1])]
    return np.stack(cols, axis=-1)


def ev(gain: float | np.ndarray) -> float | np.ndarray:
    """Linear gain -> stops, for human-readable logs."""
    return np.log2(np.asarray(gain, dtype=np.float64))
