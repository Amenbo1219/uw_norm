"""White balance estimation and temporal stabilisation (F-04, F-05).

Estimation happens in linear light on the analysis frames.  Every method boils
down to "guess what the illuminant's RGB is, then divide it out", and all of
them are normalised so that **G stays at 1.0**: the green channel is the one
water attenuates least and the one the sensor samples most densely, so scaling
red and blue relative to green keeps the overall exposure decision (F-03) and
the colour decision independent of each other.

Method selection (P-5, underwater specifics):

``shades-of-gray``  Minkowski mean with p~5.  The default.  Gray-world (p=1) is
                    dominated by the blue-green water column itself and
                    over-corrects; max-RGB (p->inf) latches onto a single glint.
                    p in [5, 6] sits between the two and is the standard
                    recommendation for underwater scenes.
``max-rgb``         Mean of the brightest percentile - good when the frame
                    genuinely contains white sand or a dive slate.
``gray-world``      p=1, kept as a comparison baseline.
``reference-frame`` Take the white point from a grey card the diver held up
                    once, and use it for the whole clip.

``red-compensation`` is not an estimator but a pre-processing step (Ancuti et
al.): red is the first wavelength water swallows, and past a few metres there is
often no red signal left to scale up - multiplying it would only amplify noise.
The compensation borrows the missing red from green where red is dark:

    R' = R + alpha * (mean(G) - mean(R)) * (1 - R) * G

The ``(1 - R)`` term confines the effect to pixels that are dark in red, so
already-red subjects are left alone.
"""

from __future__ import annotations

import numpy as np

from . import ops, stats

EPS = 1e-6


def channel_means(
    linear_rgb: np.ndarray, mask: np.ndarray, method: str, p: float
) -> np.ndarray:
    """Per-channel illuminant estimate for one frame -> array of 3 floats."""
    sel = linear_rgb[mask]
    if sel.size == 0:
        return np.array([np.nan] * 3)
    if method == "gray-world":
        return np.array([stats.minkowski_mean(sel[:, c], 1.0) for c in range(3)])
    if method == "shades-of-gray":
        return np.array([stats.minkowski_mean(sel[:, c], p) for c in range(3)])
    if method == "max-rgb":
        return np.array([stats.percentile_mean(sel[:, c], 99.0) for c in range(3)])
    if method == "reference-frame":
        # The white point comes from a user-picked frame/ROI; per-frame means are
        # still recorded so the report can show what the footage was doing.
        return np.array([stats.minkowski_mean(sel[:, c], p) for c in range(3)])
    raise ValueError(f"unknown wb method: {method!r}")


def gains_from_means(means: np.ndarray) -> tuple[float, float]:
    """Illuminant RGB -> (gain_R, gain_B), normalised so that gain_G == 1."""
    r, g, b = (float(m) for m in means)
    if not np.isfinite([r, g, b]).all() or r <= EPS or b <= EPS or g <= EPS:
        return 1.0, 1.0
    return g / r, g / b


def illuminant_curve(
    log_imbalance: np.ndarray,
    log_ratio_delta: np.ndarray,
    method: str,
    tau: float,
    fps: float,
    segments,
) -> np.ndarray:
    """Best estimate of the illuminant over time, as a log *gain to apply*.

    Exactly the complementary filter used for exposure (see
    :mod:`uwnorm.core.exposure`), because the two problems are the same shape:

    ``log_imbalance`` (``u_t = log(mean_R / mean_G)``)
        Absolute and drift-free, but it moves whenever the *scene* changes
        colour, not only when the illuminant does.  Smoothing it cannot fix
        that - a smoothed scene-driven estimate is still scene-driven.

    ``log_ratio_delta`` (``log rho_R - log rho_G`` per frame)
        The robust frame-to-frame change in the red/green balance.  Blind to
        scene content, but an integral, so it drifts.

    Taking the fast band from the integral and the slow band from the absolute
    estimate gives a curve that is trustworthy at *both* ends, which is what
    makes it possible to actually remove colour flicker instead of merely
    smoothing the correction applied to it.

    Returns the log gain that would neutralise the frame (i.e. ``-u``).
    """
    from . import smooth

    u = np.asarray(log_imbalance, dtype=np.float64)
    delta = np.asarray(log_ratio_delta, dtype=np.float64)

    # F-05: the configured smoother is applied to the raw estimate series.  Only
    # its slow band survives the anchoring below, so this is pure noise rejection.
    u_smooth = smooth.smooth_series(u, method, tau, fps, segments)

    integrated = np.zeros_like(delta)
    for start, stop in segments:
        chunk = delta[start:stop].copy()
        if chunk.size:
            chunk[0] = 0.0
            integrated[start:stop] = np.cumsum(chunk)
    # Re-anchor the integral onto the absolute estimate.
    anchored = integrated - smooth.lowpass(integrated - u_smooth, tau, fps, segments)
    return -anchored


def wb_log_gain(
    curve: np.ndarray,
    mode: str,
    tau: float,
    fps: float,
    segments,
    max_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Illuminant curve -> log gain actually applied, plus the clamp flags.

    ``drift``  Follow the illuminant all the way: apply the whole curve, so the
               output is neutral in the shallows *and* at depth.  This is what
               P-5 asks for - a single white point across a dive that changes
               depth looks wrong somewhere.
    ``fixed``  Keep one white point for the clip (the robust median of the
               curve) and only remove what changes faster than ``--wb-tau``.
               The output then still gets bluer as the dive gets deeper, the
               way a camera locked to one Kelvin value would record it, but
               without the auto-white-balance hunting.
    ``off``    No colour change at all.

    Both active modes remove the fast band, which is the part the eye reads as
    colour flicker (A-2).
    """
    from . import smooth

    curve = np.asarray(curve, dtype=np.float64)
    if mode == "off":
        return np.zeros_like(curve), np.zeros_like(curve, dtype=bool)
    if mode == "drift":
        applied = curve
    elif mode == "fixed":
        slow = smooth.lowpass(curve, tau, fps, segments)
        applied = (curve - slow) + float(np.median(curve))
    else:
        raise ValueError(f"unknown wb mode: {mode!r}")

    limit = float(np.log(max_gain))
    clamped = np.abs(applied) > limit
    return np.clip(applied, -limit, limit), clamped


def red_compensate(linear_rgb, alpha: float, mean_r: float, mean_g: float):
    """Ancuti red-channel compensation, applied per pixel in linear light."""
    if alpha <= 0.0:
        return linear_rgb
    r = linear_rgb[..., 0]
    g = linear_rgb[..., 1]
    delta = float(alpha) * (float(mean_g) - float(mean_r))
    if delta <= 0.0:
        return linear_rgb
    corrected_r = r + delta * (1.0 - r) * g
    if ops.is_torch(linear_rgb):
        import torch

        return torch.stack((corrected_r, g, linear_rgb[..., 2]), dim=-1)
    return np.stack((corrected_r, g, linear_rgb[..., 2]), axis=-1)


def reference_gains(
    linear_rgb: np.ndarray, mask: np.ndarray, method: str, p: float
) -> tuple[float, float]:
    """Gains taken from a single reference frame's ROI (a grey card)."""
    means = channel_means(linear_rgb, mask, "max-rgb" if method == "max-rgb" else "gray-world", p)
    return gains_from_means(means)
