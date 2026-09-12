"""Transfer functions (EOTF / OETF).

F-02: every measurement and every gain application happens in *linear light*.
The encoded MP4 carries a non-linear, gamma-encoded signal; multiplying that
signal directly would change contrast as well as brightness and would make the
frame-to-frame ratios meaningless.  So the pipeline is always

    encoded -> EOTF (decode to linear) -> measure / gain -> OETF -> encoded

All functions take and return float arrays normalised to [0, 1] and work on
either numpy arrays or torch tensors (see :mod:`uwnorm.core.ops`).
"""

from __future__ import annotations

from . import ops

TRCS = ("srgb", "bt709", "gamma22", "gamma24", "linear")

# Transfer characteristics names as reported by ffprobe -> our internal name.
FFMPEG_TRC_MAP = {
    "bt709": "bt709",
    "smpte170m": "bt709",
    "bt470bg": "gamma22",
    "gamma22": "gamma22",
    "gamma28": "gamma24",
    "iec61966-2-1": "srgb",
    "srgb": "srgb",
    "linear": "linear",
    "unknown": None,
    "reserved": None,
}


def _srgb_eotf(x):
    x = ops.clip(x, 0.0, 1.0)
    lo = x / 12.92
    hi = ops.power((x + 0.055) / 1.055, 2.4)
    return ops.where(x <= 0.04045, lo, hi)


def _srgb_oetf(x):
    x = ops.clip(x, 0.0, 1.0)
    lo = x * 12.92
    hi = 1.055 * ops.power(ops.maximum(x, 1e-8), 1.0 / 2.4) - 0.055
    return ops.where(x <= 0.0031308, lo, hi)


def _bt709_eotf(x):
    # Inverse of the BT.709 OETF (the camera-side curve).
    x = ops.clip(x, 0.0, 1.0)
    lo = x / 4.5
    hi = ops.power((x + 0.099) / 1.099, 1.0 / 0.45)
    return ops.where(x < 0.081, lo, hi)


def _bt709_oetf(x):
    x = ops.clip(x, 0.0, 1.0)
    lo = x * 4.5
    hi = 1.099 * ops.power(ops.maximum(x, 1e-8), 0.45) - 0.099
    return ops.where(x < 0.018, lo, hi)


def _pure_gamma_eotf(gamma):
    def f(x):
        return ops.power(ops.clip(x, 0.0, 1.0), gamma)

    return f


def _pure_gamma_oetf(gamma):
    def f(x):
        return ops.power(ops.clip(x, 0.0, 1.0), 1.0 / gamma)

    return f


def _identity(x):
    return ops.clip(x, 0.0, 1.0)


_EOTF = {
    "srgb": _srgb_eotf,
    "bt709": _bt709_eotf,
    "gamma22": _pure_gamma_eotf(2.2),
    "gamma24": _pure_gamma_eotf(2.4),
    "linear": _identity,
}

_OETF = {
    "srgb": _srgb_oetf,
    "bt709": _bt709_oetf,
    "gamma22": _pure_gamma_oetf(2.2),
    "gamma24": _pure_gamma_oetf(2.4),
    "linear": _identity,
}


def eotf(x, trc: str):
    """Encoded [0,1] -> linear light [0,1]."""
    try:
        return _EOTF[trc](x)
    except KeyError:  # pragma: no cover - guarded by the CLI
        raise ValueError(f"unknown transfer characteristic: {trc!r}")


def oetf(x, trc: str):
    """Linear light [0,1] -> encoded [0,1]."""
    try:
        return _OETF[trc](x)
    except KeyError:  # pragma: no cover - guarded by the CLI
        raise ValueError(f"unknown transfer characteristic: {trc!r}")


# BT.709 luminance weights, valid in the *linear* domain only.
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def luminance(rgb_linear):
    """Linear RGB (..., 3) -> linear relative luminance (...)."""
    w = ops.asarray_like(rgb_linear, LUMA_WEIGHTS)
    return (rgb_linear * w).sum(-1)
