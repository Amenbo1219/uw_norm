"""Robust per-frame statistics (F-03.1, F-04).

Everything here runs on the *analysis* resolution frame (long edge ~384 px), so
plain numpy is both fast enough and bit-for-bit reproducible (N-01).

The recurring idea is that a naive ``mean()`` is the wrong estimator for this
job: blown highlights are clipped (they no longer carry the scene's brightness),
crushed shadows are mostly noise, and a fish swimming through the frame is not
an exposure change.  Every estimator below therefore works on a *validity mask*
and uses order statistics rather than plain averages.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def valid_mask(encoded_rgb: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """Pixels that are neither clipped nor crushed, in the *encoded* domain.

    Clipping has to be judged on the encoded values because that is where the
    quantiser lives: once a channel reaches 255 the true scene value is unknown
    and the pixel must not influence any estimate (risk table: "saturated
    highlights").
    """
    return np.all((encoded_rgb <= hi) & (encoded_rgb >= lo), axis=-1)


def trimmed_mean(values: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> float:
    """Mean of the values between two percentiles (empty -> nan)."""
    if values.size == 0:
        return float("nan")
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    kept = values[(values >= lo) & (values <= hi)]
    if kept.size == 0:
        return float(np.mean(values))
    return float(np.mean(kept))


def trimmed_geometric_mean(
    values: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5
) -> float:
    """Geometric mean of the trimmed values.

    Exposure is multiplicative, so the natural centre of a luminance
    distribution is its geometric (log-domain) mean: doubling the exposure
    doubles this statistic regardless of the scene's histogram shape.
    """
    if values.size == 0:
        return float("nan")
    lo, hi = np.percentile(values, [lo_pct, hi_pct])
    kept = values[(values >= lo) & (values <= hi)]
    if kept.size == 0:
        kept = values
    return float(np.exp(np.mean(np.log(np.maximum(kept, EPS)))))


def minkowski_mean(channel: np.ndarray, p: float) -> float:
    """(mean(x^p))^(1/p) - the shades-of-gray illuminant estimator.

    p=1 degenerates to gray-world, p->inf to max-RGB; p in [5, 6] is the
    sweet spot for underwater footage where the scene average is strongly
    biased by the water column itself.
    """
    if channel.size == 0:
        return float("nan")
    if p == 1.0:
        return float(np.mean(channel))
    x = np.maximum(channel.astype(np.float64), 0.0)
    return float(np.power(np.mean(np.power(x, p)), 1.0 / p))


def percentile_mean(channel: np.ndarray, pct: float = 99.0) -> float:
    """Mean of the brightest ``100-pct`` percent - the max-RGB estimator.

    Using a percentile rather than the single maximum keeps one hot pixel or a
    glint of sunlight from defining the white point.
    """
    if channel.size == 0:
        return float("nan")
    thr = np.percentile(channel, pct)
    kept = channel[channel >= thr]
    if kept.size == 0:
        return float(np.max(channel))
    return float(np.mean(kept))


def ratio_estimate(
    cur_linear: np.ndarray,
    prev_linear: np.ndarray,
    cur_valid: np.ndarray,
    prev_valid: np.ndarray,
    floor: float = 1e-3,
) -> float:
    """median(I_t / I_{t-1}) over pixels usable in *both* frames (F-03.2).

    The median is what separates an exposure change from a scene change: when
    the auto-exposure moves, *every* pixel scales by the same factor, so the
    median of the per-pixel ratio is that factor.  When a fish swims past, only
    a minority of pixels change and the median ignores them.

    Returns 1.0 (i.e. "no exposure change") when too few pixels qualify.
    """
    both = cur_valid & prev_valid & (prev_linear > floor) & (cur_linear > floor)
    n = int(np.count_nonzero(both))
    if n < 32:
        return 1.0
    ratio = cur_linear[both] / prev_linear[both]
    return float(np.median(ratio))


def gain_invariant_histogram(
    linear_rgb: np.ndarray, mask: np.ndarray, bins: int = 24, span: float = 4.0
) -> np.ndarray:
    """Scene descriptor for cut detection, blind to per-channel gain.

    A histogram of raw pixel values is a poor cut detector *for this tool*: a
    half-stop auto-exposure step shifts the whole histogram and scores as high
    as a real cut (measured: 0.95 correlation between cut score and exposure
    change).  A false cut is expensive - it resets the exposure integrator and
    stops the smoother from crossing it - so the detector has to ignore the
    very thing the rest of the pipeline is measuring.

    Each channel is therefore divided by its own mean before binning, in log2
    units.  Any diagonal gain - which is precisely what exposure and white
    balance are - cancels out, while a change of scene does not.
    """
    out = np.zeros(bins * 3, dtype=np.float64)
    sel = linear_rgb[mask]
    if sel.size == 0:
        return out
    for c in range(3):
        values = sel[:, c].astype(np.float64)
        mean = float(np.mean(values))
        if mean <= EPS:
            continue
        z = np.log2(np.maximum(values, EPS) / mean)
        hist, _ = np.histogram(np.clip(z, -span, span), bins=bins, range=(-span, span))
        total = hist.sum()
        if total:
            out[c * bins : (c + 1) * bins] = hist / total
    return out / 3.0


def gain_invariant_thumbnail(
    linear_rgb: np.ndarray, size: int = 16, span: float = 4.0
) -> np.ndarray:
    """Tiny log-domain, per-channel mean-normalised thumbnail of the frame.

    The histogram above is order-less, so two different scenes that happen to
    share a brightness distribution score as identical.  This keeps a coarse
    *layout*, which a cut nearly always changes.  16x16 is deliberately small:
    a camera pan barely moves it, while a cut rearranges it completely.
    """
    import cv2

    h, w = linear_rgb.shape[:2]
    small = cv2.resize(
        np.asarray(linear_rgb, dtype=np.float32), (size, size), interpolation=cv2.INTER_AREA
    )
    out = np.empty_like(small, dtype=np.float64)
    for c in range(3):
        ch = small[..., c].astype(np.float64)
        mean = float(np.mean(ch))
        if mean <= EPS:
            out[..., c] = 0.0
            continue
        out[..., c] = np.clip(np.log2(np.maximum(ch, EPS) / mean), -span, span)
    return out


def channel_ratios(
    cur_linear: np.ndarray,
    prev_linear: np.ndarray,
    cur_valid: np.ndarray,
    prev_valid: np.ndarray,
    floor: float = 1e-3,
) -> np.ndarray:
    """Per-channel median(I_t / I_{t-1}) -> array of 3 ratios (F-04 stabilisation).

    The colour equivalent of :func:`ratio_estimate`.  If the camera's auto white
    balance nudges red up between two frames, *every* red pixel scales by the
    same factor and the median catches it; if instead a red fish swims into
    frame, only a minority of pixels change and the median ignores it.  That is
    the distinction a per-frame illuminant estimator cannot make on its own.
    """
    out = np.ones(3, dtype=np.float64)
    for c in range(3):
        cur_c = cur_linear[..., c]
        prev_c = prev_linear[..., c]
        both = (
            cur_valid & prev_valid & (prev_c > floor) & (cur_c > floor)
        )
        if int(np.count_nonzero(both)) < 32:
            continue
        out[c] = float(np.median(cur_c[both] / prev_c[both]))
    return out
