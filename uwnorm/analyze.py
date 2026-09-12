"""Pass 1 - analysis (F-01, 5.1).

Decodes the clip once at a small resolution and reduces every frame to a
handful of robust numbers, then turns those series into the gain curves the
apply pass will use.  Nothing but scalars is retained between frames (plus one
previous frame for the ratio estimate), so peak memory is independent of clip
length (N-03).

Order of operations inside the loop matters and is not arbitrary:

    decode -> /255 -> validity mask (encoded domain, where clipping is defined)
           -> EOTF (linear light, where gains are meaningful)
           -> red compensation (optional, before the illuminant is estimated)
           -> robust statistics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config, parse_range, parse_time
from .core import color, cuts as cuts_mod, exposure, roi as roi_mod
from .core import smooth, stats, wb as wb_mod
from .io import decode as decode_mod
from .io.probe import MediaInfo, probe, resolve_trc
from .logging_utils import Timer, get_logger, log_event

SCHEMA_VERSION = 2


def _round(seq, digits=6):
    return [round(float(v), digits) for v in np.asarray(seq, dtype=np.float64)]


def _reference_gains(path: str, cfg: Config, trc: str, info: MediaInfo) -> tuple[float, float]:
    """White point from one user-chosen frame (F-04 ``reference-frame``)."""
    t = parse_time(cfg.reference_frame)
    aw, ah = decode_mod.analysis_size(info.width, info.height, max(cfg.analysis_size, 128))
    with decode_mod.VideoSource(path, cfg.hwaccel, cfg.jobs) as src:
        frame = next(iter(src.frames(aw, ah, stride=1, start=t, end=t + 5.0)), None)
    if frame is None:
        raise ValueError(f"--reference-frame {cfg.reference_frame} is past the end of the clip")
    encoded = frame.array.astype(np.float32) / 255.0
    mask = roi_mod.build_roi_mask(aw, ah, cfg.reference_roi or cfg.roi, cfg.exclude_mask)
    mask &= stats.valid_mask(encoded, cfg.highlight_exclude, cfg.shadow_exclude)
    if not mask.any():
        raise ValueError("the reference ROI contains no usable (non-clipped) pixels")
    linear = color.eotf(encoded, trc)
    gr, gb = wb_mod.reference_gains(linear, mask, cfg.wb_method, cfg.wb_p)
    log_event("wb.reference", time=round(t, 3), gain_r=round(gr, 4), gain_b=round(gb, 4))
    return gr, gb


def analyze(path: str, cfg: Config, progress: bool = True) -> dict:
    """Run the analysis pass and return the analysis document."""
    logger = get_logger()
    info = probe(path)
    trc = resolve_trc(info, cfg.input_trc)
    aw, ah = decode_mod.analysis_size(info.width, info.height, cfg.analysis_size)
    roi_mask = roi_mod.build_roi_mask(aw, ah, cfg.roi, cfg.exclude_mask)

    start = end = None
    if cfg.preview_range:
        start, end = parse_range(cfg.preview_range)

    log_event(
        "analyze.start", source=Path(path).name, size=f"{info.width}x{info.height}",
        analysis_size=f"{aw}x{ah}", fps=round(info.fps, 3), trc=trc,
        stride=cfg.analysis_stride, vfr=info.is_vfr,
    )

    idx: list[int] = []
    times: list[float] = []
    ptss: list[int | None] = []
    log_luma: list[float] = []
    log_ratio: list[float] = []
    wb_raw_r: list[float] = []
    wb_raw_b: list[float] = []
    delta_r: list[float] = []
    delta_b: list[float] = []
    mean_r: list[float] = []
    mean_g: list[float] = []
    valid_frac: list[float] = []
    histograms: list[np.ndarray] = []
    thumbnails: list[np.ndarray] = []

    prev_luma: np.ndarray | None = None
    prev_valid: np.ndarray | None = None
    prev_linear: np.ndarray | None = None

    total = info.frame_count_estimate // max(cfg.analysis_stride, 1) or None
    with Timer("analyze"):
        with decode_mod.VideoSource(path, cfg.hwaccel, cfg.jobs) as src:
            iterator = src.frames(aw, ah, cfg.analysis_stride, start, end)
            if progress:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=total, unit="f", desc="analyze", leave=False)
            for frame in iterator:
                encoded = frame.array.astype(np.float32) / 255.0
                valid = stats.valid_mask(encoded, cfg.highlight_exclude, cfg.shadow_exclude)
                valid &= roi_mask
                if not valid.any():
                    # Fully blown or fully black frame: fall back to the ROI so
                    # the series stays defined rather than producing NaNs.
                    valid = roi_mask.copy()

                linear = color.eotf(encoded, trc)
                luma = color.luminance(linear)

                idx.append(frame.index)
                times.append(frame.time)
                ptss.append(frame.pts)
                valid_frac.append(float(valid.mean()))
                log_luma.append(
                    float(np.log(max(stats.trimmed_geometric_mean(luma[valid]), stats.EPS)))
                )

                if prev_luma is not None:
                    r = stats.ratio_estimate(luma, prev_luma, valid, prev_valid)
                    log_ratio.append(float(np.log(max(r, stats.EPS))))
                    rho = np.log(np.maximum(
                        stats.channel_ratios(linear, prev_linear, valid, prev_valid),
                        stats.EPS,
                    ))
                    delta_r.append(float(rho[0] - rho[1]))
                    delta_b.append(float(rho[2] - rho[1]))
                else:
                    log_ratio.append(0.0)
                    delta_r.append(0.0)
                    delta_b.append(0.0)
                prev_luma, prev_valid, prev_linear = luma, valid, linear

                sel = linear[valid]
                mr = float(np.mean(sel[:, 0]))
                mg = float(np.mean(sel[:, 1]))
                mean_r.append(mr)
                mean_g.append(mg)

                measured = linear
                if cfg.red_compensation > 0:
                    measured = wb_mod.red_compensate(linear, cfg.red_compensation, mr, mg)
                means = wb_mod.channel_means(measured, valid, cfg.wb_method, cfg.wb_p)
                gr, gb = wb_mod.gains_from_means(means)
                wb_raw_r.append(gr)
                wb_raw_b.append(gb)

                histograms.append(stats.gain_invariant_histogram(linear, valid))
                thumbnails.append(stats.gain_invariant_thumbnail(linear))

    n = len(idx)
    if n < 2:
        raise ValueError(f"{path}: only {n} frame(s) analysed - nothing to normalise")

    times_arr = np.asarray(times, dtype=np.float64)
    # Effective rate of the *analysis* series (stride-aware, VFR-aware).
    span = float(times_arr[-1] - times_arr[0])
    afps = (n - 1) / span if span > 1e-9 else info.fps / max(cfg.analysis_stride, 1)

    # --- scene cuts (F-06) -----------------------------------------------
    scores = cuts_mod.cut_scores(np.asarray(histograms), np.asarray(thumbnails))
    auto_cuts = cuts_mod.detect_cuts(scores, cfg.cut_threshold)
    manual_cuts = cuts_mod.times_to_indices(
        [parse_time(c) for c in cfg.cuts], times_arr
    ) if cfg.cuts else []
    cut_indices = sorted(set(auto_cuts) | set(manual_cuts))
    segments = smooth.segments_from_cuts(n, cut_indices)
    log_event("cuts.detected", auto=len(auto_cuts), manual=len(manual_cuts),
              segments=len(segments))

    # --- exposure (F-03) --------------------------------------------------
    log_luma_arr = np.asarray(log_luma, dtype=np.float64)
    cumulative = exposure.cumulative_log_exposure(np.asarray(log_ratio), segments)
    curve = exposure.anchor(cumulative, log_luma_arr, afps, segments)
    exp_gain, exp_clamped = exposure.exposure_log_gain(
        curve, cfg.exposure_mode, cfg.exposure_tau, afps, segments, cfg.max_exposure_gain
    )

    # --- white balance (F-04, F-05) ---------------------------------------
    raw_r = np.asarray(wb_raw_r, dtype=np.float64)
    raw_b = np.asarray(wb_raw_b, dtype=np.float64)
    # The estimators report "gain needed"; the curve machinery works on the
    # imbalance, which is its negative.
    u_r = -np.log(np.maximum(raw_r, stats.EPS))
    u_b = -np.log(np.maximum(raw_b, stats.EPS))

    if cfg.wb_method == "reference-frame":
        ref_r, ref_b = _reference_gains(path, cfg, trc, info)
        # The grey card fixes the white point; the frame-to-frame ratios still
        # supply any hunting that happened after it was filmed.
        curve_r = wb_mod.illuminant_curve(
            np.full(n, -np.log(max(ref_r, stats.EPS))), np.asarray(delta_r),
            cfg.smoother, cfg.wb_tau, afps, segments,
        )
        curve_b = wb_mod.illuminant_curve(
            np.full(n, -np.log(max(ref_b, stats.EPS))), np.asarray(delta_b),
            cfg.smoother, cfg.wb_tau, afps, segments,
        )
    else:
        curve_r = wb_mod.illuminant_curve(
            u_r, np.asarray(delta_r), cfg.smoother, cfg.wb_tau, afps, segments
        )
        curve_b = wb_mod.illuminant_curve(
            u_b, np.asarray(delta_b), cfg.smoother, cfg.wb_tau, afps, segments
        )

    log_gain_r, clamp_r = wb_mod.wb_log_gain(
        curve_r, cfg.wb_mode, cfg.wb_tau, afps, segments, cfg.max_wb_gain
    )
    log_gain_b, clamp_b = wb_mod.wb_log_gain(
        curve_b, cfg.wb_mode, cfg.wb_tau, afps, segments, cfg.max_wb_gain
    )
    gain_r = np.exp(log_gain_r)
    gain_b = np.exp(log_gain_b)
    wb_clamped = clamp_r | clamp_b

    # Smooth the means that drive the optional red compensation, so that step
    # cannot pump frame to frame.
    if cfg.red_compensation > 0:
        mean_r_out = smooth.smooth_series(np.asarray(mean_r), cfg.smoother, cfg.wb_tau, afps, segments)
        mean_g_out = smooth.smooth_series(np.asarray(mean_g), cfg.smoother, cfg.wb_tau, afps, segments)
    else:
        mean_r_out = np.asarray(mean_r, dtype=np.float64)
        mean_g_out = np.asarray(mean_g, dtype=np.float64)

    metrics = compute_metrics(
        log_luma_arr, curve, exp_gain, curve_r, curve_b, log_gain_r, log_gain_b,
        cfg, afps, segments,
    )
    log_event(
        "analyze.metrics",
        exposure_reduction=f"{metrics['exposure']['reduction_pct']:.1f}%",
        wb_r_reduction=f"{metrics['wb_r']['reduction_pct']:.1f}%",
        wb_b_reduction=f"{metrics['wb_b']['reduction_pct']:.1f}%",
        exposure_clamped=int(exp_clamped.sum()), wb_clamped=int(wb_clamped.sum()),
    )
    if exp_clamped.any() or wb_clamped.any():
        logger.warning(
            "gain clamp hit on %d exposure / %d WB frames - raise --max-exposure-gain "
            "or --max-wb-gain if the result looks under-corrected",
            int(exp_clamped.sum()), int(wb_clamped.sum()),
        )

    return {
        "uwnorm_version": __version__,
        "schema": SCHEMA_VERSION,
        "source": {
            "path": str(Path(path).resolve()),
            "name": Path(path).name,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "duration": info.duration,
            "nb_frames": info.nb_frames,
            "pix_fmt": info.pix_fmt,
            "codec": info.codec,
            "color_trc": info.color_trc,
            "resolved_trc": trc,
            "is_vfr": info.is_vfr,
            "has_audio": info.has_audio,
            "has_data": info.has_data,
        },
        "config": cfg.to_dict(),
        "analysis": {
            "width": aw, "height": ah,
            "stride": cfg.analysis_stride,
            "frames": n,
            "series_fps": afps,
            "range": [start, end],
            "roi_pixels": int(roi_mask.sum()),
        },
        "cuts": {
            "indices": cut_indices,
            "times": _round([times_arr[i] for i in cut_indices], 3),
            "auto": auto_cuts,
            "manual": manual_cuts,
            "threshold": cfg.cut_threshold,
        },
        "frames": {
            "index": [int(i) for i in idx],
            "pts": [int(p) if p is not None else None for p in ptss],
            "time": _round(times_arr, 5),
            "log_luma": _round(log_luma_arr),
            "log_ratio": _round(log_ratio),
            "exposure_curve": _round(curve),
            "exposure_log_gain": _round(exp_gain),
            "exposure_clamped": [bool(b) for b in exp_clamped],
            "wb_raw_r": _round(raw_r),
            "wb_raw_b": _round(raw_b),
            "wb_curve_r": _round(curve_r),
            "wb_curve_b": _round(curve_b),
            "wb_gain_r": _round(gain_r),
            "wb_gain_b": _round(gain_b),
            "wb_clamped": [bool(b) for b in wb_clamped],
            "mean_r": _round(mean_r_out),
            "mean_g": _round(mean_g_out),
            "cut_score": _round(scores, 4),
            "valid_fraction": _round(valid_frac, 4),
        },
        "metrics": metrics,
    }


def compute_metrics(
    log_luma, curve, exp_gain, curve_r, curve_b, log_gain_r, log_gain_b,
    cfg, fps, segments,
) -> dict:
    """Predicted before/after high-frequency variation (A-1, A-2).

    The prediction is exact rather than a guess: the gains are pure
    multiplications in linear light, so the corrected frame's robust
    log-luminance is ``log_luma + exposure_log_gain``, and the illuminant a
    re-analysis would still see is ``log(raw_gain) - log(applied_gain)``.
    The ``verify`` subcommand measures the same quantities on the real output.
    """

    def pair(before, after, tau):
        b = smooth.highpass_energy(before, tau, fps, segments)
        a = smooth.highpass_energy(after, tau, fps, segments)
        red = 100.0 * (1.0 - a / b) if b > 1e-12 else 0.0
        return {
            "hf_std_before": round(float(b), 6),
            "hf_std_after": round(float(a), 6),
            "reduction_pct": round(float(red), 2),
            "tau": tau,
        }

    return {
        "exposure": pair(log_luma, np.asarray(log_luma) + np.asarray(exp_gain), cfg.exposure_tau),
        "exposure_curve": pair(curve, np.asarray(curve) + np.asarray(exp_gain), cfg.exposure_tau),
        "wb_r": pair(curve_r, np.asarray(curve_r) - np.asarray(log_gain_r), cfg.wb_tau),
        "wb_b": pair(curve_b, np.asarray(curve_b) - np.asarray(log_gain_b), cfg.wb_tau),
        "applied": {
            "exposure_ev_min": round(float(np.min(exp_gain) / exposure.LN2), 4),
            "exposure_ev_max": round(float(np.max(exp_gain) / exposure.LN2), 4),
            "wb_gain_r_range": [
                round(float(np.exp(np.min(log_gain_r))), 4),
                round(float(np.exp(np.max(log_gain_r))), 4),
            ],
            "wb_gain_b_range": [
                round(float(np.exp(np.min(log_gain_b))), 4),
                round(float(np.exp(np.max(log_gain_b))), 4),
            ],
        },
    }


def save_analysis(doc: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, separators=(",", ":"))
    log_event("analysis.saved", path=str(path), kb=round(path.stat().st_size / 1024, 1))


def load_analysis(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: analysis schema {doc.get('schema')} is not supported "
            f"by uwnorm {__version__} (expected {SCHEMA_VERSION})"
        )
    return doc
