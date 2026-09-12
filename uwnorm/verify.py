"""Acceptance verification (A-1, A-2, A-5, A-6).

``analysis.json`` carries a *predicted* improvement, derived from the gain
curves.  That prediction is sound - the gains are exact multiplications - but it
cannot see what the encoder did afterwards.  This module measures the same
quantities on the file that was actually written, by running the analysis pass
over the output and comparing curve to curve.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np

from .config import Config
from .core import smooth
from .io.probe import probe
from .logging_utils import log_event


def _hf(series, tau, fps) -> float:
    return smooth.highpass_energy(np.asarray(series, dtype=np.float64), tau, fps)


def verify_output(source_doc: dict, output_path: str, cfg: Config) -> dict:
    """Re-analyse the written file and report the measured reduction."""
    from .analyze import analyze

    # Measuring the output with the correction switched off is the point: we
    # want the illuminant and exposure the *output* still exhibits, not another
    # round of corrections applied on top.
    probe_cfg = cfg.merged({"exposure_mode": "off", "wb_mode": "off"})
    after = analyze(output_path, probe_cfg, progress=False)

    fps_before = source_doc["analysis"]["series_fps"]
    fps_after = after["analysis"]["series_fps"]
    b, a = source_doc["frames"], after["frames"]

    def compare(key, tau):
        hb = _hf(b[key], tau, fps_before)
        ha = _hf(a[key], tau, fps_after)
        return {
            "hf_std_before": round(hb, 6),
            "hf_std_after": round(ha, 6),
            "reduction_pct": round(100.0 * (1.0 - ha / hb), 2) if hb > 1e-12 else 0.0,
        }

    result = {
        "exposure": compare("exposure_curve", cfg.exposure_tau),
        "wb_r": compare("wb_curve_r", cfg.wb_tau),
        "wb_b": compare("wb_curve_b", cfg.wb_tau),
        "frames": {"before": source_doc["analysis"]["frames"], "after": after["analysis"]["frames"]},
        "streams": compare_streams(source_doc["source"]["path"], output_path),
    }
    log_event(
        "verify.done",
        exposure=f"{result['exposure']['reduction_pct']}%",
        wb_r=f"{result['wb_r']['reduction_pct']}%",
        wb_b=f"{result['wb_b']['reduction_pct']}%",
        streams_ok=result["streams"]["ok"],
    )
    return result


def compare_streams(source: str, output: str) -> dict:
    """Check that audio / data / timing survived the round trip (A-5)."""
    a, b = probe(source), probe(output)
    checks = {
        "audio_preserved": (not a.has_audio) or b.has_audio,
        "data_preserved": (not a.has_data) or b.has_data,
        "resolution_same": (a.width, a.height) == (b.width, b.height),
        "frame_count_same": abs(a.frame_count_estimate - b.frame_count_estimate) <= 1,
        "duration_close": (
            a.duration is None or b.duration is None or abs(a.duration - b.duration) < 0.25
        ),
    }
    return {"ok": all(checks.values()), **checks}


def video_hash(path: str) -> str:
    """SHA-256 of the *video bitstream only* (A-6).

    Container metadata contains a creation timestamp, so hashing the whole file
    would report a difference on every run no matter how deterministic the
    encoder is.  Hashing the elementary stream measures what the question is
    actually about.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path,
        "-map", "0:v:0", "-c", "copy", "-f", "data", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return hashlib.sha256(proc.stdout).hexdigest()
