"""Container inspection via ffprobe (3.1, A-5)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path


@dataclass
class MediaInfo:
    path: str
    width: int
    height: int
    fps: float
    time_base: Fraction
    nb_frames: int | None
    duration: float | None
    pix_fmt: str | None
    color_trc: str | None
    color_primaries: str | None
    color_space: str | None
    codec: str | None
    is_vfr: bool
    has_audio: bool
    has_data: bool
    stream_kinds: list[str] = field(default_factory=list)

    @property
    def frame_count_estimate(self) -> int:
        if self.nb_frames:
            return int(self.nb_frames)
        if self.duration and self.fps:
            return int(round(self.duration * self.fps))
        return 0


def _ffprobe_bin() -> str:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe not found on PATH (install ffmpeg)")
    return exe


def probe(path: str | Path) -> MediaInfo:
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"no such file: {path}")
    cmd = [
        _ffprobe_bin(), "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        raise ValueError(
            f"{path}: ffprobe could not read this file"
            + (f" ({detail[-1]})" if detail else "")
        )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{path}: no video stream")

    def _frac(value, default=Fraction(30, 1)):
        try:
            f = Fraction(value)
            return f if f > 0 else default
        except Exception:
            return default

    avg = _frac(video.get("avg_frame_rate"))
    real = _frac(video.get("r_frame_rate"), avg)
    # A large gap between the nominal and the average frame rate is the usual
    # fingerprint of variable frame rate material (GoPro in low light).
    is_vfr = abs(float(real) - float(avg)) > 0.01 * max(float(avg), 1.0)

    duration = video.get("duration") or data.get("format", {}).get("duration")
    nb_frames = video.get("nb_frames")

    return MediaInfo(
        path=path,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=float(avg),
        time_base=_frac(video.get("time_base"), Fraction(1, 90000)),
        nb_frames=int(nb_frames) if nb_frames and str(nb_frames).isdigit() else None,
        duration=float(duration) if duration else None,
        pix_fmt=video.get("pix_fmt"),
        color_trc=video.get("color_transfer"),
        color_primaries=video.get("color_primaries"),
        color_space=video.get("color_space"),
        codec=video.get("codec_name"),
        is_vfr=is_vfr,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        has_data=any(s.get("codec_type") in ("data", "subtitle") for s in streams),
        stream_kinds=[s.get("codec_type", "?") for s in streams],
    )


def resolve_trc(info: MediaInfo, configured: str) -> str:
    """Pick the transfer characteristic to linearise with (F-02, 3.1).

    Container metadata wins when it says something meaningful; otherwise fall
    back to the ``--input-trc`` setting, defaulting to sRGB which is what
    consumer action cameras actually write to an 8-bit BT.709 file.
    """
    from ..core.color import FFMPEG_TRC_MAP

    if configured != "auto":
        return configured
    mapped = FFMPEG_TRC_MAP.get((info.color_trc or "").lower())
    return mapped or "srgb"
