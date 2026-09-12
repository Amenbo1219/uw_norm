"""Synthetic underwater clips with a *known* ground truth.

Real footage cannot be used as a test fixture: nobody knows what its true
exposure curve was.  These generators build a clip from a known exposure curve
and a known illuminant, so the tests can assert that the tool recovers them.

The scene deliberately contains the things that break naive approaches:
a moving subject (so a plain mean would track the subject, not the exposure),
blown highlights (so clipped pixels must be excluded), and a hard cut.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 1e-8), 1 / 2.4) - 0.055)


def make_reef(width: int, height: int, seed: int) -> np.ndarray:
    """A static, textured linear-light scene in [0, 1]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = 0.25 + 0.18 * np.sin(xx / 17.0) * np.cos(yy / 13.0)
    blobs = np.zeros((height, width), np.float32)
    for _ in range(24):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        r = rng.uniform(8, max(9.0, width / 8))
        amp = rng.uniform(-0.12, 0.18)
        blobs += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
    grain = rng.normal(0.0, 0.012, (height, width)).astype(np.float32)
    scene = np.clip(base + blobs + grain, 0.02, 1.0)
    # Albedo varies per channel so the scene is not trivially grey.
    tint = np.stack([
        scene * (0.9 + 0.2 * np.sin(xx / 29.0)),
        scene,
        scene * (0.95 + 0.1 * np.cos(yy / 31.0)),
    ], axis=-1)
    return np.clip(tint, 0.01, 1.0).astype(np.float32)


@dataclass
class SynthSpec:
    width: int = 320
    height: int = 180
    fps: float = 30.0
    n_frames: int = 180
    seed: int = 7
    illuminant: tuple[float, float, float] = (0.40, 1.0, 0.85)
    #: amplitude (in stops) of the fast AE hunting to be removed
    flicker_ev: float = 0.35
    flicker_hz: float = 4.0
    #: amplitude (in stops) of the slow, genuine lighting change to be kept
    drift_ev: float = 0.5
    drift_hz: float = 0.08
    jitter_ev: float = 0.05
    #: amplitude (in log units) of auto-white-balance hunting on R and B
    wb_flicker: float = 0.0
    wb_flicker_hz: float = 3.0
    #: slow illuminant change, e.g. descending into deeper water
    wb_drift: float = 0.0
    cut_at: int | None = None
    moving_subject: bool = True
    #: camera pan speed in pixels per frame (false-positive stress for cut detection)
    pan_px_per_frame: float = 0.0
    highlight: bool = True
    audio: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps


def illuminant_curve(spec: SynthSpec) -> np.ndarray:
    """Ground-truth per-frame illuminant, shape (n, 3), G fixed at 1."""
    rng = np.random.default_rng(spec.seed + 2000)
    t = np.arange(spec.n_frames) / spec.fps
    base = np.asarray(spec.illuminant, dtype=np.float64)
    hunt_r = spec.wb_flicker * np.sin(2 * np.pi * spec.wb_flicker_hz * t)
    hunt_b = spec.wb_flicker * np.sin(2 * np.pi * spec.wb_flicker_hz * t + 1.1)
    if spec.wb_flicker:
        hunt_r = hunt_r + rng.normal(0, spec.wb_flicker / 3, spec.n_frames)
        hunt_b = hunt_b + rng.normal(0, spec.wb_flicker / 3, spec.n_frames)
    drift = spec.wb_drift * np.linspace(0.0, 1.0, spec.n_frames)
    out = np.empty((spec.n_frames, 3))
    out[:, 0] = base[0] * np.exp(hunt_r - drift)
    out[:, 1] = base[1]
    out[:, 2] = base[2] * np.exp(hunt_b + 0.3 * drift)
    return out


def exposure_curve(spec: SynthSpec) -> np.ndarray:
    """Ground-truth log2 exposure (stops) per frame."""
    rng = np.random.default_rng(spec.seed + 1000)
    t = np.arange(spec.n_frames) / spec.fps
    flicker = spec.flicker_ev * np.sin(2 * np.pi * spec.flicker_hz * t)
    drift = spec.drift_ev * np.sin(2 * np.pi * spec.drift_hz * t)
    jitter = rng.normal(0.0, spec.jitter_ev, spec.n_frames)
    return flicker + drift + jitter


def render_frames(spec: SynthSpec) -> tuple[np.ndarray, np.ndarray]:
    """Render the clip -> (frames uint8 RGB, ground-truth exposure in stops)."""
    scene_a = make_reef(spec.width, spec.height, spec.seed)
    scene_b = make_reef(spec.width, spec.height, spec.seed + 99) * 0.7 if spec.cut_at else None
    ev = exposure_curve(spec)
    illum_series = illuminant_curve(spec).astype(np.float32)
    out = np.empty((spec.n_frames, spec.height, spec.width, 3), dtype=np.uint8)

    for i in range(spec.n_frames):
        scene = scene_b if (spec.cut_at and i >= spec.cut_at) else scene_a
        frame = scene.copy()
        if spec.pan_px_per_frame:
            shift = int(round(i * spec.pan_px_per_frame)) % spec.width
            frame = np.roll(frame, -shift, axis=1)
        if spec.moving_subject:
            # A fish crossing the frame: ~15% of the pixels, moving every frame.
            w = max(8, spec.width // 7)
            x0 = int((i / max(spec.n_frames - 1, 1)) * (spec.width - w))
            y0 = spec.height // 3
            frame[y0 : y0 + spec.height // 4, x0 : x0 + w] *= 0.25
        if spec.highlight:
            # A dive light hotspot that clips - must be excluded from statistics.
            frame[5 : 5 + spec.height // 12, 5 : 5 + spec.width // 12] = 1.6
        lit = frame * illum_series[i] * (2.0 ** ev[i])
        out[i] = np.clip(srgb_oetf(lit) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out, ev


def write_clip(path: str | Path, spec: SynthSpec, crf: int = 12) -> np.ndarray:
    """Render and encode to an MP4 (returns the ground-truth exposure curve)."""
    frames, ev = render_frames(spec)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{spec.width}x{spec.height}", "-r", str(spec.fps), "-i", "pipe:0",
    ]
    if spec.audio:
        cmd += ["-f", "lavfi", "-t", str(spec.duration), "-i",
                "sine=frequency=440:sample_rate=48000", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-threads", "1", str(path)]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode()[-2000:])
    return ev
