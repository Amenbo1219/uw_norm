"""Configuration object, YAML merging and presets (F-09).

Precedence, lowest to highest:

    dataclass defaults  <  --config file / preset  <  explicit CLI flags

The CLI parser is built with ``default=None`` for every option so that "the user
typed it" is distinguishable from "it kept its default", which is what makes the
three-layer merge possible.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PRESET_DIR = Path(__file__).parent / "presets"

EXPOSURE_MODES = ("deflicker", "lock", "off")
WB_MODES = ("drift", "fixed", "off")
WB_METHODS = ("shades-of-gray", "max-rgb", "gray-world", "reference-frame")
SMOOTHERS = ("savgol", "median-gauss")
ENCODERS = ("x264", "x265", "prores", "ffv1", "x264_nvenc", "x265_nvenc")
DENOISERS = ("off", "hqdn3d", "nlmeans", "vaguedenoiser")
ENCODER_PRESETS = (
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
)


@dataclass
class Config:
    # --- colour -----------------------------------------------------------
    input_trc: str = "auto"          # auto -> from container metadata, else srgb

    # --- analysis pass ----------------------------------------------------
    analysis_size: int = 384         # long edge of the decode used for statistics
    analysis_stride: int = 1         # measure every Nth frame (gains interpolated)

    # --- exposure (F-03) --------------------------------------------------
    exposure_mode: str = "deflicker"
    exposure_tau: float = 1.5        # seconds; cut-off between "AE hunting" and "scene"
    max_exposure_gain: float = 1.5   # +/- EV clamp

    # --- white balance (F-04, F-05) ---------------------------------------
    wb_method: str = "shades-of-gray"
    wb_p: float = 5.0                # Minkowski norm order for shades-of-gray
    wb_mode: str = "drift"
    wb_tau: float = 3.0              # seconds
    max_wb_gain: float = 3.0
    red_compensation: float = 0.0    # Ancuti alpha; 0 disables
    reference_frame: str | None = None   # timestamp for wb_method=reference-frame
    reference_roi: str | None = None     # "x,y,w,h" normalised
    smoother: str = "savgol"

    # --- scene cuts (F-06) ------------------------------------------------
    cut_threshold: float = 0.35
    cuts: list[str] = field(default_factory=list)  # manual cut timestamps

    # --- ROI / masks (F-07) ----------------------------------------------
    roi: str | None = None           # "center:0.6" or "x,y,w,h" normalised
    exclude_mask: str | None = None  # PNG (white = exclude) or "x,y,w,h"
    highlight_exclude: float = 0.98  # encoded value above which a pixel is "clipped"
    shadow_exclude: float = 0.01     # ... and below which it is crushed/noise

    # --- highlight handling ----------------------------------------------
    soft_clip_knee: float = 0.8      # linear value where the roll-off starts

    # --- preview / verification (F-08) ------------------------------------
    preview_range: str | None = None
    compare: bool = False
    dry_run: bool = False

    # --- encoding ---------------------------------------------------------
    encoder: str = "x264"
    encoder_preset: str = "medium"   # x264/x265 speed-quality trade-off
    crf: int = 16
    chroma: str = "420"              # 420 | 422 | 444
    bit_depth: int = 8               # 8 | 10
    profile: str | None = None       # "archive" -> ProRes 422 10-bit
    denoise: str = "off"

    # --- performance ------------------------------------------------------
    jobs: int = 0                    # 0 -> os.cpu_count()
    device: str = "auto"             # auto | cpu | cuda
    hwaccel: str = "none"            # none | cuda (NVDEC decode)

    # ---------------------------------------------------------------------
    def merged(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with ``overrides`` (None values ignored) applied."""
        data = dataclasses.asdict(self)
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in data:
                raise ValueError(f"unknown configuration key: {key!r}")
            data[key] = value
        cfg = Config(**data)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    # ---------------------------------------------------------------------
    def validate(self) -> None:
        def _one_of(name, allowed):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(
                    f"--{name.replace('_', '-')} must be one of {', '.join(allowed)} "
                    f"(got {value!r})"
                )

        _one_of("exposure_mode", EXPOSURE_MODES)
        _one_of("wb_mode", WB_MODES)
        _one_of("wb_method", WB_METHODS)
        _one_of("smoother", SMOOTHERS)
        _one_of("encoder", ENCODERS)
        _one_of("denoise", DENOISERS)
        _one_of("device", ("auto", "cpu", "cuda"))
        _one_of("hwaccel", ("none", "cuda"))
        _one_of("chroma", ("420", "422", "444"))
        if self.input_trc not in ("auto", "srgb", "bt709", "gamma22", "gamma24", "linear"):
            raise ValueError(f"--input-trc {self.input_trc!r} is not supported")
        if self.encoder_preset not in ENCODER_PRESETS:
            raise ValueError(
                f"--encoder-preset must be one of {', '.join(ENCODER_PRESETS)}"
            )
        if self.bit_depth not in (8, 10):
            raise ValueError("--bit-depth must be 8 or 10")
        if self.analysis_size < 32:
            raise ValueError("--analysis-size must be >= 32")
        if self.analysis_stride < 1:
            raise ValueError("--analysis-stride must be >= 1")
        if self.exposure_tau <= 0 or self.wb_tau <= 0:
            raise ValueError("time constants must be > 0")
        if self.max_wb_gain < 1.0:
            raise ValueError("--max-wb-gain must be >= 1.0")
        if self.max_exposure_gain < 0:
            raise ValueError("--max-exposure-gain must be >= 0")
        if not 0.0 < self.soft_clip_knee < 1.0:
            raise ValueError("--soft-clip-knee must be in (0, 1)")
        if self.wb_method == "reference-frame" and not self.reference_frame:
            raise ValueError("--wb-method reference-frame requires --reference-frame")
        if self.profile == "archive":
            pass  # resolved in resolve_profile()

    def resolve_profile(self) -> "Config":
        """Expand ``--profile`` shorthands into concrete encoder settings."""
        if self.profile == "archive":
            cfg = dataclasses.replace(self, encoder="prores", chroma="422", bit_depth=10)
            cfg.validate()
            return cfg
        return self


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    out = {k.replace("-", "_"): v for k, v in data.items()}
    # ``description`` documents the preset for ``uwnorm presets``; it is not a
    # setting, so it never reaches Config.
    out.pop("description", None)
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load ``--config``: either a bundled preset name or a YAML path."""
    if path is None:
        return {}
    preset = PRESET_DIR / f"{path}.yaml"
    if preset.exists():
        return load_yaml(preset)
    return load_yaml(path)


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


_TIME_RE = re.compile(
    r"^(?:(?P<h>\d+):)?(?:(?P<m>\d+):)?(?P<s>\d+(?:\.\d+)?)$"
)


def parse_time(value: str | float) -> float:
    """``'1:02.5'`` / ``'00:12.3'`` / ``'12.3'`` -> seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    m = _TIME_RE.match(str(value).strip())
    if not m:
        raise ValueError(f"cannot parse timestamp: {value!r}")
    h = float(m.group("h") or 0)
    mi = float(m.group("m") or 0)
    s = float(m.group("s"))
    if m.group("h") is not None and m.group("m") is None:
        # "12:34" parsed as h=12 -> actually mm:ss
        h, mi = 0.0, float(m.group("h"))
    return h * 3600 + mi * 60 + s


def parse_range(value: str) -> tuple[float, float]:
    """``'00:30-00:45'`` -> ``(30.0, 45.0)``."""
    parts = value.split("-")
    if len(parts) != 2:
        raise ValueError(f"--preview-range must be START-END (got {value!r})")
    start, end = parse_time(parts[0]), parse_time(parts[1])
    if end <= start:
        raise ValueError("--preview-range end must be after start")
    return start, end
