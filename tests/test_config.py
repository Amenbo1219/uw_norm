"""Configuration merging, presets and validation (F-09)."""

import pytest

from uwnorm.cli import build_parser, config_from_args
from uwnorm.config import Config, list_presets, load_config, parse_range, parse_time


def test_defaults_validate():
    Config().validate()


@pytest.mark.parametrize("text,seconds", [
    ("12.3", 12.3), ("00:12.3", 12.3), ("1:30", 90.0),
    ("01:02:03", 3723.0), ("0:00", 0.0),
])
def test_parse_time(text, seconds):
    assert parse_time(text) == pytest.approx(seconds)


def test_parse_time_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_time("half past two")


def test_parse_range():
    assert parse_range("00:30-00:45") == (30.0, 45.0)
    with pytest.raises(ValueError):
        parse_range("00:45-00:30")
    with pytest.raises(ValueError):
        parse_range("00:30")


@pytest.mark.parametrize("field,value", [
    ("exposure_mode", "sometimes"), ("wb_mode", "maybe"), ("wb_method", "vibes"),
    ("encoder", "divx"), ("denoise", "hope"), ("chroma", "411"), ("bit_depth", 12),
    ("input_trc", "pq"), ("max_wb_gain", 0.5), ("exposure_tau", 0.0),
    ("soft_clip_knee", 1.0), ("analysis_stride", 0),
])
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValueError):
        Config().merged({field: value})


def test_reference_frame_requires_a_timestamp():
    with pytest.raises(ValueError, match="reference-frame"):
        Config().merged({"wb_method": "reference-frame"})
    Config().merged({"wb_method": "reference-frame", "reference_frame": "00:05"})


def test_merged_ignores_none():
    base = Config(crf=20)
    assert base.merged({"crf": None}).crf == 20
    assert base.merged({"crf": 12}).crf == 12


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown configuration key"):
        Config().merged({"saturation": 2.0})


def test_archive_profile_expands():
    resolved = Config(profile="archive").resolve_profile()
    assert (resolved.encoder, resolved.chroma, resolved.bit_depth) == ("prores", "422", 10)


@pytest.mark.parametrize("name", list_presets())
def test_every_preset_loads_and_validates(name):
    merged = Config().merged(load_config(name))
    merged.validate()


def test_presets_cover_the_documented_set():
    assert {"shallow_reef", "deep_blue", "cave_with_lights"} <= set(list_presets())


def test_yaml_accepts_dashes_and_drops_description(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("description: hi\nexposure-mode: lock\nwb-tau: 4.5\n")
    data = load_config(path)
    assert data == {"exposure_mode": "lock", "wb_tau": 4.5}


def test_cli_flags_override_the_preset():
    args = build_parser().parse_args(
        ["run", "in.mp4", "-o", "out.mp4", "--config", "deep_blue", "--wb-tau", "9.5"]
    )
    cfg = config_from_args(args)
    assert cfg.wb_tau == 9.5                 # flag wins
    assert cfg.red_compensation == 1.0       # preset survives where no flag was given
    assert cfg.exposure_mode == "deflicker"  # dataclass default survives both


def test_cli_cuts_are_split():
    args = build_parser().parse_args(
        ["analyze", "in.mp4", "--cuts", "00:12.3,01:45.0"]
    )
    assert config_from_args(args).cuts == ["00:12.3", "01:45.0"]


def test_jobs_defaults_to_cpu_count():
    args = build_parser().parse_args(["analyze", "in.mp4"])
    assert config_from_args(args).jobs >= 1
