"""Subcommand behaviour (section 8)."""

import json
import shutil

import pytest

from uwnorm.cli import main
from uwnorm.io.probe import probe


def test_analyze_writes_json_and_report(flicker_clip, tmp_path, capsys):
    analysis = tmp_path / "a.json"
    report = tmp_path / "r.html"
    rc = main(["analyze", flicker_clip["path"], "-a", str(analysis),
               "--report", str(report), "--no-progress"])
    assert rc == 0
    doc = json.loads(analysis.read_text())
    assert doc["analysis"]["frames"] > 0
    assert report.exists() and report.stat().st_size > 10_000
    assert "reduction" in capsys.readouterr().out


def test_analyze_default_json_path(flicker_clip, tmp_path):
    local = tmp_path / "clip.mp4"
    shutil.copy(flicker_clip["path"], local)
    assert main(["analyze", str(local), "--no-progress"]) == 0
    assert (tmp_path / "clip.analysis.json").exists()


def test_apply_consumes_an_existing_analysis(flicker_clip, tmp_path):
    analysis = tmp_path / "a.json"
    out = tmp_path / "o.mp4"
    assert main(["analyze", flicker_clip["path"], "-a", str(analysis), "--no-progress"]) == 0
    assert main(["apply", flicker_clip["path"], "-o", str(out),
                 "-a", str(analysis), "--no-progress"]) == 0
    assert probe(str(out)).frame_count_estimate > 0


def test_run_does_both(flicker_clip, tmp_path):
    out = tmp_path / "o.mp4"
    rc = main(["run", flicker_clip["path"], "-o", str(out),
               "-a", str(tmp_path / "a.json"), "--no-progress"])
    assert rc == 0 and out.exists()


def test_run_verify_reports_pass(flicker_clip, tmp_path, capsys):
    rc = main(["run", flicker_clip["path"], "-o", str(tmp_path / "o.mp4"),
               "-a", str(tmp_path / "a.json"), "--verify", "--no-progress"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "FAIL" not in out, out
    assert out.count("PASS") >= 4


def test_dry_run_writes_no_video(flicker_clip, tmp_path):
    report = tmp_path / "r.html"
    rc = main(["run", flicker_clip["path"], "--dry-run", "--report", str(report),
               "-a", str(tmp_path / "a.json"), "--no-progress"])
    assert rc == 0
    assert report.exists()
    assert not list(tmp_path.glob("*.mp4"))


def test_report_subcommand_regenerates(flicker_clip, tmp_path):
    analysis = tmp_path / "a.json"
    main(["analyze", flicker_clip["path"], "-a", str(analysis), "--no-progress"])
    png = tmp_path / "r.png"
    assert main(["report", str(analysis), "-o", str(png)]) == 0
    assert png.stat().st_size > 10_000


def test_verify_subcommand_exit_code(flicker_clip, tmp_path):
    analysis = tmp_path / "a.json"
    out = tmp_path / "o.mp4"
    main(["run", flicker_clip["path"], "-o", str(out), "-a", str(analysis), "--no-progress"])
    assert main(["verify", flicker_clip["path"], str(out), "-a", str(analysis)]) == 0


def test_compare_renders_side_by_side(flicker_clip, tmp_path):
    out = tmp_path / "o.mp4"
    rc = main(["run", flicker_clip["path"], "-o", str(out), "-a", str(tmp_path / "a.json"),
               "--compare", "--no-progress"])
    assert rc == 0
    compare = tmp_path / "o.compare.mp4"
    assert compare.exists()
    source = probe(flicker_clip["path"])
    assert probe(str(compare)).width == source.width * 2


def test_preset_is_applied(flicker_clip, tmp_path):
    analysis = tmp_path / "a.json"
    rc = main(["analyze", flicker_clip["path"], "-a", str(analysis),
               "--config", "deep_blue", "--no-progress"])
    assert rc == 0
    doc = json.loads(analysis.read_text())
    assert doc["config"]["red_compensation"] == 1.0
    assert doc["config"]["wb_p"] == 6.0


def test_batch_mode_processes_a_directory(flicker_clip, quiet_clip, tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    shutil.copy(flicker_clip["path"], in_dir / "one.mp4")
    shutil.copy(quiet_clip["path"], in_dir / "two.mp4")
    rc = main(["run", str(in_dir), "-o", str(out_dir), "--parallel", "2", "--no-progress"])
    assert rc == 0
    assert (out_dir / "one.mp4").exists() and (out_dir / "two.mp4").exists()


def test_cache_dir_skips_the_second_analysis(flicker_clip, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    args = ["run", flicker_clip["path"], "-o", str(tmp_path / "o.mp4"),
            "--cache-dir", str(cache), "--no-progress"]
    assert main(args) == 0
    cached = next(cache.glob("*.analysis.json"))
    stamp = cached.stat().st_mtime_ns
    assert main(args) == 0
    assert cached.stat().st_mtime_ns == stamp   # N-04: reused, not recomputed


def test_jsonl_log_is_machine_readable(flicker_clip, tmp_path):
    log = tmp_path / "log.jsonl"
    main(["analyze", flicker_clip["path"], "-a", str(tmp_path / "a.json"),
          "--log-jsonl", str(log), "--no-progress"])
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert {"analyze.start", "analyze.metrics"} <= {e.get("event") for e in events}


def test_presets_subcommand_lists_all(capsys):
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    assert "deep_blue" in out and "cave_with_lights" in out


def test_missing_file_exits_nonzero(tmp_path):
    assert main(["analyze", str(tmp_path / "nope.mp4"), "--no-progress"]) != 0


def test_bad_option_is_rejected(flicker_clip):
    with pytest.raises(SystemExit):
        main(["analyze", flicker_clip["path"], "--exposure-mode", "whenever"])


def test_apply_inherits_output_settings_from_the_analysis(flicker_clip, tmp_path):
    """`analyze --config archive` then a bare `apply` must still write ProRes."""
    analysis = tmp_path / "a.json"
    main(["analyze", flicker_clip["path"], "-a", str(analysis),
          "--config", "archive", "--no-progress"])
    out = tmp_path / "o.mov"
    assert main(["apply", flicker_clip["path"], "-o", str(out),
                 "-a", str(analysis), "--no-progress"]) == 0
    assert probe(str(out)).codec == "prores"


def test_apply_flag_overrides_the_stored_setting(flicker_clip, tmp_path):
    analysis = tmp_path / "a.json"
    main(["analyze", flicker_clip["path"], "-a", str(analysis),
          "--config", "archive", "--no-progress"])
    out = tmp_path / "o.mp4"
    assert main(["apply", flicker_clip["path"], "-o", str(out), "-a", str(analysis),
                 "--encoder", "x264", "--bit-depth", "8", "--chroma", "420",
                 "--no-progress"]) == 0
    assert probe(str(out)).codec == "h264"


def test_encoder_preset_reaches_the_encoder(flicker_clip, tmp_path):
    out = tmp_path / "o.mp4"
    rc = main(["run", flicker_clip["path"], "-o", str(out), "-a", str(tmp_path / "a.json"),
               "--encoder-preset", "veryfast", "--no-progress"])
    assert rc == 0 and probe(str(out)).frame_count_estimate > 0
