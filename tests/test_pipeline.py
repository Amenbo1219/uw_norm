"""End-to-end tests, organised around the acceptance criteria in section 9.

These run the real decoder, the real gain maths and the real encoder, then
measure the result the same way the spec does.
"""

import json

import numpy as np
import pytest

from uwnorm.analyze import analyze, load_analysis, save_analysis
from uwnorm.apply import apply_gains, build_gain_table, check_container
from uwnorm.config import Config
from uwnorm.core import exposure, ops, smooth
from uwnorm.io.probe import probe
from uwnorm.verify import compare_streams, verify_output, video_hash

TARGET = 80.0   # the "80% or better" bar from A-1 / A-2


@pytest.fixture(scope="module")
def analysed(flicker_clip):
    cfg = Config()
    return cfg, analyze(flicker_clip["path"], cfg, progress=False)


# --- A-1 ------------------------------------------------------------------

def test_a1_exposure_high_frequency_reduced(analysed):
    _, doc = analysed
    assert doc["metrics"]["exposure"]["reduction_pct"] >= TARGET


def test_a1_measured_on_the_written_file(analysed, flicker_clip, tmp_path):
    cfg, doc = analysed
    out = str(tmp_path / "out.mp4")
    apply_gains(flicker_clip["path"], out, doc, cfg, progress=False)
    result = verify_output(doc, out, cfg)
    assert result["exposure"]["reduction_pct"] >= TARGET


def test_exposure_curve_tracks_the_ground_truth(analysed, flicker_clip):
    """The recovered curve must match the exposure the clip was rendered with."""
    _, doc = analysed
    recovered = np.asarray(doc["frames"]["exposure_curve"]) / exposure.LN2
    truth = flicker_clip["ev"][: len(recovered)]
    recovered -= recovered.mean()
    truth = truth - truth.mean()
    assert np.corrcoef(recovered, truth)[0, 1] > 0.98
    assert np.std(recovered - truth) < 0.1   # stops


# --- A-2 ------------------------------------------------------------------

def test_a2_wb_high_frequency_reduced(analysed):
    _, doc = analysed
    assert doc["metrics"]["wb_r"]["reduction_pct"] >= TARGET
    assert doc["metrics"]["wb_b"]["reduction_pct"] >= TARGET


def test_a2_measured_on_the_written_file(analysed, flicker_clip, tmp_path):
    cfg, doc = analysed
    out = str(tmp_path / "wb.mp4")
    apply_gains(flicker_clip["path"], out, doc, cfg, progress=False)
    result = verify_output(doc, out, cfg)
    assert result["wb_r"]["reduction_pct"] >= TARGET
    assert result["wb_b"]["reduction_pct"] >= TARGET


# --- A-3: the correction moves neutrals towards neutral -------------------

def test_a3_output_is_closer_to_neutral(analysed, flicker_clip, tmp_path):
    """A grey-card stand-in: the water's colour cast must actually shrink."""
    cfg, doc = analysed
    out = str(tmp_path / "neutral.mp4")
    apply_gains(flicker_clip["path"], out, doc, cfg, progress=False)
    probe_cfg = cfg.merged({"exposure_mode": "off", "wb_mode": "off"})
    after = analyze(out, probe_cfg, progress=False)

    def cast(document):
        r = np.asarray(document["frames"]["wb_raw_r"])
        b = np.asarray(document["frames"]["wb_raw_b"])
        # how far the required gains sit from 1.0 -> how far the frame is from neutral
        return float(np.mean(np.abs(np.log(r))) + np.mean(np.abs(np.log(b))))

    assert cast(after) < 0.25 * cast(doc)


# --- A-4: cuts ------------------------------------------------------------

def test_a4_cut_is_found_and_not_smoothed_across(cut_clip):
    cfg = Config()
    doc = analyze(cut_clip["path"], cfg, progress=False)
    cut_index = cut_clip["spec"].cut_at
    assert any(abs(c - cut_index) <= 1 for c in doc["cuts"]["indices"]), doc["cuts"]
    # The exposure integrator restarts, so the curve is free to jump here.
    gain = np.asarray(doc["frames"]["exposure_log_gain"])
    segs = smooth.segments_from_cuts(len(gain), doc["cuts"]["indices"])
    assert len(segs) >= 2


def test_a4_no_false_cuts_on_a_fast_pan(pan_clip):
    doc = analyze(pan_clip["path"], Config(), progress=False)
    assert doc["cuts"]["indices"] == []


def test_a4_no_false_cuts_from_exposure_flicker(flicker_clip):
    doc = analyze(flicker_clip["path"], Config(), progress=False)
    assert doc["cuts"]["indices"] == []


def test_manual_cuts_are_honoured(flicker_clip):
    cfg = Config(cuts=["2.0"])
    doc = analyze(flicker_clip["path"], cfg, progress=False)
    assert 60 in doc["cuts"]["indices"]
    assert doc["cuts"]["manual"] == [60]


# --- A-5: streams and timing ---------------------------------------------

def test_a5_audio_and_timing_survive(analysed, flicker_clip, tmp_path):
    cfg, doc = analysed
    out = str(tmp_path / "streams.mp4")
    apply_gains(flicker_clip["path"], out, doc, cfg, progress=False)
    checks = compare_streams(flicker_clip["path"], out)
    assert checks["ok"], checks


def test_a5_frame_count_and_resolution_match(analysed, flicker_clip, tmp_path):
    cfg, doc = analysed
    out = str(tmp_path / "count.mp4")
    result = apply_gains(flicker_clip["path"], out, doc, cfg, progress=False)
    source, written = probe(flicker_clip["path"]), probe(out)
    assert result["frames"] == source.frame_count_estimate
    assert (written.width, written.height) == (source.width, source.height)


# --- A-6: reproducibility -------------------------------------------------

def test_a6_two_runs_are_bit_identical(analysed, flicker_clip, tmp_path):
    cfg, doc = analysed
    a, b = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")
    apply_gains(flicker_clip["path"], a, doc, cfg, progress=False)
    apply_gains(flicker_clip["path"], b, doc, cfg, progress=False)
    assert video_hash(a) == video_hash(b)


def test_a6_analysis_is_deterministic(flicker_clip):
    cfg = Config()
    first = analyze(flicker_clip["path"], cfg, progress=False)
    second = analyze(flicker_clip["path"], cfg, progress=False)
    assert first["frames"] == second["frames"]
    assert first["metrics"] == second["metrics"]


@pytest.mark.skipif(not ops.cuda_available(), reason="CUDA not available")
def test_cpu_and_gpu_agree(analysed, flicker_clip, tmp_path):
    """The GPU path is an optimisation, not a different algorithm."""
    cfg, doc = analysed
    cpu = str(tmp_path / "cpu.mp4")
    gpu = str(tmp_path / "gpu.mp4")
    apply_gains(flicker_clip["path"], cpu, doc, cfg.merged({"device": "cpu"}), progress=False)
    apply_gains(flicker_clip["path"], gpu, doc, cfg.merged({"device": "cuda"}), progress=False)
    assert video_hash(cpu) == video_hash(gpu)


# --- behaviour on clean footage ------------------------------------------

def test_a_clean_clip_is_left_alone(quiet_clip):
    """No flicker in, almost no gain out - the tool must not invent motion."""
    doc = analyze(quiet_clip["path"], Config(), progress=False)
    gain_ev = np.abs(np.asarray(doc["frames"]["exposure_log_gain"]) / exposure.LN2)
    assert gain_ev.max() < 0.08


def test_modes_off_produce_a_transparent_pass(quiet_clip, tmp_path):
    cfg = Config(exposure_mode="off", wb_mode="off")
    doc = analyze(quiet_clip["path"], cfg, progress=False)
    table = build_gain_table(doc)["gains"]
    assert table == pytest.approx(np.ones_like(table))


# --- F-10: hand-edited gain curves ---------------------------------------

def test_edited_gain_curve_is_used_verbatim(analysed, flicker_clip, tmp_path):
    cfg, doc = analysed
    path = tmp_path / "edited.json"
    edited = json.loads(json.dumps(doc))
    n = len(edited["frames"]["exposure_log_gain"])
    edited["frames"]["exposure_log_gain"] = [0.0] * n
    edited["frames"]["wb_gain_r"] = [2.0] * n
    edited["frames"]["wb_gain_b"] = [1.0] * n
    save_analysis(edited, path)
    table = build_gain_table(load_analysis(path))["gains"]
    assert table[:, 0] == pytest.approx(np.full(table.shape[0], 2.0))
    assert table[:, 1] == pytest.approx(np.ones(table.shape[0]))


def test_analysis_json_roundtrips(analysed, tmp_path):
    _, doc = analysed
    path = tmp_path / "a.json"
    save_analysis(doc, path)
    assert load_analysis(path) == doc


def test_incompatible_schema_is_rejected(analysed, tmp_path):
    _, doc = analysed
    path = tmp_path / "old.json"
    stale = json.loads(json.dumps(doc))
    stale["schema"] = 0
    path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="schema"):
        load_analysis(path)


# --- F-08 / encoders ------------------------------------------------------

def test_preview_range_processes_only_the_window(flicker_clip, tmp_path):
    cfg = Config(preview_range="1.0-3.0")
    doc = analyze(flicker_clip["path"], cfg, progress=False)
    times = doc["frames"]["time"]
    assert 0.99 <= times[0] and times[-1] <= 3.01
    out = str(tmp_path / "preview.mp4")
    apply_gains(flicker_clip["path"], out, doc, cfg, progress=False)
    assert probe(out).frame_count_estimate == pytest.approx(len(times), abs=2)


@pytest.mark.parametrize("encoder,suffix,bits,chroma", [
    ("x264", ".mp4", 8, "420"),
    ("x264", ".mp4", 10, "422"),
    ("x265", ".mp4", 8, "420"),
    ("prores", ".mov", 10, "422"),
    ("ffv1", ".mkv", 8, "444"),
])
def test_encoders_produce_a_playable_file(analysed, flicker_clip, tmp_path,
                                          encoder, suffix, bits, chroma):
    cfg, doc = analysed
    out = str(tmp_path / f"enc_{encoder}_{bits}{suffix}")
    use = cfg.merged({"encoder": encoder, "bit_depth": bits, "chroma": chroma})
    apply_gains(flicker_clip["path"], out, doc, use, progress=False)
    written = probe(out)
    assert written.frame_count_estimate == pytest.approx(doc["analysis"]["frames"], abs=1)


def test_ffv1_in_mp4_is_refused_with_advice():
    with pytest.raises(ValueError, match=r"\.mkv"):
        check_container("out.mp4", "ffv1")


def test_denoise_runs(analysed, flicker_clip, tmp_path):
    cfg, doc = analysed
    out = str(tmp_path / "dn.mp4")
    apply_gains(flicker_clip["path"], out, doc, cfg.merged({"denoise": "hqdn3d"}), progress=False)
    assert probe(out).frame_count_estimate == pytest.approx(doc["analysis"]["frames"], abs=1)


def test_analysis_stride_matches_full_rate_analysis(flicker_clip):
    """Skipping frames must change the cost, not the answer."""
    full = analyze(flicker_clip["path"], Config(), progress=False)
    strided = analyze(flicker_clip["path"], Config(analysis_stride=3), progress=False)
    assert strided["analysis"]["frames"] == pytest.approx(full["analysis"]["frames"] / 3, abs=2)
    assert strided["metrics"]["exposure"]["reduction_pct"] > 70


def test_roi_and_exclusion_change_the_estimate(flicker_clip):
    plain = analyze(flicker_clip["path"], Config(), progress=False)
    masked = analyze(flicker_clip["path"], Config(roi="center:0.5"), progress=False)
    assert plain["analysis"]["roi_pixels"] > masked["analysis"]["roi_pixels"]
    assert masked["metrics"]["exposure"]["reduction_pct"] >= TARGET


def test_red_compensation_lifts_red(analysed, flicker_clip, tmp_path):
    cfg, doc = analysed
    plain = str(tmp_path / "plain.mp4")
    lifted = str(tmp_path / "lifted.mp4")
    apply_gains(flicker_clip["path"], plain, doc, cfg, progress=False)
    red_cfg = Config(red_compensation=1.0)
    red_doc = analyze(flicker_clip["path"], red_cfg, progress=False)
    apply_gains(flicker_clip["path"], lifted, red_doc, red_cfg, progress=False)
    probe_cfg = Config(exposure_mode="off", wb_mode="off")
    a = analyze(plain, probe_cfg, progress=False)
    b = analyze(lifted, probe_cfg, progress=False)
    # more red in the picture -> less red gain still required
    assert np.mean(b["frames"]["wb_raw_r"]) < np.mean(a["frames"]["wb_raw_r"])


def test_reference_frame_white_point(flicker_clip):
    cfg = Config(wb_method="reference-frame", reference_frame="1.0", reference_roi="0.4,0.4,0.2,0.2")
    doc = analyze(flicker_clip["path"], cfg, progress=False)
    assert np.all(np.asarray(doc["frames"]["wb_gain_r"]) > 1.0)


# --- the lookup-table fast path ------------------------------------------

def test_lut_matches_the_elementwise_reference(analysed):
    """The optimisation must be exact, not merely close."""
    import numpy as np

    from uwnorm.apply import FrameProcessor, process_frame
    from uwnorm.core.device import Device

    cfg, doc = analysed
    table = build_gain_table(doc)
    dev = Device("cpu")
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    worker = FrameProcessor(table, cfg, doc["source"]["resolved_trc"], dev)
    assert worker.use_lut
    for index in (0, 7, table["n"] - 1):
        reference = process_frame(frame, table, index, cfg, doc["source"]["resolved_trc"], dev)
        assert np.array_equal(worker(frame, index), reference)


def test_red_compensation_forces_the_elementwise_path(analysed):
    from uwnorm.apply import FrameProcessor
    from uwnorm.core.device import Device

    cfg, doc = analysed
    worker = FrameProcessor(build_gain_table(doc), cfg.merged({"red_compensation": 1.0}),
                            doc["source"]["resolved_trc"], Device("cpu"))
    assert not worker.use_lut


def test_lut_is_exact_for_10_bit_output():
    import numpy as np

    from uwnorm.core import color, gains as g

    gain = np.array([1.4, 1.0, 0.8])
    lut = g.build_lut(gain, 0.8, "srgb", out_bits=16)
    assert lut.dtype == np.uint16
    encoded = (np.arange(256, dtype=np.float32) / 255.0)
    expected = np.clip(
        color.oetf(np.clip(g.soft_clip(color.eotf(encoded, "srgb") * np.float32(gain[0]), 0.8),
                           0.0, 1.0), "srgb") * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    assert np.array_equal(lut[0], expected)


def test_lut_is_monotonic():
    """A non-monotonic table would posterise or invert gradients."""
    import numpy as np

    from uwnorm.core import gains as g

    for gain in (0.5, 1.0, 2.5, 6.0):
        lut = g.build_lut(np.array([gain, gain, gain]), 0.8, "srgb")
        assert np.all(np.diff(lut[0].astype(int)) >= 0)


def test_mismatched_preview_range_is_refused(flicker_clip, tmp_path):
    """Applying a different window than was analysed would shift every gain."""
    cfg = Config(preview_range="1.0-3.0")
    doc = analyze(flicker_clip["path"], cfg, progress=False)
    with pytest.raises(ValueError, match="does not match"):
        apply_gains(flicker_clip["path"], str(tmp_path / "x.mp4"), doc,
                    cfg.merged({"preview_range": "2.0-4.0"}), progress=False)


def test_matching_preview_range_is_accepted(flicker_clip, tmp_path):
    cfg = Config(preview_range="1.0-3.0")
    doc = analyze(flicker_clip["path"], cfg, progress=False)
    out = str(tmp_path / "ok.mp4")
    apply_gains(flicker_clip["path"], out, doc, cfg, progress=False)
    assert probe(out).frame_count_estimate > 0


def test_output_is_tagged_with_its_transfer_function(analysed, flicker_clip, tmp_path):
    """An untagged output is how TRC ambiguity gets created in the first place."""
    cfg, doc = analysed
    out = str(tmp_path / "tagged.mp4")
    apply_gains(flicker_clip["path"], out, doc, cfg, progress=False)
    written = probe(out)
    assert written.color_trc == "iec61966-2-1"          # sRGB
    assert written.color_primaries == "bt709"
    assert written.color_space == "bt709"
    # ... and the tag round-trips through our own resolver
    from uwnorm.io.probe import resolve_trc
    assert resolve_trc(written, "auto") == doc["source"]["resolved_trc"]
