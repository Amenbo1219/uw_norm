"""Temporal filtering, cut detection, exposure and WB curve maths."""

import numpy as np
import pytest

from uwnorm.core import cuts, exposure, gains, smooth, wb

FPS = 30.0


def series(n=900, slow_hz=0.15, slow=0.30, fast_hz=6.0, fast=0.05, seed=0):
    t = np.arange(n) / FPS
    rng = np.random.default_rng(seed)
    slow_part = slow * np.sin(2 * np.pi * slow_hz * t)
    fast_part = fast * np.sin(2 * np.pi * fast_hz * t) + rng.normal(0, fast / 3, n)
    return slow_part, fast_part


# --- smoothing -----------------------------------------------------------

def test_lowpass_keeps_slow_and_kills_fast():
    slow, fast = series()
    out = smooth.lowpass(slow + fast, 1.5, FPS)
    assert np.abs(out - slow).max() < 0.10       # the dive's real arc survives
    assert np.std(out - slow) < 0.02             # the flicker does not


def test_deflicker_reduces_high_frequency_by_over_80_percent():
    slow, fast = series()
    x = slow + fast
    before = smooth.highpass_energy(x, 1.5, FPS)
    after = smooth.highpass_energy(smooth.lowpass(x, 1.5, FPS), 1.5, FPS)
    assert 100 * (1 - after / before) > 80


def test_filters_do_not_bleed_across_a_cut():
    x = np.concatenate([np.zeros(300), np.ones(300)])
    segs = smooth.segments_from_cuts(600, [300])
    for out in (smooth.lowpass(x, 1.5, FPS, segs),
                smooth.smooth_series(x, "savgol", 1.5, FPS, segs),
                smooth.smooth_series(x, "median-gauss", 1.5, FPS, segs)):
        assert out[:300] == pytest.approx(np.zeros(300), abs=1e-9)
        assert out[300:] == pytest.approx(np.ones(300), abs=1e-9)


def test_short_segments_do_not_crash():
    segs = smooth.segments_from_cuts(10, [1, 2, 5])
    for n in (1, 2, 3, 7):
        x = np.linspace(0, 1, 10)
        assert np.isfinite(smooth.lowpass(x, 1.5, FPS, segs)).all()
        assert np.isfinite(smooth.smooth_series(x, "savgol", 1.5, FPS, segs)).all()


def test_segments_from_cuts_partitions_exactly():
    segs = smooth.segments_from_cuts(100, [10, 50, 50, 0, 200])
    assert segs == [(0, 10), (10, 50), (50, 100)]
    assert sum(b - a for a, b in segs) == 100


def test_tau_controls_the_cutoff():
    slow, fast = series(fast_hz=1.0, fast=0.1)
    x = slow + fast
    # A 1 Hz wiggle survives a 0.5 s tau and is removed by a 4 s one.
    assert np.std(smooth.lowpass(x, 0.5, FPS) - x) < np.std(smooth.lowpass(x, 4.0, FPS) - x)


# --- cut detection --------------------------------------------------------

def test_histogram_distance_bounds():
    a = np.zeros(9); a[0] = 1.0
    b = np.zeros(9); b[8] = 1.0
    assert cuts.histogram_distance(a, a) == pytest.approx(0.0)
    assert cuts.histogram_distance(a, b) == pytest.approx(1.0)


def test_spatial_distance_tolerates_a_shift():
    rng = np.random.default_rng(1)
    t1 = rng.normal(0, 1, (16, 16, 3))
    shifted = np.roll(t1, 2, axis=1)
    unrelated = rng.normal(0, 1, (16, 16, 3))
    assert cuts.spatial_distance(t1, shifted) < cuts.spatial_distance(t1, unrelated)


def test_detect_cuts_collapses_a_burst():
    scores = np.zeros(20)
    scores[10] = 0.5
    scores[11] = 0.7   # same transition, one frame later
    assert cuts.detect_cuts(scores, 0.35) == [11]


def test_manual_cuts_map_to_nearest_frame():
    ts = np.arange(100) / 30.0
    assert cuts.times_to_indices([1.0, 2.0], ts) == [30, 60]
    assert cuts.times_to_indices([0.0], ts) == []   # frame 0 is not a cut


# --- exposure -------------------------------------------------------------

def test_cumulative_restarts_at_every_cut():
    ratios = np.full(10, 0.1)
    segs = smooth.segments_from_cuts(10, [5])
    out = exposure.cumulative_log_exposure(ratios, segs)
    assert out[0] == 0.0 and out[5] == 0.0
    assert out[4] == pytest.approx(0.4)


def test_anchor_beats_the_raw_integrator():
    n = 900
    truth = series(n)[0]
    rng = np.random.default_rng(2)
    increments = np.diff(truth, prepend=truth[0]) + rng.normal(0, 0.003, n)
    absolute = truth + rng.normal(0, 0.06, n)       # noisy but drift-free
    segs = smooth.segments_from_cuts(n, [])
    integrated = exposure.cumulative_log_exposure(increments, segs)
    anchored = exposure.anchor(integrated, absolute, FPS, segs)
    drift = np.abs(integrated - truth - np.mean(integrated - truth)).max()
    assert np.abs(anchored - truth).max() < drift


@pytest.mark.parametrize("mode,expected_flat", [("lock", True), ("deflicker", False)])
def test_exposure_modes(mode, expected_flat):
    slow, fast = series()
    curve = slow + fast
    segs = smooth.segments_from_cuts(len(curve), [])
    gain, _ = exposure.exposure_log_gain(curve, mode, 1.5, FPS, segs, 10.0)
    after = exposure.predicted_curve(curve, gain)
    if expected_flat:
        assert np.std(after) < 1e-6           # lock flattens everything
    else:
        assert np.std(after - slow) < 0.02    # deflicker keeps the arc
        assert np.std(after) > 0.1


def test_exposure_off_is_a_no_op():
    curve = series()[0]
    segs = smooth.segments_from_cuts(len(curve), [])
    gain, clamped = exposure.exposure_log_gain(curve, "off", 1.5, FPS, segs, 1.5)
    assert not gain.any() and not clamped.any()


def test_exposure_gain_is_clamped_and_flagged():
    curve = np.concatenate([np.zeros(300), np.full(300, 8.0)])
    segs = smooth.segments_from_cuts(600, [])
    gain, clamped = exposure.exposure_log_gain(curve, "lock", 1.5, FPS, segs, 1.5)
    limit = 1.5 * np.log(2)
    assert np.abs(gain).max() <= limit + 1e-9
    assert clamped.any()


# --- white balance --------------------------------------------------------

def test_gains_from_means_normalises_green_to_one():
    gr, gb = wb.gains_from_means(np.array([0.25, 0.5, 0.4]))
    assert (gr, gb) == pytest.approx((2.0, 1.25))


def test_gains_from_means_survives_a_black_frame():
    assert wb.gains_from_means(np.array([0.0, 0.0, 0.0])) == (1.0, 1.0)
    assert wb.gains_from_means(np.array([np.nan, 1.0, 1.0])) == (1.0, 1.0)


def test_illuminant_curve_recovers_colour_flicker_a_smoother_cannot():
    """The whole reason for the ratio channel.

    The absolute estimate is dominated by scene colour; the ratios see the true
    illuminant.  A pipeline that only smoothed the absolute estimate would track
    the scene, not the camera.
    """
    n = 600
    t = np.arange(n) / FPS
    true_imbalance = 0.25 * np.sin(2 * np.pi * 4.0 * t) + 0.1 * t / t[-1]
    rng = np.random.default_rng(4)
    scene_noise = rng.normal(0, 0.15, n)              # what the frame's colour does
    absolute = true_imbalance + scene_noise
    deltas = np.diff(true_imbalance, prepend=true_imbalance[0])
    segs = smooth.segments_from_cuts(n, [])
    curve = wb.illuminant_curve(absolute, deltas, "savgol", 3.0, FPS, segs)
    # the curve is the *gain*, i.e. the negative of the imbalance
    assert np.corrcoef(-curve, true_imbalance)[0, 1] > 0.97
    assert np.std(-curve - true_imbalance) < np.std(absolute - true_imbalance)


@pytest.mark.parametrize("mode", ["drift", "fixed", "off"])
def test_wb_modes_remove_the_fast_band(mode):
    n = 600
    t = np.arange(n) / FPS
    curve = 0.2 * np.sin(2 * np.pi * 4.0 * t) + 0.3 * t / t[-1]
    segs = smooth.segments_from_cuts(n, [])
    applied, _ = wb.wb_log_gain(curve, mode, 3.0, FPS, segs, 10.0)
    residual = curve - applied
    hf_before = smooth.highpass_energy(curve, 3.0, FPS)
    hf_after = smooth.highpass_energy(residual, 3.0, FPS)
    if mode == "off":
        assert hf_after == pytest.approx(hf_before)
    else:
        assert 100 * (1 - hf_after / hf_before) > 80
    if mode == "fixed":
        # the slow depth-driven change is deliberately left in the picture
        assert np.std(residual) > 0.05
    elif mode == "drift":
        assert np.std(residual) < 1e-9


def test_wb_gain_is_clamped():
    curve = np.full(100, 5.0)
    segs = smooth.segments_from_cuts(100, [])
    applied, clamped = wb.wb_log_gain(curve, "drift", 3.0, FPS, segs, 3.0)
    assert np.exp(applied).max() <= 3.0 + 1e-9
    assert clamped.all()


def test_red_compensation_only_lifts_dark_red():
    rgb = np.zeros((2, 1, 3), dtype=np.float32)
    rgb[0] = [0.05, 0.6, 0.5]   # red-starved
    rgb[1] = [0.95, 0.6, 0.5]   # already red
    out = wb.red_compensate(rgb, 1.0, mean_r=0.1, mean_g=0.6)
    assert out[0, 0, 0] > rgb[0, 0, 0] * 2
    assert out[1, 0, 0] == pytest.approx(rgb[1, 0, 0], abs=0.02)
    assert out[..., 1:] == pytest.approx(rgb[..., 1:])


def test_red_compensation_disabled_is_identity():
    rgb = np.full((2, 2, 3), 0.4, dtype=np.float32)
    assert wb.red_compensate(rgb, 0.0, 0.1, 0.6) is rgb


# --- gain composition -----------------------------------------------------

def test_compose_folds_exposure_and_wb_into_one_triplet():
    table = gains.compose(np.array([np.log(2.0)]), np.array([1.5]), np.array([0.8]))
    assert table[0] == pytest.approx([3.0, 2.0, 1.6])


def test_soft_clip_is_monotonic_continuous_and_bounded():
    x = np.linspace(0.0, 4.0, 4001, dtype=np.float64)
    y = gains.soft_clip(x, 0.8)
    assert np.all(np.diff(y) >= 0)
    assert y.max() < 1.0
    assert y[x <= 0.8] == pytest.approx(x[x <= 0.8], abs=1e-9)
    slope = np.diff(y) / np.diff(x)
    knee = int(0.8 / 4.0 * 4000)
    assert slope[knee - 1] == pytest.approx(slope[knee + 1], abs=0.01)


def test_apply_gain_matches_hand_computation():
    frame = np.full((2, 2, 3), 0.2, dtype=np.float32)
    out = gains.apply_gain(frame, np.array([2.0, 1.0, 0.5]), 0.9)
    assert out[0, 0] == pytest.approx([0.4, 0.2, 0.1], abs=1e-6)


def test_interpolate_fills_strided_analysis():
    out = gains.interpolate_to_frames(np.array([0.0, 1.0, 2.0]), np.array([0, 2, 4]), 5)
    assert out == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])
    table = gains.interpolate_to_frames(np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([0, 2]), 3)
    assert table.shape == (3, 2)
