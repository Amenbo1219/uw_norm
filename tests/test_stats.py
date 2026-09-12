"""Robust per-frame estimators (F-03.1, F-04).

Each test targets the specific failure a naive estimator would have.
"""

import numpy as np
import pytest

from uwnorm.core import roi, stats


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    return (rng.random((64, 64), dtype=np.float32) * 0.4 + 0.1)


def test_geometric_mean_tracks_exposure_exactly(scene):
    for factor in (0.5, 1.25, 2.0):
        ratio = stats.trimmed_geometric_mean((scene * factor).ravel()) / \
            stats.trimmed_geometric_mean(scene.ravel())
        assert ratio == pytest.approx(factor, rel=1e-4)


def test_trimmed_mean_ignores_outliers():
    values = np.concatenate([np.full(1000, 0.5), np.full(6, 1e6)])
    assert stats.trimmed_mean(values) == pytest.approx(0.5, abs=1e-3)
    assert np.mean(values) > 1000  # the estimator it replaces


def test_ratio_estimate_recovers_a_global_scale(scene):
    valid = np.ones_like(scene, dtype=bool)
    assert stats.ratio_estimate(scene * 1.3, scene, valid, valid) == pytest.approx(1.3, rel=1e-5)


def test_ratio_estimate_ignores_a_moving_subject(scene):
    """A fish over 40% of the frame must not be read as an exposure change."""
    valid = np.ones_like(scene, dtype=bool)
    moved = scene * 1.3
    moved[:26, :] = 0.9
    assert stats.ratio_estimate(moved, scene, valid, valid) == pytest.approx(1.3, rel=0.02)


def test_ratio_estimate_falls_back_when_starved(scene):
    valid = np.zeros_like(scene, dtype=bool)
    valid[:2, :2] = True
    assert stats.ratio_estimate(scene, scene, valid, valid) == 1.0


def test_valid_mask_excludes_clipped_and_crushed():
    frame = np.zeros((3, 1, 3), dtype=np.float32)
    frame[0] = 0.5
    frame[1] = 0.99   # blown
    frame[2] = 0.002  # crushed
    mask = stats.valid_mask(frame, hi=0.98, lo=0.01)
    assert mask.ravel().tolist() == [True, False, False]


def test_minkowski_order_ranks_as_expected(scene):
    flat = scene.ravel()
    p1 = stats.minkowski_mean(flat, 1.0)
    p5 = stats.minkowski_mean(flat, 5.0)
    assert p1 == pytest.approx(float(np.mean(flat)), rel=1e-5)
    assert p1 < p5 < stats.percentile_mean(flat, 99.0)


def test_channel_ratios_isolate_a_colour_shift():
    rng = np.random.default_rng(3)
    prev = rng.random((48, 48, 3), dtype=np.float32) * 0.4 + 0.1
    cur = prev * np.array([1.2, 1.0, 0.9], dtype=np.float32)
    valid = np.ones((48, 48), dtype=bool)
    got = stats.channel_ratios(cur, prev, valid, valid)
    assert got == pytest.approx([1.2, 1.0, 0.9], rel=1e-5)


def test_gain_invariant_histogram_ignores_diagonal_gain():
    rng = np.random.default_rng(5)
    frame = rng.random((64, 64, 3), dtype=np.float32) * 0.5 + 0.1
    mask = np.ones((64, 64), dtype=bool)
    a = stats.gain_invariant_histogram(frame, mask)
    b = stats.gain_invariant_histogram(frame * np.array([2.0, 1.5, 0.7], np.float32), mask)
    assert np.abs(a - b).sum() < 1e-9


def test_gain_invariant_thumbnail_ignores_diagonal_gain():
    rng = np.random.default_rng(6)
    frame = rng.random((64, 64, 3), dtype=np.float32) * 0.5 + 0.1
    a = stats.gain_invariant_thumbnail(frame)
    b = stats.gain_invariant_thumbnail(frame * np.array([2.0, 1.5, 0.7], np.float32))
    assert np.abs(a - b).max() < 1e-5


def test_roi_center_and_rect():
    assert roi.build_roi_mask(100, 80, "center:0.5").mean() == pytest.approx(0.25)
    mask = roi.build_roi_mask(100, 80, "0,0,0.5,1.0")
    assert mask[:, :50].all() and not mask[:, 50:].any()


def test_exclude_rect_removes_a_hotspot():
    mask = roi.build_roi_mask(100, 80, None, "0,0,0.25,0.25")
    assert not mask[:20, :25].any()
    assert mask[40:, 50:].all()


def test_empty_roi_is_an_error():
    with pytest.raises(ValueError, match="no pixels"):
        roi.build_roi_mask(100, 80, "0,0,0.5,0.5", "0,0,1.0,1.0")
