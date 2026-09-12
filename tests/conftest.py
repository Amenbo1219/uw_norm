"""Shared fixtures.

The synthetic clips are session-scoped: encoding them costs a second or two
each, and every test wants the same ones.
"""

from __future__ import annotations

import logging

import pytest

from uwnorm.logging_utils import setup_logging

from .synth import SynthSpec, write_clip

# Keep the test output readable; individual tests can raise the level again.
setup_logging()
logging.getLogger("uwnorm").setLevel(logging.ERROR)

SMALL = dict(width=240, height=136, n_frames=180, illuminant=(0.55, 1.0, 0.85))


@pytest.fixture(scope="session")
def clip_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("clips")


@pytest.fixture(scope="session")
def flicker_clip(clip_dir):
    """Exposure flicker + AWB hunting + slow drift, no cut."""
    spec = SynthSpec(**SMALL, wb_flicker=0.12, wb_drift=0.2)
    path = clip_dir / "flicker.mp4"
    ev = write_clip(path, spec)
    return {"path": str(path), "spec": spec, "ev": ev}


@pytest.fixture(scope="session")
def cut_clip(clip_dir):
    """Same, with a hard cut half way through."""
    spec = SynthSpec(**SMALL, wb_flicker=0.12, cut_at=90)
    path = clip_dir / "cut.mp4"
    ev = write_clip(path, spec)
    return {"path": str(path), "spec": spec, "ev": ev}


@pytest.fixture(scope="session")
def pan_clip(clip_dir):
    """A fast pan and no cut - the cut detector's false-positive stress test."""
    spec = SynthSpec(**SMALL, wb_flicker=0.08, pan_px_per_frame=9.0)
    path = clip_dir / "pan.mp4"
    ev = write_clip(path, spec)
    return {"path": str(path), "spec": spec, "ev": ev}


@pytest.fixture(scope="session")
def quiet_clip(clip_dir):
    """No flicker at all - the pipeline must leave it alone."""
    spec = SynthSpec(**SMALL, flicker_ev=0.0, jitter_ev=0.0, drift_ev=0.0,
                     moving_subject=False, audio=False)
    path = clip_dir / "quiet.mp4"
    ev = write_clip(path, spec)
    return {"path": str(path), "spec": spec, "ev": ev}
