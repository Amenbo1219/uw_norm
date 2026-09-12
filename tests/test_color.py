"""Transfer functions and the array backend adapter (F-02)."""

import numpy as np
import pytest

from uwnorm.core import color, ops
from uwnorm.core.device import Device

TRCS = ["srgb", "bt709", "gamma22", "gamma24", "linear"]


@pytest.mark.parametrize("trc", TRCS)
def test_eotf_oetf_roundtrip(trc):
    x = np.linspace(0.0, 1.0, 257, dtype=np.float32)
    assert np.allclose(color.oetf(color.eotf(x, trc), trc), x, atol=1e-6)


@pytest.mark.parametrize("trc", TRCS)
def test_eotf_is_monotonic_and_bounded(trc):
    x = np.linspace(0.0, 1.0, 257, dtype=np.float32)
    y = color.eotf(x, trc)
    assert np.all(np.diff(y) >= -1e-7)
    assert y.min() >= 0.0 and y.max() <= 1.0 + 1e-6


def test_eotf_anchors():
    # Both curves must pin black to black and white to white, or a "no-op"
    # pass would change the overall brightness of the clip.
    for trc in TRCS:
        assert color.eotf(np.float32(0.0), trc) == pytest.approx(0.0, abs=1e-6)
        assert color.eotf(np.float32(1.0), trc) == pytest.approx(1.0, abs=1e-6)


def test_srgb_midpoint_is_below_linear_mid():
    # sRGB 0.5 encodes about 21% of linear light; getting this backwards would
    # silently invert every gain decision.
    assert color.eotf(np.float32(0.5), "srgb") == pytest.approx(0.2140, abs=1e-3)


def test_luminance_weights_sum_to_one():
    assert sum(color.LUMA_WEIGHTS) == pytest.approx(1.0)
    white = np.ones((4, 4, 3), dtype=np.float32)
    assert color.luminance(white) == pytest.approx(np.ones((4, 4)), abs=1e-6)


def test_unknown_trc_rejected():
    with pytest.raises(ValueError):
        color.eotf(np.zeros(4, dtype=np.float32), "rec2020-pq")


@pytest.mark.skipif(not ops.cuda_available(), reason="CUDA not available")
@pytest.mark.parametrize("trc", TRCS)
def test_cuda_matches_numpy(trc):
    import torch

    x = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    cpu = color.eotf(x, trc)
    gpu = color.eotf(torch.from_numpy(x).cuda(), trc).cpu().numpy()
    assert np.allclose(cpu, gpu, atol=1e-6)


def test_device_uint8_roundtrip():
    dev = Device("cpu")
    frame = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    back = dev.to_uint8(dev.from_uint8(frame))
    assert np.array_equal(frame, back)


def test_device_cuda_rejected_without_torch(monkeypatch):
    monkeypatch.setattr(ops, "cuda_available", lambda: False)
    monkeypatch.setattr("uwnorm.core.device.ops.cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        Device("cuda")
