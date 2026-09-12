"""Compute device selection and host<->device transfers.

The heavy per-pixel work of the apply pass (linearise, gain, soft clip, OETF on
a 3840x2160x3 float array, 30 times a second) is embarrassingly parallel, so it
runs on the GPU when torch + CUDA are present.  Everything falls back to numpy
when they are not: the algorithms are identical, only the array type changes.
"""

from __future__ import annotations

import numpy as np

from . import ops


class Device:
    """Thin wrapper around 'where do arrays live'."""

    def __init__(self, kind: str = "auto"):
        if kind not in ("auto", "cpu", "cuda"):
            raise ValueError(f"unknown device: {kind!r}")
        if kind == "auto":
            kind = "cuda" if ops.cuda_available() else "cpu"
        if kind == "cuda" and not ops.cuda_available():
            raise RuntimeError(
                "--device cuda requested but torch/CUDA is unavailable "
                "(install the 'gpu' extra: pip install 'uwnorm[gpu]')"
            )
        self.kind = kind
        self._torch = None
        if kind == "cuda":
            import torch

            self._torch = torch
            self._stream = torch.cuda.Stream()

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def describe(self) -> str:
        if self.is_cuda:
            return f"cuda ({self._torch.cuda.get_device_name(0)})"
        return "cpu (numpy)"

    def from_uint8(self, frame: np.ndarray):
        """uint8 HxWx3 host array -> float32 [0,1] array on the device."""
        if self.is_cuda:
            t = self._torch.from_numpy(frame)
            t = t.to("cuda", non_blocking=True)
            return t.to(self._torch.float32).div_(255.0)
        return frame.astype(np.float32) / 255.0

    def to_uint8(self, arr) -> np.ndarray:
        """float32 [0,1] device array -> uint8 HxWx3 host array (round-half-up)."""
        if self.is_cuda:
            t = arr.mul_(255.0).add_(0.5).clamp_(0.0, 255.0).to(self._torch.uint8)
            return t.cpu().numpy()
        return np.clip(arr * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)

    def to_uint16(self, arr, bits: int = 10) -> np.ndarray:
        """float32 [0,1] device array -> uint16 host array at ``bits`` depth."""
        peak = float((1 << bits) - 1)
        if self.is_cuda:
            t = arr.mul_(peak).add_(0.5).clamp_(0.0, peak).to(self._torch.int32)
            return t.cpu().numpy().astype(np.uint16)
        return np.clip(arr * peak + 0.5, 0.0, peak).astype(np.uint16)

    def to_numpy(self, arr) -> np.ndarray:
        if ops.is_torch(arr):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def synchronize(self) -> None:
        if self.is_cuda:
            self._torch.cuda.synchronize()
