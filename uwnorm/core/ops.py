"""Array backend adapter.

The pixel pipeline must run unchanged on numpy arrays (CPU, the default and the
only requirement of the slim Docker image) and on torch tensors (CUDA).  Rather
than duplicating the maths, every operation that is spelled differently between
the two libraries is funnelled through this module.

Only element-wise operations are needed by the apply pass, so the adapter stays
tiny and has no measurable overhead.
"""

from __future__ import annotations

import numpy as np

try:  # torch is an optional dependency (GPU extra)
    import torch as _torch
except Exception:  # pragma: no cover - exercised on CPU-only installs
    _torch = None


def torch_available() -> bool:
    return _torch is not None


def cuda_available() -> bool:
    return _torch is not None and _torch.cuda.is_available()


def is_torch(x) -> bool:
    return _torch is not None and isinstance(x, _torch.Tensor)


def clip(x, lo, hi):
    return x.clamp(lo, hi) if is_torch(x) else np.clip(x, lo, hi)


def where(cond, a, b):
    return _torch.where(cond, a, b) if is_torch(cond) else np.where(cond, a, b)


def tanh(x):
    return _torch.tanh(x) if is_torch(x) else np.tanh(x)


def power(x, p):
    return _torch.pow(x, p) if is_torch(x) else np.power(x, p)


def maximum(x, y):
    if is_torch(x):
        y = y if is_torch(y) else _torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return _torch.maximum(x, y)
    return np.maximum(x, y)


def full_like(x, value):
    return _torch.full_like(x, value) if is_torch(x) else np.full_like(x, value)


def asarray_like(x, values):
    """Turn a small python/numpy sequence into an array living next to ``x``."""
    if is_torch(x):
        return _torch.as_tensor(np.asarray(values, dtype=np.float32), device=x.device)
    return np.asarray(values, dtype=np.float32)
