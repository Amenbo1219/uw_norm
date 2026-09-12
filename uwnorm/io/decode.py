"""Frame decoding (PyAV), for both passes.

The analysis pass asks for a *small* frame: swscale downscales inside the
decoder's own output conversion, which is far cheaper than handing a 4K numpy
array to Python and resizing it there.  The apply pass asks for full
resolution.  Both yield the presentation timestamp so that VFR material keeps
its timing (3.1, risk table "VFR PTS drift").
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator

import av
import numpy as np


@dataclass
class DecodedFrame:
    index: int          # sequential index within the decoded stream
    pts: int | None     # raw presentation timestamp in stream time_base units
    time: float         # seconds
    array: np.ndarray   # HxWx3 uint8 RGB


def analysis_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Downscale target that keeps the aspect ratio, never upscales, and stays
    even in both axes (swscale is happier, and chroma-aware filters need it)."""
    scale = min(1.0, float(long_edge) / max(width, height))
    w = max(2, int(round(width * scale)) & ~1)
    h = max(2, int(round(height * scale)) & ~1)
    return w, h


class VideoSource:
    """Iterate RGB frames out of a container."""

    def __init__(self, path: str, hwaccel: str = "none", threads: int = 0):
        self.path = path
        self.hwaccel = hwaccel
        self.threads = threads
        self._container = None

    def __enter__(self) -> "VideoSource":
        self._open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _open(self) -> None:
        kwargs = {}
        if self.hwaccel == "cuda":
            try:
                kwargs["hwaccel"] = av.codec.hwaccel.HWAccel(
                    device_type="cuda", allow_software_fallback=True
                )
            except Exception:  # pragma: no cover - depends on the PyAV build
                kwargs = {}
        self._container = av.open(self.path, **kwargs)
        self.stream = self._container.streams.video[0]
        self.stream.thread_type = "AUTO"
        if self.threads:
            self.stream.thread_count = int(self.threads)

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None

    @property
    def time_base(self) -> Fraction:
        return self.stream.time_base or Fraction(1, 90000)

    def frames(
        self,
        width: int | None = None,
        height: int | None = None,
        stride: int = 1,
        start: float | None = None,
        end: float | None = None,
    ) -> Iterator[DecodedFrame]:
        """Yield frames as uint8 RGB arrays.

        ``stride`` skips *measurement*, not decoding: inter-frame codecs need
        every frame decoded anyway, and skipping the numpy conversion is where
        the saving actually is.
        """
        if self._container is None:
            self._open()
        if start:
            # Seek to the keyframe at or before ``start``; frames before the
            # requested time are dropped below, so the output is exact.
            offset = int(start / float(self.time_base))
            self._container.seek(offset, stream=self.stream, backward=True)

        index = 0
        for frame in self._container.decode(video=0):
            t = float(frame.pts * self.time_base) if frame.pts is not None else float(index / 30.0)
            if start is not None and t < start - 1e-9:
                continue
            if end is not None and t > end + 1e-9:
                break
            if index % stride == 0:
                if width and height and (frame.width, frame.height) != (width, height):
                    array = frame.reformat(
                        width=width, height=height, format="rgb24"
                    ).to_ndarray()
                else:
                    array = frame.to_ndarray(format="rgb24")
                yield DecodedFrame(index=index, pts=frame.pts, time=t, array=array)
            index += 1
