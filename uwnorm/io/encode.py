"""Encoding and muxing (3.2, F-11, A-5).

Video is encoded with PyAV rather than by piping raw frames to an ffmpeg
process, for one reason: a rawvideo pipe carries no timestamps, so VFR material
would silently be re-timed to a constant rate and drift out of sync with its
audio.  Writing through PyAV lets every frame keep the exact PTS it was decoded
with.

Audio, data (GoPro GPMF telemetry) and subtitle streams are never re-encoded:
the processed video is muxed against the original file with ``-c copy``, and
global metadata is carried over, so GPS and creation time survive (A-5).
"""

from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

#: our TRC names -> the AVCOL_TRC value ffmpeg tags the bitstream with.
#: Anything absent here is left "unspecified" rather than tagged wrongly.
AVCOL_TRC = {"bt709": 1, "gamma22": 4, "linear": 8, "srgb": 13}
AVCOL_PRI_BT709 = 1
AVCOL_SPC_BT709 = 1
AVCOL_RANGE_MPEG = 1   # limited ("tv") range, which is what yuv420p carries

#: encoder name -> (codec, container suffix, extra options builder)
ENCODERS = {
    "x264": ("libx264", ".mp4"),
    "x265": ("libx265", ".mp4"),
    "prores": ("prores_ks", ".mov"),
    "ffv1": ("ffv1", ".mkv"),
    "x264_nvenc": ("h264_nvenc", ".mp4"),
    "x265_nvenc": ("hevc_nvenc", ".mp4"),
}


def pick_pix_fmt(encoder: str, chroma: str, bit_depth: int) -> str:
    """Choose the encoder's working pixel format.

    Raising the intermediate chroma to 4:2:2 or 4:4:4 is offered because strong
    red gains on 4:2:0 chroma smear colour across a 2x2 block (premise P-4);
    ``--chroma 444`` keeps the corrected colour where it belongs at the cost of
    bitrate.
    """
    if encoder == "prores":
        return "yuva444p10le" if chroma == "444" else "yuv422p10le"
    if encoder.endswith("nvenc"):
        if bit_depth == 10:
            return "p010le"
        return "yuv444p" if chroma == "444" else "yuv420p"
    suffix = "10le" if bit_depth == 10 else ""
    return f"yuv{chroma}p{suffix}"


#: x264/x265 preset -> the nearest NVENC preset (p1 fastest ... p7 slowest).
_NVENC_PRESETS = {
    "ultrafast": "p1", "superfast": "p1", "veryfast": "p2", "faster": "p3",
    "fast": "p4", "medium": "p5", "slow": "p6", "slower": "p7", "veryslow": "p7",
}


def encoder_options(
    encoder: str, crf: int, threads: int, preset: str = "medium"
) -> dict[str, str]:
    """Codec options.

    Thread count is always pinned: x264/x265 partition the picture by thread, so
    the bitstream depends on how many threads ran.  Pinning it is what makes
    N-01 / A-6 (byte-identical output across runs) hold.
    """
    if encoder == "x264":
        return {"crf": str(crf), "preset": preset, "threads": str(threads)}
    if encoder == "x265":
        return {
            "crf": str(crf),
            "preset": preset,
            "x265-params": f"log-level=error:frame-threads={max(1, min(threads, 16))}",
        }
    if encoder == "prores":
        return {"profile": "3", "vendor": "apl0"}
    if encoder == "ffv1":
        return {"level": "3", "coder": "1", "context": "1", "g": "1", "threads": str(threads)}
    if encoder.endswith("nvenc"):
        return {
            "preset": _NVENC_PRESETS.get(preset, "p5"),
            "rc": "vbr", "cq": str(crf), "b:v": "0",
        }
    raise ValueError(f"unknown encoder: {encoder!r}")


class DenoiseFilter:
    """Optional temporal/spatial denoise (F-11), run in-process via libavfilter.

    Gain amplifies noise, and because the source ISO was moving the noise level
    moves too (premise P-3), so evening out the brightness can make the
    grain *more* obvious.  This runs the real ffmpeg filters on the corrected
    frames before they reach the encoder, keeping the whole job to one encode.
    """

    SPECS = {
        "hqdn3d": "hqdn3d=luma_spatial=2:chroma_spatial=1.5:luma_tmp=4:chroma_tmp=4",
        "nlmeans": "nlmeans=s=1.5:p=5:r=11",
        "vaguedenoiser": "vaguedenoiser=threshold=2:method=soft",
    }

    def __init__(self, name: str, width: int, height: int, pix_fmt: str, time_base: Fraction):
        if name not in self.SPECS:
            raise ValueError(f"unknown denoiser: {name!r}")
        self.graph = av.filter.Graph()
        src = self.graph.add_buffer(
            width=width, height=height, format=pix_fmt, time_base=time_base
        )
        spec = self.SPECS[name]
        filt_name, _, args = spec.partition("=")
        node = self.graph.add(filt_name, args or None)
        src.link_to(node)
        sink = self.graph.add("buffersink")
        node.link_to(sink)
        self.graph.configure()

    def push(self, frame):
        """Feed one frame in, yield whatever comes out (may be zero or more)."""
        self.graph.push(frame)
        while True:
            try:
                yield self.graph.pull()
            except av.error.BlockingIOError:
                return
            except av.error.EOFError:
                return

    def flush(self):
        try:
            self.graph.push(None)
        except Exception:
            return
        while True:
            try:
                yield self.graph.pull()
            except (av.error.BlockingIOError, av.error.EOFError):
                return


class VideoEncoder:
    """Write processed frames, preserving presentation timestamps."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        rate: Fraction,
        time_base: Fraction,
        encoder: str = "x264",
        crf: int = 16,
        chroma: str = "420",
        bit_depth: int = 8,
        threads: int = 1,
        denoise: str = "off",
        preset: str = "medium",
        trc: str = "srgb",
    ):
        codec, _suffix = ENCODERS[encoder]
        self.path = str(path)
        self.container = av.open(self.path, mode="w")
        self.stream = self.container.add_stream(codec, rate=rate)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = pick_pix_fmt(encoder, chroma, bit_depth)
        self.stream.time_base = time_base
        self.stream.codec_context.thread_count = max(1, int(threads))
        self.stream.options = encoder_options(encoder, crf, threads, preset)
        # Tag the bitstream so players do not have to guess how to decode it.
        # The clip is written back with the same transfer function it was read
        # with, so the tag must say so - an untagged file is exactly how the
        # ambiguity that --input-trc exists to resolve gets created.
        ctx = self.stream.codec_context
        ctx.color_range = AVCOL_RANGE_MPEG
        ctx.color_primaries = AVCOL_PRI_BT709
        ctx.colorspace = AVCOL_SPC_BT709
        if trc in AVCOL_TRC:
            ctx.color_trc = AVCOL_TRC[trc]
        self.bit_depth = bit_depth
        self.src_format = "rgb48le" if bit_depth == 10 else "rgb24"
        self.time_base = time_base
        self._denoise = None
        if denoise != "off":
            self._denoise = DenoiseFilter(
                denoise, width, height, self.src_format, time_base
            )
        self._count = 0

    def write(self, array: np.ndarray, pts: int | None) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format=self.src_format)
        frame.pts = pts if pts is not None else self._count
        frame.time_base = self.time_base
        self._count += 1
        if self._denoise is not None:
            for filtered in self._denoise.push(frame):
                self._mux(filtered)
        else:
            self._mux(frame)

    def _mux(self, frame) -> None:
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self._denoise is not None:
            for filtered in self._denoise.flush():
                self._mux(filtered)
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH")
    return exe


def mux_with_source(video_path: str, source_path: str, output_path: str) -> None:
    """Combine the processed video with the original non-video streams (A-5)."""
    from .probe import probe

    info = probe(source_path)
    cmd = [
        _ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path, "-i", source_path,
        "-map", "0:v:0",
    ]
    if info.has_audio:
        cmd += ["-map", "1:a?"]
    if info.has_data:
        cmd += ["-map", "1:d?", "-map", "1:s?"]
    cmd += [
        "-c", "copy",
        "-map_metadata", "1",
        "-movflags", "use_metadata_tags+faststart",
        "-fps_mode", "passthrough",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Older ffmpeg builds spell it -vsync; retry once before giving up.
        retry = [c if c != "-fps_mode" else "-vsync" for c in cmd]
        result = subprocess.run(retry, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mux failed: {result.stderr.strip()[:800]}")


def make_comparison(before: str, after: str, output: str, crf: int = 18) -> None:
    """Side-by-side Before/After video for review (F-08 --compare)."""
    cmd = [
        _ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", before, "-i", after,
        "-filter_complex",
        "[0:v]drawtext=text='BEFORE':x=20:y=20:fontsize=h/20:fontcolor=white:"
        "box=1:boxcolor=black@0.5[a];"
        "[1:v]drawtext=text='AFTER':x=20:y=20:fontsize=h/20:fontcolor=white:"
        "box=1:boxcolor=black@0.5[b];[a][b]hstack=inputs=2",
        "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an",
        output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # drawtext needs libfreetype; fall back to a plain stack.
        cmd_plain = [
            _ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", before, "-i", after, "-filter_complex", "[0:v][1:v]hstack=inputs=2",
            "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", output,
        ]
        result = subprocess.run(cmd_plain, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"comparison render failed: {result.stderr.strip()[:500]}")
