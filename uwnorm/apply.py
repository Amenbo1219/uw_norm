"""Pass 2 - application (F-01, 5.2 option A).

Full-resolution decode, one channel-wise gain in linear light, re-encode.

The pass is a three-stage pipeline running in three threads:

    decode (PyAV)  ->  queue  ->  gain  ->  queue  ->  encode (PyAV)

All three stages release the GIL for the bulk of their work - libavcodec inside
PyAV, OpenCV inside the gain stage - so the threads genuinely overlap.  The
queues are small and bounded, which is what keeps peak memory proportional to
the frame size rather than the clip length (N-03).

The gain stage has two implementations.  :class:`FrameProcessor` normally folds
the whole per-pixel chain into a per-channel lookup table (see
:func:`uwnorm.core.gains.build_lut`), which is both exact and roughly twenty
times faster than evaluating it per pixel.  Only ``--red-compensation``, which
mixes green into red and so is not separable per channel, falls back to the
element-wise path in :func:`process_frame` - and that one honours ``--device``.
"""

from __future__ import annotations

import queue
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np

from .config import Config, parse_range
from .core import color, device as device_mod, gains as gains_mod, wb as wb_mod
from .io import decode as decode_mod
from .io.encode import ENCODERS, VideoEncoder, mux_with_source
from .io.probe import probe
from .logging_utils import Timer, get_logger, log_event

_SENTINEL = object()

#: encoders that cannot legally live in a given container
_CONTAINER_RULES = {
    ".mp4": {"ffv1", "prores"},
    ".m4v": {"ffv1", "prores"},
    ".mov": {"ffv1"},
}


def check_container(output: str, encoder: str) -> None:
    suffix = Path(output).suffix.lower()
    bad = _CONTAINER_RULES.get(suffix, set())
    if encoder in bad:
        want = ENCODERS[encoder][1]
        raise ValueError(
            f"{encoder} cannot be stored in a {suffix} container - "
            f"use a {want} output path (e.g. {Path(output).with_suffix(want)})"
        )


def _ranges_match(a, b, tol: float = 1e-6) -> bool:
    return all(
        (x is None and y is None) or (x is not None and y is not None and abs(x - y) <= tol)
        for x, y in zip(a, b)
    )


def build_gain_table(doc: dict) -> dict[str, np.ndarray]:
    """Expand the analysis series onto every frame of the apply pass (F-10).

    This is also the seam an artist edits: hand-modified ``exposure_log_gain`` /
    ``wb_gain_r`` / ``wb_gain_b`` arrays in the JSON are picked up here with no
    re-analysis, because nothing downstream re-derives them.
    """
    frames = doc["frames"]
    src_index = np.asarray(frames["index"], dtype=np.float64)
    n = int(src_index[-1]) + 1

    exp_gain = gains_mod.interpolate_to_frames(
        np.asarray(frames["exposure_log_gain"]), src_index, n
    )
    gain_r = gains_mod.interpolate_to_frames(np.asarray(frames["wb_gain_r"]), src_index, n)
    gain_b = gains_mod.interpolate_to_frames(np.asarray(frames["wb_gain_b"]), src_index, n)
    table = gains_mod.compose(exp_gain, gain_r, gain_b)
    return {
        "gains": table,
        "mean_r": gains_mod.interpolate_to_frames(np.asarray(frames["mean_r"]), src_index, n),
        "mean_g": gains_mod.interpolate_to_frames(np.asarray(frames["mean_g"]), src_index, n),
        "n": n,
    }


def process_frame(array, table, index: int, cfg: Config, trc: str, dev):
    """Reference implementation: one frame, element-wise (F-02, F-03.6, F-04).

    Used whenever ``--red-compensation`` is on, because that step mixes the
    green pixel into the red one and so cannot be expressed as a per-channel
    lookup.  :class:`FrameProcessor` uses the much faster table path otherwise.
    """
    idx = min(index, table["n"] - 1)
    encoded = dev.from_uint8(array)
    linear = color.eotf(encoded, trc)
    if cfg.red_compensation > 0:
        linear = wb_mod.red_compensate(
            linear, cfg.red_compensation, table["mean_r"][idx], table["mean_g"][idx]
        )
    corrected = gains_mod.apply_gain(linear, table["gains"][idx], cfg.soft_clip_knee)
    out = color.oetf(corrected, trc)
    if cfg.bit_depth == 10:
        return dev.to_uint16(out, bits=16)
    return dev.to_uint8(out)


class FrameProcessor:
    """Applies the gain curves to frames, by lookup table where possible.

    The table path is selected automatically; it produces byte-identical output
    to :func:`process_frame` (there is a test asserting exactly that) and is
    roughly 20x faster, which is what lets the CPU-only build meet N-02.
    """

    def __init__(self, table: dict, cfg: Config, trc: str, dev):
        self.table = table
        self.cfg = cfg
        self.trc = trc
        self.dev = dev
        self.out_bits = 16 if cfg.bit_depth == 10 else 8
        self.use_lut = cfg.red_compensation <= 0
        self._cache: tuple | None = None

    def describe(self) -> str:
        if self.use_lut:
            # The table path stays on the CPU deliberately: cv2.LUT runs at
            # ~113 fps on a 4K frame, faster than shipping the frame across PCIe
            # to do the same lookup on the GPU.  The device setting still
            # governs the element-wise path below.
            return "lut (cv2, cpu)"
        return f"elementwise ({self.dev.describe()})"

    def _lut(self, index: int):
        # Consecutive frames very often share a gain triplet to the last bit
        # (the curves are smooth and rounded), so caching saves rebuilding it.
        key = tuple(self.table["gains"][index])
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        lut = gains_mod.to_cv_lut(
            gains_mod.build_lut(
                self.table["gains"][index], self.cfg.soft_clip_knee, self.trc, self.out_bits
            )
        )
        self._cache = (key, lut)
        return lut

    def __call__(self, array, index: int):
        idx = min(index, self.table["n"] - 1)
        if not self.use_lut:
            return process_frame(array, self.table, idx, self.cfg, self.trc, self.dev)
        return gains_mod.apply_lut(array, self._lut(idx))


def apply_gains(
    source: str,
    output: str,
    doc: dict,
    cfg: Config,
    progress: bool = True,
) -> dict:
    """Run the apply pass; returns a small summary dict."""
    logger = get_logger()
    check_container(output, cfg.encoder)
    info = probe(source)
    trc = doc["source"]["resolved_trc"]
    table = build_gain_table(doc)

    # The gain table is indexed by position within the analysed range, so the
    # apply pass must cover exactly the same range.  Silently applying frame
    # 0's gain to what is really frame 900 would be far worse than an error.
    analysed = doc["analysis"].get("range") or [None, None]
    start, end = analysed[0], analysed[1]
    if cfg.preview_range:
        requested = parse_range(cfg.preview_range)
        if (start, end) != (None, None) and not _ranges_match(requested, (start, end)):
            raise ValueError(
                f"--preview-range {cfg.preview_range} does not match the range this "
                f"analysis covers ({start}-{end}); re-run the analysis for that "
                f"range (uwnorm run ... --preview-range {cfg.preview_range})"
            )
        start, end = requested

    threads = cfg.jobs or 0
    dev = device_mod.Device(cfg.device)
    log_event(
        "apply.start", source=Path(source).name, output=Path(output).name,
        device=dev.describe(), encoder=f"{cfg.encoder}/{cfg.encoder_preset}", crf=cfg.crf,
        pix_bits=cfg.bit_depth, chroma=cfg.chroma, denoise=cfg.denoise,
        frames=table["n"],
    )

    tmp_video = Path(output).with_suffix(f".uwnorm-video{ENCODERS[cfg.encoder][1]}")
    tmp_video.parent.mkdir(parents=True, exist_ok=True)

    decode_q: queue.Queue = queue.Queue(maxsize=6)
    encode_q: queue.Queue = queue.Queue(maxsize=6)
    errors: list[BaseException] = []
    written = 0

    def decoder():
        try:
            with decode_mod.VideoSource(source, cfg.hwaccel, threads) as src:
                for frame in src.frames(stride=1, start=start, end=end):
                    decode_q.put((frame.index, frame.pts, frame.array))
        except BaseException as exc:  # noqa: BLE001 - forwarded to the main thread
            errors.append(exc)
        finally:
            decode_q.put(_SENTINEL)

    worker = FrameProcessor(table, cfg, trc, dev)
    log_event("apply.path", path=worker.describe())

    def processor():
        try:
            while True:
                item = decode_q.get()
                if item is _SENTINEL:
                    break
                index, pts, array = item
                encode_q.put((worker(array, index), pts))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            encode_q.put(_SENTINEL)

    with Timer("apply"):
        with decode_mod.VideoSource(source, cfg.hwaccel, threads) as probe_src:
            time_base = probe_src.time_base
        rate = Fraction(info.fps).limit_denominator(90000)

        t_dec = threading.Thread(target=decoder, daemon=True)
        t_proc = threading.Thread(target=processor, daemon=True)
        t_dec.start()
        t_proc.start()

        bar = None
        if progress:
            from tqdm import tqdm

            bar = tqdm(total=table["n"], unit="f", desc="apply", leave=False)

        encoder = VideoEncoder(
            tmp_video, info.width, info.height, rate, time_base,
            encoder=cfg.encoder, crf=cfg.crf, chroma=cfg.chroma,
            bit_depth=cfg.bit_depth, threads=max(1, threads or 1),
            denoise=cfg.denoise, preset=cfg.encoder_preset, trc=trc,
        )
        try:
            while True:
                item = encode_q.get()
                if item is _SENTINEL:
                    break
                array, pts = item
                encoder.write(array, pts)
                written += 1
                if bar:
                    bar.update(1)
        finally:
            encoder.close()
            if bar:
                bar.close()
            t_dec.join(timeout=5)
            t_proc.join(timeout=5)

    if errors:
        tmp_video.unlink(missing_ok=True)
        raise errors[0]

    with Timer("mux"):
        mux_with_source(str(tmp_video), source, output)
    tmp_video.unlink(missing_ok=True)

    log_event("apply.done", frames=written, output=str(output))
    if written != table["n"]:
        logger.warning(
            "wrote %d frames but the analysis covered %d - the source may have "
            "been re-encoded or trimmed since it was analysed",
            written, table["n"],
        )
    # Report the path that actually ran, not the device that was requested:
    # the lookup-table path deliberately stays on the CPU (see FrameProcessor).
    return {"frames": written, "output": output, "device": worker.describe()}
