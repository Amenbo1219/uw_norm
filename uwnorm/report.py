"""Visualisation of the analysis (3.2, F-08).

Produces a self-contained HTML page (PNG embedded as a data URI, so it can be
mailed or archived as one file) plus the PNG alongside it.  The plots are the
evidence for A-1/A-2 and the first place to look when a clip comes out wrong:
a gain curve pinned to the clamp, or a cut that was missed, is obvious here and
invisible in the video.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np

from . import __version__
from .core import exposure as exposure_mod
from .logging_utils import log_event


def _figure(doc: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = doc["frames"]
    t = np.asarray(f["time"], dtype=float)
    cut_times = doc["cuts"]["times"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    fig.suptitle(
        f"uwnorm {__version__} - {doc['source']['name']}  "
        f"({doc['source']['width']}x{doc['source']['height']}, "
        f"{doc['source']['fps']:.3f} fps, TRC {doc['source']['resolved_trc']})",
        fontsize=12,
    )

    def mark_cuts(ax):
        for ct in cut_times:
            ax.axvline(ct, color="0.5", linestyle=":", linewidth=1, zorder=0)

    # --- 1. exposure, in stops relative to the clip's own median -----------
    curve = np.asarray(f["exposure_curve"], dtype=float) / exposure_mod.LN2
    gain = np.asarray(f["exposure_log_gain"], dtype=float) / exposure_mod.LN2
    ref = float(np.median(curve))
    ax = axes[0]
    ax.plot(t, curve - ref, color="#c0392b", linewidth=0.9, label="before")
    ax.plot(t, curve + gain - ref, color="#27ae60", linewidth=1.4, label="after")
    ax.set_ylabel("exposure [stops]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    mark_cuts(ax)
    m = doc["metrics"]["exposure"]
    ax.set_title(
        f"Exposure - mode={doc['config']['exposure_mode']}, tau={doc['config']['exposure_tau']}s"
        f"   |   high-frequency sigma {m['hf_std_before']:.4f} -> {m['hf_std_after']:.4f}"
        f"  ({m['reduction_pct']:.1f}% reduction)",
        fontsize=9, loc="left",
    )

    # --- 2. applied exposure gain and its clamp ---------------------------
    ax = axes[1]
    ax.plot(t, gain, color="#2980b9", linewidth=1.0)
    limit = doc["config"]["max_exposure_gain"]
    ax.axhline(limit, color="#e67e22", linestyle="--", linewidth=0.8)
    ax.axhline(-limit, color="#e67e22", linestyle="--", linewidth=0.8, label="clamp")
    clamped = np.asarray(f["exposure_clamped"], dtype=bool)
    if clamped.any():
        ax.plot(t[clamped], gain[clamped], "r.", markersize=3, label="clamped")
    ax.set_ylabel("applied gain [stops]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    mark_cuts(ax)

    # --- 3. white balance -------------------------------------------------
    ax = axes[2]
    for rawkey, gkey, colour, label in (
        ("wb_raw_r", "wb_gain_r", "#c0392b", "R/G"),
        ("wb_raw_b", "wb_gain_b", "#2980b9", "B/G"),
    ):
        # The raw per-frame estimate is what a naive frame-by-frame white
        # balance would apply; the applied curve is what the ratio-anchored
        # estimate produces.  The gap between them is the estimator noise the
        # pipeline rejects.
        raw = np.asarray(f[rawkey], dtype=float)
        applied = np.asarray(f[gkey], dtype=float)
        ax.plot(t, raw, color=colour, linewidth=0.7, alpha=0.40,
                label=f"{label} per-frame estimate")
        ax.plot(t, applied, color=colour, linewidth=1.5, label=f"{label} applied")
    ax.axhline(doc["config"]["max_wb_gain"], color="#e67e22", linestyle="--", linewidth=0.8)
    ax.set_ylabel("WB gain (G=1)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    mark_cuts(ax)
    mr, mb = doc["metrics"]["wb_r"], doc["metrics"]["wb_b"]
    ax.set_title(
        f"White balance - method={doc['config']['wb_method']}, "
        f"mode={doc['config']['wb_mode']}, tau={doc['config']['wb_tau']}s"
        f"   |   HF reduction R {mr['reduction_pct']:.1f}% / B {mb['reduction_pct']:.1f}%",
        fontsize=9, loc="left",
    )

    # --- 4. cut detection --------------------------------------------------
    ax = axes[3]
    ax.plot(t, np.asarray(f["cut_score"], dtype=float), color="#8e44ad", linewidth=0.8)
    ax.axhline(doc["cuts"]["threshold"], color="#e67e22", linestyle="--",
               linewidth=0.9, label="--cut-threshold")
    ax.set_ylabel("cut score")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    mark_cuts(ax)
    ax.set_title(
        f"Scene cuts - {len(doc['cuts']['indices'])} detected "
        f"({len(doc['cuts']['auto'])} automatic, {len(doc['cuts']['manual'])} manual)",
        fontsize=9, loc="left",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def write_png(doc: dict, path: str | Path) -> Path:
    fig = _figure(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _png_bytes(doc: dict) -> bytes:
    fig = _figure(doc)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf.getvalue()


_ROW = "<tr><th>{}</th><td>{}</td></tr>"


def _table(rows) -> str:
    return "<table>" + "".join(_ROW.format(html.escape(str(k)), html.escape(str(v)))
                               for k, v in rows) + "</table>"


def write_html(doc: dict, path: str | Path, verification: dict | None = None) -> Path:
    """Self-contained HTML report."""
    png = base64.b64encode(_png_bytes(doc)).decode("ascii")
    src, cfg, met = doc["source"], doc["config"], doc["metrics"]

    source_rows = [
        ("File", src["name"]),
        ("Resolution", f"{src['width']}x{src['height']}"),
        ("Frame rate", f"{src['fps']:.3f} fps" + (" (VFR)" if src["is_vfr"] else "")),
        ("Duration", f"{src['duration']:.2f} s" if src.get("duration") else "-"),
        ("Codec / pixel format", f"{src['codec']} / {src['pix_fmt']}"),
        ("Transfer characteristic", f"{src['resolved_trc']} (tagged: {src['color_trc'] or 'none'})"),
        ("Audio / data streams", f"{'yes' if src['has_audio'] else 'no'} / "
                                 f"{'yes' if src['has_data'] else 'no'}"),
        ("Analysed", f"{doc['analysis']['frames']} frames at "
                     f"{doc['analysis']['width']}x{doc['analysis']['height']}, "
                     f"stride {doc['analysis']['stride']}"),
    ]
    settings_rows = [
        ("Exposure", f"mode={cfg['exposure_mode']}, tau={cfg['exposure_tau']}s, "
                     f"max={cfg['max_exposure_gain']} EV"),
        ("White balance", f"method={cfg['wb_method']} (p={cfg['wb_p']}), mode={cfg['wb_mode']}, "
                          f"tau={cfg['wb_tau']}s, max={cfg['max_wb_gain']}x"),
        ("Red compensation", cfg["red_compensation"] or "off"),
        ("ROI / exclusion", f"{cfg['roi'] or 'full frame'} / {cfg['exclude_mask'] or 'none'}"),
        ("Cut threshold", cfg["cut_threshold"]),
        ("Soft-clip knee", cfg["soft_clip_knee"]),
        ("Encoder", f"{cfg['encoder']} crf={cfg['crf']} {cfg['chroma']} {cfg['bit_depth']}-bit"
                    + (f", denoise={cfg['denoise']}" if cfg["denoise"] != "off" else "")),
    ]

    def metric_block(title, data, target=80.0):
        rows = "".join(
            f"<tr><th>{html.escape(name)}</th>"
            f"<td>{d['hf_std_before']:.5f}</td><td>{d['hf_std_after']:.5f}</td>"
            f"<td class='{'pass' if d['reduction_pct'] >= target else 'fail'}'>"
            f"{d['reduction_pct']:.1f}%</td></tr>"
            for name, d in data
        )
        return (
            f"<h3>{html.escape(title)}</h3><table class='metrics'>"
            f"<tr><th></th><th>HF sigma before</th><th>HF sigma after</th>"
            f"<th>reduction (target &ge;{target:.0f}%)</th></tr>{rows}</table>"
        )

    predicted = metric_block(
        "Predicted from the gain curves",
        [("Exposure (A-1)", met["exposure"]), ("WB R/G (A-2)", met["wb_r"]),
         ("WB B/G (A-2)", met["wb_b"])],
    )
    measured = ""
    if verification:
        measured = metric_block(
            "Measured on the written output",
            [("Exposure (A-1)", verification["exposure"]),
             ("WB R/G (A-2)", verification["wb_r"]),
             ("WB B/G (A-2)", verification["wb_b"])],
        )
        streams = verification["streams"]
        measured += _table([(k.replace("_", " ").capitalize(),
                             "OK" if v else "FAILED") for k, v in streams.items()])

    applied = met["applied"]
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>uwnorm report - {html.escape(src['name'])}</title>
<style>
 body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 2rem auto;
        max-width: 1180px; padding: 0 1rem; color: #1c2023; background: #fbfbfc; }}
 h1 {{ font-size: 1.4rem; margin-bottom: .2rem; }}
 h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #dde; padding-bottom: .3rem; }}
 h3 {{ font-size: .95rem; margin: 1.2rem 0 .4rem; }}
 table {{ border-collapse: collapse; margin: .4rem 0 1rem; font-size: .88rem; }}
 th, td {{ text-align: left; padding: .32rem .8rem .32rem 0; vertical-align: top; }}
 th {{ color: #555; font-weight: 600; white-space: nowrap; }}
 table.metrics td, table.metrics th {{ border-bottom: 1px solid #e6e6ea; padding-right: 1.6rem; }}
 .pass {{ color: #1e8449; font-weight: 700; }}
 .fail {{ color: #c0392b; font-weight: 700; }}
 img {{ width: 100%; border: 1px solid #dde; border-radius: 6px; background: #fff; }}
 .cols {{ display: flex; gap: 3rem; flex-wrap: wrap; }}
 footer {{ margin-top: 2.5rem; color: #777; font-size: .8rem; }}
</style></head><body>
<h1>uwnorm analysis report</h1>
<p>{html.escape(src['name'])} &mdash; {doc['analysis']['frames']} analysed frames,
   {len(doc['cuts']['indices'])} scene cut(s)</p>
<h2>Acceptance metrics</h2>
{predicted}
{measured}
<p>Applied gains: exposure {applied['exposure_ev_min']:+.2f} to
   {applied['exposure_ev_max']:+.2f} EV, WB R/G
   {applied['wb_gain_r_range'][0]:.3f}&ndash;{applied['wb_gain_r_range'][1]:.3f},
   B/G {applied['wb_gain_b_range'][0]:.3f}&ndash;{applied['wb_gain_b_range'][1]:.3f}.</p>
<h2>Curves</h2>
<img src="data:image/png;base64,{png}" alt="analysis plots">
<h2>Source and settings</h2>
<div class="cols"><div>{_table(source_rows)}</div><div>{_table(settings_rows)}</div></div>
<footer>Generated by uwnorm {__version__}.</footer>
</body></html>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    log_event("report.written", path=str(path), kb=round(path.stat().st_size / 1024, 1))
    return path


def write_report(doc: dict, path: str | Path, verification: dict | None = None) -> Path:
    """Write HTML or PNG depending on the requested extension."""
    path = Path(path)
    if path.suffix.lower() == ".png":
        out = write_png(doc, path)
        log_event("report.written", path=str(out))
        return out
    return write_html(doc, path, verification)
