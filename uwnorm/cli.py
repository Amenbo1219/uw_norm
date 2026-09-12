"""Command line interface (section 8).

Subcommands mirror the two-pass design:

    analyze   pass 1 only  -> analysis.json (+ report)
    apply     pass 2 only  -> reads an analysis.json, writes video
    run       both
    report    re-render a report from an existing analysis.json
    verify    re-analyse a written output and check A-1/A-2/A-5
    presets   list the bundled configurations

Every option defaults to ``None`` so that "not given" stays distinguishable
from "given the default value"; that is what lets a YAML preset sit between the
dataclass defaults and the flags the user actually typed (F-09).
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import __version__
from .config import (
    DENOISERS, ENCODER_PRESETS, ENCODERS, EXPOSURE_MODES, SMOOTHERS,
    WB_METHODS, WB_MODES, Config, list_presets, load_config,
)
from .logging_utils import get_logger, log_event, setup_logging

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uwnorm",
        description="Normalise exposure and white balance of underwater video over time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"uwnorm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("-c", "--config", help="YAML file or bundled preset name")
        p.add_argument("-v", "--verbose", action="store_true")
        p.add_argument("--log-jsonl", help="append structured logs to this file")
        p.add_argument("--no-progress", action="store_true")

    def add_analysis_opts(p):
        g = p.add_argument_group("analysis")
        g.add_argument("--input-trc", choices=["auto", "srgb", "bt709", "gamma22", "gamma24", "linear"])
        g.add_argument("--analysis-size", type=int, help="long edge of the analysis decode")
        g.add_argument("--analysis-stride", type=int, help="measure every Nth frame")

        g = p.add_argument_group("exposure (F-03)")
        g.add_argument("--exposure-mode", choices=EXPOSURE_MODES)
        g.add_argument("--exposure-tau", type=float, help="seconds; faster than this is flicker")
        g.add_argument("--max-exposure-gain", type=float, metavar="EV")

        g = p.add_argument_group("white balance (F-04, F-05)")
        g.add_argument("--wb-method", choices=WB_METHODS)
        g.add_argument("--wb-p", type=float, help="Minkowski order for shades-of-gray")
        g.add_argument("--wb-mode", choices=WB_MODES)
        g.add_argument("--wb-tau", type=float, metavar="SECONDS")
        g.add_argument("--max-wb-gain", type=float)
        g.add_argument("--red-compensation", type=float, metavar="ALPHA",
                       help="Ancuti red-channel compensation strength (0 = off)")
        g.add_argument("--reference-frame", metavar="TIME",
                       help="timestamp of the grey-card frame")
        g.add_argument("--reference-roi", metavar="X,Y,W,H",
                       help="normalised ROI within the reference frame")
        g.add_argument("--smoother", choices=SMOOTHERS)

        g = p.add_argument_group("scene cuts (F-06)")
        g.add_argument("--cut-threshold", type=float)
        g.add_argument("--cuts", help="comma separated manual cut timestamps")

        g = p.add_argument_group("region of interest (F-07)")
        g.add_argument("--roi", metavar="SPEC", help="'center:0.6' or 'x,y,w,h'")
        g.add_argument("--exclude-mask", metavar="PNG|X,Y,W,H")
        g.add_argument("--highlight-exclude", type=float)
        g.add_argument("--shadow-exclude", type=float)

    def add_apply_opts(p):
        g = p.add_argument_group("output")
        g.add_argument("--encoder", choices=ENCODERS)
        g.add_argument("--crf", type=int)
        g.add_argument("--encoder-preset", choices=list(ENCODER_PRESETS),
                       help="x264/x265 speed-quality trade-off (mapped to p1-p7 for NVENC)")
        g.add_argument("--chroma", choices=["420", "422", "444"])
        g.add_argument("--bit-depth", type=int, choices=[8, 10])
        g.add_argument("--profile", choices=["archive"], help="ProRes 422 10-bit preset")
        g.add_argument("--denoise", choices=DENOISERS)
        g.add_argument("--soft-clip-knee", type=float)

        g = p.add_argument_group("performance")
        g.add_argument("-j", "--jobs", type=int, help="worker/encoder threads (0 = all cores)")
        g.add_argument("--device", choices=["auto", "cpu", "cuda"],
                       help="where the per-pixel maths runs")
        g.add_argument("--hwaccel", choices=["none", "cuda"], help="NVDEC decoding")

        g = p.add_argument_group("preview (F-08)")
        g.add_argument("--preview-range", metavar="START-END")
        g.add_argument("--compare", action="store_true",
                       help="also render a side-by-side Before/After video")

    # analyze -------------------------------------------------------------
    p = sub.add_parser("analyze", help="pass 1: measure and build the gain curves")
    p.add_argument("input")
    p.add_argument("-a", "--analysis", help="output analysis.json (default: <input>.analysis.json)")
    p.add_argument("--report", help="report path (.html or .png)")
    p.add_argument("--preview-range", metavar="START-END")
    add_analysis_opts(p)
    add_common(p)

    # apply ---------------------------------------------------------------
    p = sub.add_parser("apply", help="pass 2: apply an existing analysis.json")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-a", "--analysis", help="analysis.json (default: <input>.analysis.json)")
    p.add_argument("--report", help="report path (.html or .png)")
    p.add_argument("--verify", action="store_true", help="re-analyse the output and check A-1/A-2")
    add_apply_opts(p)
    add_common(p)

    # run -----------------------------------------------------------------
    p = sub.add_parser("run", help="analyze + apply")
    p.add_argument("input", help="video file, or a directory for batch mode (N-07)")
    p.add_argument("-o", "--output", help="output file, or directory in batch mode")
    p.add_argument("-a", "--analysis", help="analysis.json path")
    p.add_argument("--report", help="report path (.html or .png)")
    p.add_argument("--dry-run", action="store_true", help="analyse and report only (F-08)")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--parallel", type=int, default=1,
                   help="files processed concurrently in batch mode")
    p.add_argument("--cache-dir", help="reuse analysis.json from here (N-04)")
    add_analysis_opts(p)
    add_apply_opts(p)
    add_common(p)

    # report --------------------------------------------------------------
    p = sub.add_parser("report", help="re-render a report from an analysis.json")
    p.add_argument("analysis")
    p.add_argument("-o", "--output", required=True, help="report path (.html or .png)")
    add_common(p)

    # verify --------------------------------------------------------------
    p = sub.add_parser("verify", help="check a written output against A-1/A-2/A-5")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("-a", "--analysis", help="analysis.json of the input")
    add_common(p)

    sub.add_parser("presets", help="list the bundled presets")
    return parser


_CONFIG_KEYS = set(Config().to_dict())


def config_from_args(args: argparse.Namespace) -> Config:
    """Layer dataclass defaults < preset/YAML < explicit flags (F-09)."""
    base = Config()
    preset = load_config(getattr(args, "config", None))
    cfg = base.merged(preset)
    overrides = {k: v for k, v in vars(args).items() if k in _CONFIG_KEYS and v is not None}
    if isinstance(overrides.get("cuts"), str):
        overrides["cuts"] = [c for c in overrides["cuts"].split(",") if c.strip()]
    if overrides.get("compare") is False:
        overrides.pop("compare")
    if overrides.get("dry_run") is False:
        overrides.pop("dry_run")
    cfg = cfg.merged(overrides)
    if cfg.jobs == 0:
        cfg = cfg.merged({"jobs": os.cpu_count() or 4})
    return cfg.resolve_profile()


def default_analysis_path(input_path: str, explicit: str | None, cache_dir: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    name = Path(input_path).with_suffix(".analysis.json").name
    if cache_dir:
        return Path(cache_dir) / name
    return Path(input_path).with_suffix(".analysis.json")


def _maybe_report(doc, report_path, verification=None) -> None:
    if not report_path:
        return
    from .report import write_report

    write_report(doc, report_path, verification)


def cmd_analyze(args) -> int:
    from .analyze import analyze, save_analysis

    cfg = config_from_args(args)
    doc = analyze(args.input, cfg, progress=not args.no_progress)
    out = default_analysis_path(args.input, args.analysis)
    save_analysis(doc, out)
    _maybe_report(doc, args.report)
    _print_summary(doc)
    return 0


def cmd_apply(args) -> int:
    from .analyze import load_analysis
    from .apply import apply_gains

    doc = load_analysis(default_analysis_path(args.input, args.analysis))
    # The analysis pass owns the measurement settings, so they are read back
    # from the document rather than re-derived: that is what keeps a
    # hand-edited gain curve meaningful.  Output-side settings are taken from
    # this invocation, but only where the user actually asked for them - so
    # `analyze --config archive` followed by a bare `apply` still writes ProRes.
    explicit = {k: v for k, v in vars(args).items() if k in _APPLY_SIDE_KEYS and v is not None}
    if explicit.get("compare") is False:
        explicit.pop("compare")
    preset = {k: v for k, v in load_config(args.config).items() if k in _APPLY_SIDE_KEYS}
    stored = dict(doc["config"])
    # A stored `--profile archive` expands to concrete encoder settings; if this
    # invocation names any of them explicitly, the shorthand has been overruled
    # and must not expand again on top of the flags the user just typed.
    if _PROFILE_KEYS & set(explicit):
        stored["profile"] = None
    cfg = Config(**stored).merged({**preset, **explicit})
    if cfg.jobs == 0:
        cfg = cfg.merged({"jobs": os.cpu_count() or 4})
    cfg = cfg.resolve_profile()
    result = apply_gains(args.input, args.output, doc, cfg, progress=not args.no_progress)
    verification = _post_apply(args, doc, cfg, args.input, args.output)
    _maybe_report(doc, args.report, verification)
    print(f"wrote {result['output']} ({result['frames']} frames, {result['device']})")
    return 0


#: the concrete settings ``--profile`` is shorthand for
_PROFILE_KEYS = {"encoder", "chroma", "bit_depth"}

#: settings that may be changed between analyse and apply without invalidating
#: the curves - everything else is a measurement decision baked into them.
_APPLY_SIDE_KEYS = {
    "encoder", "encoder_preset", "crf", "chroma", "bit_depth", "profile",
    "denoise", "soft_clip_knee", "jobs", "device", "hwaccel", "compare",
    # accepted so that `apply --preview-range` is not silently ignored; a range
    # that disagrees with the analysis is rejected in apply_gains().
    "preview_range",
}


def _post_apply(args, doc, cfg, source, output):
    verification = None
    if getattr(args, "verify", False):
        from .verify import verify_output

        verification = verify_output(doc, output, cfg)
        _print_verification(verification)
    if cfg.compare:
        from .io.encode import make_comparison

        cmp_path = str(Path(output).with_suffix(".compare" + Path(output).suffix))
        make_comparison(source, output, cmp_path)
        print(f"wrote {cmp_path}")
    return verification


def _run_one(path: str, output: str | None, args, cfg: Config) -> dict:
    from .analyze import analyze, load_analysis, save_analysis
    from .apply import apply_gains

    analysis_path = default_analysis_path(path, args.analysis, args.cache_dir)
    doc = None
    if analysis_path.exists() and args.cache_dir:
        # N-04: a run interrupted after the analysis pass resumes from here.
        try:
            doc = load_analysis(analysis_path)
            log_event("analysis.cached", path=str(analysis_path))
        except Exception:
            doc = None
    if doc is None:
        doc = analyze(path, cfg, progress=not args.no_progress)
        save_analysis(doc, analysis_path)

    verification = None
    if not cfg.dry_run:
        if not output:
            raise SystemExit("run: -o/--output is required unless --dry-run is given")
        apply_gains(path, output, doc, cfg, progress=not args.no_progress)
        verification = _post_apply(args, doc, cfg, path, output)
    _maybe_report(doc, args.report, verification)
    return {"doc": doc, "verification": verification, "output": output}


def cmd_run(args) -> int:
    cfg = config_from_args(args)
    in_path = Path(args.input)

    if in_path.is_dir():  # N-07 batch mode
        files = sorted(p for p in in_path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
        if not files:
            raise SystemExit(f"{in_path}: no video files found")
        out_dir = Path(args.output) if args.output else None
        if out_dir and not cfg.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        log_event("batch.start", files=len(files), parallel=args.parallel)

        def job(p: Path):
            out = str(out_dir / p.name) if out_dir else None
            # Reports are per-file in batch mode, next to the output.
            per_file = argparse.Namespace(**vars(args))
            if args.report and out_dir:
                per_file.report = str(out_dir / (p.stem + Path(args.report).suffix))
            per_file.analysis = None
            return _run_one(str(p), out, per_file, cfg)

        failures = 0
        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {pool.submit(job, p): p for p in files}
                for fut, p in futures.items():
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        failures += 1
                        get_logger().error("%s: %s", p.name, exc)
        else:
            for p in files:
                try:
                    job(p)
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    get_logger().error("%s: %s", p.name, exc)
        print(f"batch complete: {len(files) - failures}/{len(files)} succeeded")
        return 1 if failures else 0

    result = _run_one(str(in_path), args.output, args, cfg)
    _print_summary(result["doc"])
    if result["output"]:
        print(f"wrote {result['output']}")
    return 0


def cmd_report(args) -> int:
    from .analyze import load_analysis
    from .report import write_report

    doc = load_analysis(args.analysis)
    path = write_report(doc, args.output)
    print(f"wrote {path}")
    return 0


def cmd_verify(args) -> int:
    from .analyze import load_analysis
    from .verify import verify_output

    doc = load_analysis(default_analysis_path(args.input, args.analysis))
    cfg = Config(**doc["config"])
    verification = verify_output(doc, args.output, cfg)
    _print_verification(verification)
    ok = (
        verification["exposure"]["reduction_pct"] >= 80.0
        and verification["wb_r"]["reduction_pct"] >= 80.0
        and verification["wb_b"]["reduction_pct"] >= 80.0
        and verification["streams"]["ok"]
    )
    return 0 if ok else 2


def cmd_presets(args) -> int:
    import yaml as _yaml

    from .config import PRESET_DIR

    for name in list_presets():
        with open(PRESET_DIR / f"{name}.yaml", encoding="utf-8") as fh:
            raw = _yaml.safe_load(fh) or {}
        print(f"{name:20s} {raw.get('description', '')}")
    return 0


def _print_summary(doc: dict) -> None:
    m = doc["metrics"]
    print(
        f"analysed {doc['analysis']['frames']} frames, "
        f"{len(doc['cuts']['indices'])} cut(s)\n"
        f"  exposure high-frequency variation: "
        f"{m['exposure']['hf_std_before']:.4f} -> {m['exposure']['hf_std_after']:.4f} "
        f"({m['exposure']['reduction_pct']:.1f}% reduction)\n"
        f"  WB R/G: {m['wb_r']['reduction_pct']:.1f}%   "
        f"WB B/G: {m['wb_b']['reduction_pct']:.1f}%   (predicted)"
    )


def _print_verification(v: dict) -> None:
    def line(name, d):
        mark = "PASS" if d["reduction_pct"] >= 80.0 else "FAIL"
        return (f"  [{mark}] {name:12s} {d['hf_std_before']:.5f} -> {d['hf_std_after']:.5f} "
                f"({d['reduction_pct']:.1f}%)")

    print("measured on the written output:")
    print(line("exposure", v["exposure"]))
    print(line("WB R/G", v["wb_r"]))
    print(line("WB B/G", v["wb_b"]))
    s = v["streams"]
    print(f"  [{'PASS' if s['ok'] else 'FAIL'}] streams      "
          + ", ".join(f"{k}={'ok' if val else 'FAILED'}"
                      for k, val in s.items() if k != "ok"))


COMMANDS = {
    "analyze": cmd_analyze,
    "apply": cmd_apply,
    "run": cmd_run,
    "report": cmd_report,
    "verify": cmd_verify,
    "presets": cmd_presets,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "verbose", False), getattr(args, "log_jsonl", None))
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        get_logger().error("interrupted")
        return 130
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        get_logger().error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
