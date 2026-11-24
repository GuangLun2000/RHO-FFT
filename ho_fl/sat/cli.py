"""Command-line entry points of the satellite scenario (brief section 8.2).

    ho-fl-sat-trace      build and check the trace of a scenario
    ho-fl-sat-train      one training run
    ho-fl-sat-tune       server step size per method                 (milestone P-S4)
    ho-fl-sat-topology   topology computation for S1 and S2          (milestone P-S7)
    ho-fl-sat-figures    tables and figures from the run directories (milestone P-S7)

Each command is also reachable as ``python -m ho_fl.sat.cli <command> ...``, which is how the notebook calls them
(the package is not installed there).
"""
import argparse
import sys
from pathlib import Path


def _later(command, milestone):
    def main(argv=None):
        raise SystemExit(f"{command} belongs to milestone {milestone} and is not implemented yet")
    return main


def trace_main(argv=None):
    """Build the trace of a scenario, write it with its statistics and compare those of S1 with appendix A."""
    from .config import SCENARIOS
    from .scenario import Trace, build_trace, check_against_appendix, load_trace, sanity_statistics, write_trace

    parser = argparse.ArgumentParser(prog="ho-fl-sat-trace", description=trace_main.__doc__)
    parser.add_argument("--scenario", default="s1", choices=sorted(SCENARIOS))
    parser.add_argument("--output", default="", help="trace directory; default <cache-dir>/sat/traces/<scenario>")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--slot-s", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42, help="only enters the substitute for the land mask")
    parser.add_argument("--land", default="auto", choices=["auto", "mask", "duty_cycle"])
    parser.add_argument("--force", action="store_true", help="rebuild a trace that already exists")
    args = parser.parse_args(argv)
    out = Path(args.output) if args.output else Path(args.cache_dir) / "sat" / "traces" / args.scenario
    if (out / "meta.json").exists() and not args.force:
        trace = load_trace(out)
        print(f"trace exists at {out} (hashes verified); --force rebuilds it", flush=True)
    else:
        arrays, meta = build_trace(args.scenario, args.hours, args.slot_s, args.seed, args.land)
        trace = Trace(arrays, meta)
        write_trace(out, arrays, meta, sanity_statistics(trace))
        print(f"trace written to {out}", flush=True)
    stats = sanity_statistics(trace)
    meta = trace.meta
    print(f"scenario {meta['scenario']}: {meta['satellites']} satellites, {meta['slots']} slots of {meta['slot_s']} s, "
          f"land from {meta['land_source']}, trace sha256 {meta['trace_sha256']}", flush=True)
    if meta["land_source"] != "global_land_mask":
        print("NOTE: the land mask package is not installed, so imaging uses the 30% duty-cycle substitute "
              "(install the optional group 'sat'); the imaging share and the trace hash differ from a trace built with the mask.")
    reference = args.scenario == "s1" and args.hours == 24.0 and args.slot_s == 30
    if not reference:
        print("no reference values for this scenario and duration; statistics are in sanity.json")
        return 0
    rows = check_against_appendix(stats)
    print("\n| statistic | trace | appendix A | tolerance | ok |\n| --- | --- | --- | --- | --- |")
    for name, value, expected, tolerance, ok in rows:
        print(f"| {name} | {value:.4f} | {expected:g} | {tolerance:g} | {'yes' if ok else 'NO'} |")
    failed = [name for name, *_, ok in rows if not ok]
    print(f"\n{len(rows) - len(failed)} of {len(rows)} statistics agree with appendix A" + (f"; differing: {failed}" if failed else ""), flush=True)
    return 1 if failed else 0


train_main = _later("ho-fl-sat-train", "P-S3")
tune_main = _later("ho-fl-sat-tune", "P-S4")
topology_main = _later("ho-fl-sat-topology", "P-S7")
figures_main = _later("ho-fl-sat-figures", "P-S7")

COMMANDS = {"trace": "trace_main", "train": "train_main", "tune": "tune_main", "topology": "topology_main",
            "figures": "figures_main"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit(f"usage: python -m ho_fl.sat.cli {{{','.join(COMMANDS)}}} [options]")
    return globals()[COMMANDS[argv[0]]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
