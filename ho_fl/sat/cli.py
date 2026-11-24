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


def model_tables(trace, slots_per_round=20, round_s=600, deadline_s=480.0):
    """The link, timing and energy numbers of the system model as Markdown tables (appendix A of the brief),
    computed from a trace and the nominal sizes of the two models; no model is loaded."""
    import numpy as np

    from .config import CLASS_SPECS, NOMINAL_PARAMETERS, TASKS, SatConfig
    from .energy import Batteries, compute_energy_j, compute_time_s
    from .links import GROUND_LINK, LinkBudget, hop_delay_s, inter_satellite_radio
    from .network import timeline

    def size(task, cut):
        block, head = NOMINAL_PARAMETERS[task]
        return 4 * ((TASKS[task].depth - cut) * block + head)

    lines, roles = [], trace.roles
    in_plane = float(trace.edge_dist_km[0, 0])
    gains = (20, 17, 15, 13, 12, 11, 10, 9, 8)
    radio = inter_satellite_radio(20.0)
    lines += ["Radio rate between satellites at 20 dBi: " + ", ".join(f"{km:.0f} km {radio.rate_bps(km) / 1e6:.2f} Mbps" for km in (2505, 3803, in_plane, 5000)),
              "Ground link rate: " + ", ".join(f"{km} km {GROUND_LINK.rate_bps(km) / 1e9:.3f} Gbps" for km in (550, 1000, 2000)), ""]
    rate = radio.rate_bps(in_plane)
    lines += ["| payload | bytes | in-plane hop at 20 dBi (s) |", "| --- | --- | --- |"]
    for task in ("t1", "t2"):
        for cut in (0, *TASKS[task].ladder):
            if size(task, cut):
                lines.append(f"| {task} first-order update, cut {cut} | {size(task, cut)} | {hop_delay_s(size(task, cut), rate, in_plane):.3f} |")
    lines += [f"| zeroth-order report | 58 | {hop_delay_s(58, rate, in_plane):.4f} (11 hops: {11 * hop_delay_s(58, rate, in_plane):.3f}) |", ""]
    # Deadline and the latest arrival of a hybrid satellite four hops from its root, by antenna gain.
    view = trace.view(0, slots_per_round)
    lines += ["| task | gain (dBi) | in-plane rate (Mbps) | ring (s) | broadcast (s) | deadline (s) | hybrid satellite, 4 hops, idle cut: arrival in sun / in eclipse (s) | next cut: sun / eclipse (s) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    slowdown = {kind: spec.slowdown for kind, spec in CLASS_SPECS.items()}
    for task in ("t1", "t2"):
        spec = TASKS[task]
        cfg = SatConfig(task=task, round_s=round_s, deadline_s=deadline_s)
        for gain in gains:
            budget = LinkBudget(trace, view, rf_gain_dbi=gain)
            times = timeline(cfg, budget, trace, size(task, 0))
            cells = []
            for cut in (spec.cut_idle, spec.cut_imaging):
                arrivals = [compute_time_s("H", lit, cut, spec.depth, 5, 2, True, spec.t_fwd, slowdown)
                            + budget.path(roles.path_to_root(4), size(task, cut)) for lit in (True, False)]
                cells.append(" / ".join(f"{value:.1f}{' LATE' if value > times.deadline_s else ''}" for value in arrivals))
            lines.append(f"| {task} | {gain} | {inter_satellite_radio(gain).rate_bps(in_plane) / 1e6:.3f} | {times.ring_s:.3f} | "
                         f"{times.broadcast_s:.2f} | {times.deadline_s:.2f} | {cells[0]} | {cells[1]} |")
    lines += ["", "| task | class | cut | zeroth-order | compute time in sun / eclipse (s) | energy in sun (Wh) |", "| --- | --- | --- | --- | --- | --- |"]
    for task in ("t1", "t2"):
        spec = TASKS[task]
        for kind, cut, zo in (("B", 0, True), ("H", spec.cut_idle, True), ("H", spec.cut_imaging, True), ("Z", spec.depth, True),
                              ("H", spec.cut_idle, False), ("H", spec.cut_imaging, False)):
            sun, dark = (compute_time_s(kind, lit, cut, spec.depth, 5, 2, zo, spec.t_fwd, slowdown) for lit in (True, False))
            lines.append(f"| {task} | {kind} | {cut} | {'on' if zo else 'off'} | {sun:.2f} / {'no power' if dark is None else f'{dark:.2f}'} | "
                         f"{compute_energy_j(kind, True, sun) / 3600:.4f} |")
    # Lowest battery charge over the trace: the floor set by the eclipses, and with as much training as possible
    # (every satellite trains in every round it has power for, at the cost of a participation of the text task).
    spec, lines = TASKS["t1"], lines + [""]
    for label, training in (("without any training", False), ("with training in every round a satellite has power for", True)):
        batteries, lowest = Batteries(roles), np.ones(roles.count)
        for r in range(trace.slots // slots_per_round):
            round_view, spent = trace.view(r, slots_per_round), np.zeros(roles.count)
            for sid in (roles.members("H") + roles.members("Z") if training else ()):
                kind, lit = roles.kind(sid), bool(round_view.sunlit[sid])
                seconds = compute_time_s(kind, lit, spec.depth, spec.depth, 5, 2, True, spec.t_fwd, slowdown)
                if seconds is not None:
                    spent[sid] = compute_energy_j(kind, lit, seconds)
            batteries.update(round_view.sun_seconds, round_s, spent)
            lowest = np.minimum(lowest, batteries.level / np.maximum(batteries.capacity, 1.0))
        lines.append(f"Lowest battery charge {label}: hybrid satellites {lowest[roles.members('H')].min():.3f}, "
                     f"forward-only satellites {lowest[roles.members('Z')].min():.3f} of the capacity.")
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--tables", action="store_true", help="also print the link, timing and energy tables of the model")
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
    if args.tables and args.scenario == "s1":
        print("\nLink, timing and energy numbers of the model (nominal update sizes of Qwen2.5-1.5B and ViT-B/16):\n")
        print(model_tables(trace), flush=True)
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


def train_main(argv=None):
    """One training run of the satellite scenario; the options are the fields of ``SatConfig``."""
    from .config import parse_sat_config
    from .engine import run

    run(parse_sat_config(argv))
    return 0


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
