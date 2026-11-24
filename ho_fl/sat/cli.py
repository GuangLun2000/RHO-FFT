"""Command-line entry points of the satellite scenario (brief section 8.2).

    ho-fl-sat-trace      build and check the trace of a scenario
    ho-fl-sat-train      one training run
    ho-fl-sat-tune       server step size per method                 (milestone P-S4)
    ho-fl-sat-topology   topology computation for S1 and S2          (milestone P-S7)
    ho-fl-sat-figures    tables and figures from the run directories (milestone P-S7)

Each command is also reachable as ``python -m ho_fl.sat.cli <command> ...``, which is how the notebook calls them
(the package is not installed there).
"""
import sys


def _later(command, milestone):
    def main(argv=None):
        raise SystemExit(f"{command} belongs to milestone {milestone} and is not implemented yet")
    return main


trace_main = _later("ho-fl-sat-trace", "P-S1")
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
