#!/usr/bin/env python
"""Turn the printed output of an earlier Colab session into a reference file for the P1 feedback.

Sessions do not keep their run directories, but a session's output (the Markdown the notebook prints) contains
every evaluation line of every run. Runs are deterministic on a given GPU and library version, so a later session
can pair new arms with these references instead of training the same arm again.

    python experiments/p1/import_reference.py <session output .md> <model> <revision> <hardware slug> <output .json>
"""
import json
import re
import sys
from pathlib import Path

EVALUATION = re.compile(r"\[(?P<model>[a-z0-9]+)-s(?P<seed>\d+)-(?P<arm>[\w.\-]+)\] eval @ round (?P<round>\d+): accuracy=(?P<accuracy>[\d.]+) task_loss=(?P<loss>[\d.]+)")


def parse(text, model):
    runs = {}
    for match in EVALUATION.finditer(text):
        if match["model"] != model:
            continue
        run = runs.setdefault(match["arm"], {"seed": int(match["seed"]), "curve": {}})
        run["curve"][int(match["round"])] = {"round": int(match["round"]), "accuracy_pct": round(100 * float(match["accuracy"]), 2),
                                             "eval_loss": float(match["loss"])}
    for run in runs.values():
        run["curve"] = [run["curve"][key] for key in sorted(run["curve"])]
        run["final_round"] = run["curve"][-1]["round"]
        run["final_accuracy_pct"], run["final_eval_loss"] = run["curve"][-1]["accuracy_pct"], run["curve"][-1]["eval_loss"]
    return runs


def main(source, model, revision, hardware, output):
    runs = parse(Path(source).read_text(), model)
    record = {"source": source, "model": model, "revision": revision, "hardware": hardware,
              "note": "evaluation lines printed during the session (accuracy to 4 decimals of a fraction, loss to 4 decimals)", "runs": runs}
    Path(output).write_text(json.dumps(record, indent=1) + "\n")
    print(f"{len(runs)} runs: {sorted(runs)}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
