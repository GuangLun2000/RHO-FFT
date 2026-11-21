#!/usr/bin/env python
"""Figures of the P1 feedback: test curves with a legend that says what every curve is.

Reads feedback_P1.json (written by `p1.py feedback`) and writes p1_curves.png (the ablation ladder) and, when
control arms were run, p1_controls.png. Needs matplotlib only, so it runs in the notebook's own Python.

    python experiments/p1/plot_curves.py <feedback_P1.json> <output directory>

Encoding: hue = what the server does with the reports; solid = uniform sampling, dashed = biased sampling;
hollow markers without a line = the same arm with the zeroth-order updates switched off. With several seeds the
line is the mean over seeds and the individual seeds are drawn thin behind it.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8983"
SOLID, DASHED, DOTTED, DASHDOT = "-", (0, (5, 2.2)), (0, (1.2, 1.6)), (0, (6, 1.5, 1.2, 1.5))


def arm(color, text, ls="none", marker=None, filled=False):
    return dict(color=color, text=text, ls=ls, marker=marker, filled=filled)


LADDER = [
    ("Reference methods", None),
    ("fedavg", arm(INK_2, "FedAvg: every layer first-order (upper bound)", DASHDOT)),
    ("decomfl", arm(GRAY, "DeComFL: every layer zeroth-order (lower bound)", DOTTED)),
    ("HO-FL as published: equal-weight mean of all reports", None),
    ("A-b0", arm(BLUE, "uniform sampling (beta = 0)", SOLID)),
    ("A-b0.75", arm(BLUE, "biased sampling (beta = 0.75, the default)", DASHED)),
    ("B-b0", arm(BLUE, "uniform sampling, zeroth-order updates off", marker="o")),
    ("B-b0.75", arm(BLUE, "biased sampling, zeroth-order updates off", marker="s")),
    ("Each layer averages its first-order reports only (fo_mean)", None),
    ("C-b0", arm(ORANGE, "uniform sampling (beta = 0)", SOLID)),
    ("C-b0.75", arm(ORANGE, "biased sampling (beta = 0.75)", DASHED)),
    ("D-b0", arm(ORANGE, "uniform sampling, zeroth-order updates off", marker="o")),
    ("D-b0.75", arm(ORANGE, "biased sampling, zeroth-order updates off", marker="s")),
    ("Step-size control: published method, larger server step", None),
    ("Ax2-b0", arm(AQUA, "uniform sampling, server step x2", SOLID, "^", True)),
    ("Ax3-b0", arm(AQUA, "uniform sampling, server step x3", SOLID, "D", True)),
]
CONTROLS = [
    ("Server step x1 (lines) against x2 (lines with triangles)", None),
    ("A-b0", arm(BLUE, "published method, uniform sampling", SOLID)),
    ("Ax2-b0", arm(BLUE, "published method, uniform sampling, step x2", SOLID, "^", True)),
    ("C-b0", arm(ORANGE, "fo_mean, uniform sampling", SOLID)),
    ("Cx2-b0", arm(ORANGE, "fo_mean, uniform sampling, step x2", SOLID, "^", True)),
    ("A-b0.75", arm(BLUE, "published method, biased sampling", DASHED)),
    ("Ax2-b0.75", arm(BLUE, "published method, biased sampling, step x2", DASHED, "^", True)),
    ("fedavg", arm(INK_2, "FedAvg", DASHDOT)),
    ("fedavg-x2", arm(INK_2, "FedAvg, step x2", DASHDOT, "^", True)),
    ("More zeroth-order directions", None),
    ("A-b0-q8", arm(AQUA, "published method, uniform sampling, q = 8", SOLID, "x", True)),
    ("Three times as many rounds (right panel)", None),
    ("A-b0-r480", arm(BLUE, "published method, uniform sampling", SOLID, "o", True)),
    ("C-b0-r480", arm(ORANGE, "fo_mean, uniform sampling", SOLID, "o", True)),
    ("Ax3-b0-r480", arm(AQUA, "published method, server step x3", SOLID, "D", True)),
]
LONG = ("A-b0-r480", "C-b0-r480", "Ax3-b0-r480")
ZO_CHECK = [
    ("HO-FL default (beta = 0.75), heterogeneous boundaries", None),
    ("A-b0.75", arm(BLUE, "zeroth-order updates on (published HO-FL)", SOLID)),
    ("B-b0.75", arm(BLUE, "bottom frozen: Top-FO-Only of the rebuttal", marker="o")),
    ("Every client at the highest boundary (beta = 0): only the top segment first-order", None),
    ("Au24-b0", arm(ORANGE, "zeroth-order updates on (paper Table 9: 63.22)", SOLID)),
    ("Bu24-b0", arm(ORANGE, "bottom frozen", marker="o")),
    ("Every layer zeroth-order", None),
    ("decomfl", arm(GRAY, "DeComFL (paper: 55.83)", DOTTED)),
]


def style(spec, scale=1.0):
    line = dict(color=spec["color"], linewidth=2.0 * scale, linestyle=spec["ls"], solid_capstyle="round", zorder=3)
    if spec["marker"]:
        filled = spec["filled"]
        line.update(marker=spec["marker"], markersize=(6.0 if filled else 7.0) * scale, markeredgewidth=1.2 if filled else 1.6,
                    markerfacecolor=spec["color"] if filled else SURFACE,
                    markeredgecolor=spec["color"] if not filled or spec["marker"] == "x" else SURFACE, zorder=4 if filled else 5)
    return line


def read_curves(data):
    """{arm: {"seeds": {seed: [(round, accuracy, loss)]}, "mean": [...], "final": mean final accuracy}}."""
    curves = {}
    for run in data["runs"]:
        points = [(row["round"], row["accuracy_pct"], row["eval_loss"]) for row in run["curve"]]
        if points:
            curves.setdefault(run["summary"]["arm"], {"seeds": {}})["seeds"][run["summary"]["seed"]] = points
    for entry in curves.values():
        series = list(entry["seeds"].values())
        common = sorted(set.intersection(*(set(point[0] for point in points) for points in series)))
        lookup = [{point[0]: point for point in points} for points in series]
        entry["mean"] = [(r, sum(table[r][1] for table in lookup) / len(lookup), sum(table[r][2] for table in lookup) / len(lookup)) for r in common]
        entry["final"] = sum(points[-1][1] for points in series) / len(series)
    return curves


def draw_panel(axis, curves, rows, metric, title, window=None, skip=()):
    index = 1 if metric == "accuracy" else 2
    values = []
    for key, spec in rows:
        if spec is None or key not in curves or key in skip:
            continue
        for points in curves[key]["seeds"].values() if len(curves[key]["seeds"]) > 1 else []:
            axis.plot([p[0] for p in points], [p[index] for p in points], color=spec["color"], linewidth=0.8, alpha=0.3, zorder=2)
        points = curves[key]["mean"]
        axis.plot([p[0] for p in points], [p[index] for p in points], **style(spec))
        if window:
            values += [p[index] for p in points if window[0] <= p[0] <= window[1]]
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=12, color=INK, pad=9)
    axis.set_xlabel("round", fontsize=10.5, color=INK_2)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK_2, labelsize=9.5, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    if window and values:
        margin = 0.08 * (max(values) - min(values) or 1.0)
        axis.set_xlim(window[0] - 1, window[1] + 1)
        axis.set_ylim(min(values) - margin, max(values) + margin)


LADDER_NOTE = ("Solid = uniform sampling, dashed = biased sampling.\n"
               "Hollow markers without a line = the same arm with the zeroth-order updates off;\n"
               "where they sit on the line, switching them off changes nothing.\n"
               "With several seeds: mean over seeds, single seeds thin behind it.")
ZO_NOTE = ("Lines: zeroth-order updates on. Hollow markers without a line: the same run\n"
           "with the bottom frozen (zeroth-order updates off, direct averaging);\n"
           "where they sit on the line, the zeroth-order updates change nothing.")


def draw_legend(axis, curves, rows, last_column, note=LADDER_NOTE):
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(len(rows) + 3.0, -0.6)
    axis.text(0.0, -0.25, "Legend", fontsize=12, color=INK, va="center")
    axis.text(0.84, -0.25, "arm", fontsize=9, color=INK_2, va="center", ha="right")
    axis.text(1.0, -0.25, last_column, fontsize=9, color=INK_2, va="center", ha="right", linespacing=1.15)
    y = 0.75
    for key, spec in rows:
        if spec is None:
            y += 0.28
            axis.text(0.0, y, key, fontsize=10, color=INK, va="center", fontweight="bold")
            axis.plot([0, 1], [y + 0.48, y + 0.48], color=GRID, linewidth=0.8)
        else:
            sample = style(spec)
            xs = [0.010, 0.045, 0.080] if sample["linestyle"] != "none" else [0.045]
            axis.plot(xs, [y] * len(xs), **{**sample, "markevery": [len(xs) // 2]})
            axis.text(0.10, y, spec["text"], fontsize=9.5, color=INK if key in curves else GRAY, va="center")
            axis.text(0.84, y, key, fontsize=8.5, color=INK_2, va="center", ha="right", family="monospace")
            seeds = len(curves[key]["seeds"]) if key in curves else 0
            axis.text(1.0, y, f"{curves[key]['final']:.2f}" + (f" (n={seeds})" if seeds > 1 else "") if seeds else "not run",
                      fontsize=10, color=INK if seeds else GRAY, va="center", ha="right")
        y += 1.0
    axis.text(0.0, y + 0.3, note,
              fontsize=9, color=INK_2, va="top", linespacing=1.45)


def figure_for(curves, rows, title, panels, out, last_column, note=LADDER_NOTE):
    figure = plt.figure(figsize=(6.0 * len(panels) + 8.0, 6.6), facecolor=SURFACE)
    grid = figure.add_gridspec(1, len(panels) + 1, width_ratios=[1] * len(panels) + [1.38], left=0.04, right=0.995, top=0.86, bottom=0.10, wspace=0.17)
    for position, panel in enumerate(panels):
        draw_panel(figure.add_subplot(grid[0, position]), curves, rows, **panel)
    draw_legend(figure.add_subplot(grid[0, len(panels)]), curves, rows, last_column, note)
    figure.suptitle(title, x=0.04, y=0.965, ha="left", fontsize=14, color=INK)
    figure.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return out


def main(feedback_json, out_dir):
    data = json.loads(Path(feedback_json).read_text())
    curves, out_dir, written = read_curves(data), Path(out_dir), []
    if not curves:
        print("no completed run to plot")
        return written
    rounds = data["rounds"]
    header = f"{data['model']} / AG News (Dirichlet 0.5), seeds {data['seeds']}"
    late = (rounds - rounds * 3 // 8, rounds)
    written.append(figure_for(curves, LADDER, f"{header}, {rounds} rounds: the ablation ladder", [
        dict(metric="accuracy", title="Test accuracy (%), higher is better"),
        dict(metric="accuracy", title=f"Test accuracy (%), rounds {late[0]}-{late[1]} enlarged", window=late, skip=("decomfl",)),
        dict(metric="loss", title="Test loss, lower is better")], out_dir / "p1_curves.png", f"accuracy %\nat round {rounds}"))
    if any(key in curves for key in ("Au24-b0", "Bu24-b0")):
        written.append(figure_for(curves, ZO_CHECK, f"{header}: do the zeroth-order updates help?", [
            dict(metric="accuracy", title="Test accuracy (%), higher is better"),
            dict(metric="loss", title="Test loss, lower is better")], out_dir / "p1_zo_check.png", "final\naccuracy %", note=ZO_NOTE))
    if any(key in curves for key, spec in CONTROLS if spec and key not in dict(LADDER)):
        written.append(figure_for(curves, CONTROLS, f"{header}: controls (not proposed methods)", [
            dict(metric="accuracy", title=f"Test accuracy (%), {rounds}-round arms, rounds {late[0]}-{late[1]}", window=late, skip=LONG),
            dict(metric="accuracy", title="Test accuracy (%), arms with three times as many rounds", skip=tuple(k for k, s in CONTROLS if s and k not in LONG))],
            out_dir / "p1_controls.png", "final\naccuracy %"))
    for path in written:
        print("written:", path)
    return written


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
